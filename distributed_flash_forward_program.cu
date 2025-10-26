// dist_flash_forward_program.cu
// Distributed baseline forward-only single-head attention (educational, professor-aligned).
// Splits keys/values across GPUs (by contiguous token blocks).
// Strategy (baseline, correct, easy-to-reason-about):
// 1) Host creates Q, K, V (shape T x D).
// 2) Partition K and V across G GPUs by token blocks: each GPU owns S_local tokens.
// 3) For each GPU:
//      - Copy Q (full) and K_local, V_local to GPU.
//      - Compute local scores: S_local = Q (T x D) * K_local^T (D x S_local) -> (T x S_local).
//      - Copy S_local back to host (host assembles full score row).
// 4) Host computes global softmax per query row over the concatenated scores (dimension T).
//      (This centralizes the softmax for correctness and stability.)
// 5) Host sends per-GPU weight chunks back to GPUs.
// 6) Each GPU computes partial outputs: O_partial = W_local (T x S_local) * V_local (S_local x D) -> (T x D).
// 7) Host gathers partial outputs from GPUs and sums them to produce final O (T x D).
//
// Notes:
//  - This is a baseline *distributed* forward pass that demonstrates splitting experts/tokens across multiple GPUs.
//  - It is intentionally simple and professor-aligned: correctness prioritized and code is heavily commented for clarity.
//  - Performance: heavy host-device transfers (naive all-gather/dispatch). For production, you'd implement
//    GPU-native all-reduce/all-gather and in-GPU softmax (or use NCCL/NVSHMEM), and fuse kernels.
//  - This file compiles with nvcc. Example compile:
//      nvcc -O3 -arch=sm_86 dist_flash_forward_program.cu -o dist_flash_forward_program
//
// Usage:
//    ./dist_flash_forward_program <num_gpus> <sequence_length_T> <head_dim_D>
// Example:
//    ./dist_flash_forward_program 1 1024 64
//    ./dist_flash_forward_program 2 2048 128
//
// Author: Sarthak Mahesh Sargar (student-friendly, professor-aligned baseline)
// Date: 2025

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>
#include <chrono>

#define CHECK_CUDA(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1);                                              \
    }                                                         \
} while (0)

// Simple kernel: compute C = A (T x D) * B^T (D x S) -> C (T x S)
// Each thread computes one element C[row, col]
__global__ void matmul_A_Bt_kernel(const float* A, const float* B_local, float* C,
                                   int T, int D, int S_local)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < T && col < S_local) {
        float sum = 0.0f;
        // A is row-major T x D
        // B_local is row-major S_local x D, but we want B_local^T, so index appropriately:
        // B_local (r, d) stored at B_local[r * D + d]
        for (int d = 0; d < D; ++d) {
            sum += A[row * D + d] * B_local[col * D + d];
        }
        C[row * S_local + col] = sum;
    }
}

// Kernel: compute O_partial = W_local (T x S_local) * V_local (S_local x D) -> O_partial (T x D)
// Each thread computes one element O[row, d]
__global__ void matmul_W_V_kernel(const float* W_local, const float* V_local, float* O_partial,
                                  int T, int S_local, int D)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int d   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < T && d < D) {
        float sum = 0.0f;
        // W_local row-major T x S_local
        // V_local row-major S_local x D
        for (int s = 0; s < S_local; ++s) {
            sum += W_local[row * S_local + s] * V_local[s * D + d];
        }
        O_partial[row * D + d] = sum;
    }
}

// Host softmax on a vector (in place), numerically stable
static void softmax_inplace(float* row, int len) {
    float maxv = row[0];
    for (int i = 1; i < len; ++i) if (row[i] > maxv) maxv = row[i];
    double sum = 0.0;
    for (int i = 0; i < len; ++i) {
        row[i] = expf(row[i] - maxv);
        sum += row[i];
    }
    float inv = 1.0f / (float)sum;
    for (int i = 0; i < len; ++i) row[i] *= inv;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <num_gpus> <T sequence length> <D head dim>\n", argv[0]);
        return 1;
    }

    int num_gpus = atoi(argv[1]);
    int T = atoi(argv[2]);
    int D = atoi(argv[3]);

    // Validate
    int available = 0;
    CHECK_CUDA(cudaGetDeviceCount(&available));
    if (num_gpus < 1 || num_gpus > available || num_gpus > 8) {
        fprintf(stderr, "Requested %d GPUs but %d available; choose 1..min(8,%d)\n", num_gpus, available, available);
        return 1;
    }

    printf("Distributed FlashAttention forward baseline\n");
    printf("GPUs: %d, Sequence length T: %d, head dim D: %d\n", num_gpus, T, D);

    // Partition tokens across GPUs: S_local roughly = ceil(T / num_gpus) for first partitions
    std::vector<int> S(num_gpus);
    int base = T / num_gpus;
    int rem = T % num_gpus;
    for (int g = 0; g < num_gpus; ++g) {
        S[g] = base + (g < rem ? 1 : 0);
    }
    // compute start offsets
    std::vector<int> offset(num_gpus);
    offset[0] = 0;
    for (int g = 1; g < num_gpus; ++g) offset[g] = offset[g-1] + S[g-1];

    // Host allocate Q,K,V (T x D)
    size_t bytes_TD = (size_t)T * D * sizeof(float);
    float* h_Q = (float*)malloc(bytes_TD);
    float* h_K = (float*)malloc(bytes_TD);
    float* h_V = (float*)malloc(bytes_TD);
    float* h_O = (float*)malloc(bytes_TD); // final output

    if (!h_Q || !h_K || !h_V || !h_O) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Initialize random (fixed seed for reproducibility)
    srand(12345);
    for (int i = 0; i < T * D; ++i) {
        // small values for stability
        h_Q[i] = ((float)(rand() % 1000) / 1000.0f) - 0.5f;
        h_K[i] = ((float)(rand() % 1000) / 1000.0f) - 0.5f;
        h_V[i] = ((float)(rand() % 1000) / 1000.0f) - 0.5f;
        h_O[i] = 0.0f;
    }

    // Per-GPU resources (host side)
    struct GPUState {
        int dev;
        float* d_Q;      // full Q on device
        float* d_Klocal; // K chunk
        float* d_Vlocal; // V chunk
        float* d_scores; // T x S_local
        float* h_scores; // pinned host buffer for scores (T x S_local)
        float* d_Wlocal; // weights (T x S_local) on device
        float* d_Opartial; // partial output (T x D)
        float* h_Opartial; // host buffer for partial (T x D)
    };

    std::vector<GPUState> gpus(num_gpus);

    // allocate per-GPU memory and copy necessary data
    for (int g = 0; g < num_gpus; ++g) {
        int dev = g; // assume devices numbered 0..num_gpus-1 accessible
        gpus[g].dev = dev;
        CHECK_CUDA(cudaSetDevice(dev));

        int Sg = S[g];
        size_t bytes_Klocal = (size_t)Sg * D * sizeof(float);
        size_t bytes_scores = (size_t)T * Sg * sizeof(float);
        size_t bytes_Wlocal = bytes_scores;
        size_t bytes_Opartial = bytes_TD; // T x D

        // allocate device buffers
        CHECK_CUDA(cudaMalloc(&gpus[g].d_Q, bytes_TD));            // full Q
        CHECK_CUDA(cudaMalloc(&gpus[g].d_Klocal, bytes_Klocal));  // local K
        CHECK_CUDA(cudaMalloc(&gpus[g].d_Vlocal, bytes_Klocal));  // local V (Sg x D)
        CHECK_CUDA(cudaMalloc(&gpus[g].d_scores, bytes_scores));  // local scores
        CHECK_CUDA(cudaMalloc(&gpus[g].d_Wlocal, bytes_Wlocal));  // local weights (after softmax)
        CHECK_CUDA(cudaMalloc(&gpus[g].d_Opartial, bytes_Opartial)); // partial output T x D

        // pinned host buffers for transfers
        CHECK_CUDA(cudaMallocHost(&gpus[g].h_scores, bytes_scores));
        gpus[g].h_Opartial = (float*)malloc(bytes_Opartial);
        if (!gpus[g].h_Opartial) { fprintf(stderr, "host malloc partial failed\n"); exit(1); }

        // copy full Q
        CHECK_CUDA(cudaMemcpy(gpus[g].d_Q, h_Q, bytes_TD, cudaMemcpyHostToDevice));
        // copy local K and V chunk from host
        CHECK_CUDA(cudaMemcpy(gpus[g].d_Klocal, h_K + offset[g] * D, bytes_Klocal, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpus[g].d_Vlocal, h_V + offset[g] * D, bytes_Klocal, cudaMemcpyHostToDevice));
    }

    // Step 1: per-GPU compute local scores = Q * K_local^T
    // We'll compute on each GPU and copy local scores to host (h_scores).
    // Launch kernels:
    dim3 blockScores(16, 16);
    for (int g = 0; g < num_gpus; ++g) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        int Sg = S[g];
        dim3 grid((Sg + blockScores.x - 1) / blockScores.x, (T + blockScores.y - 1) / blockScores.y);
        matmul_A_Bt_kernel<<<grid, blockScores>>>(gpus[g].d_Q, gpus[g].d_Klocal, gpus[g].d_scores, T, D, Sg);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        // copy scores to pinned host buffer
        size_t bytes_scores = (size_t)T * Sg * sizeof(float);
        CHECK_CUDA(cudaMemcpy(gpus[g].h_scores, gpus[g].d_scores, bytes_scores, cudaMemcpyDeviceToHost));
    }

    // Step 2: assemble global scores on host and softmax per query row (over length T)
    // We'll build a temp array per row of length T by concatenating per-gpu segments.
    // For memory efficiency, reuse a single row buffer of length T.
    float* tmp_row = (float*)malloc(T * sizeof(float));
    if (!tmp_row) { fprintf(stderr, "tmp_row malloc failed\n"); exit(1); }

    // We will produce weights_w (same shape as scores full): but instead of assembling the entire big matrix
    // we will compute softmax per row into per-gpu host buffers and then copy per-gpu weight chunks back.
    // Prepare per-gpu host weight buffers (pinned) - reuse h_scores to store weights to copy back.
    for (int row = 0; row < T; ++row) {
        // gather row across GPUs
        int idx = 0;
        for (int g = 0; g < num_gpus; ++g) {
            int Sg = S[g];
            // h_scores is T x Sg (row-major). copy row'th slice
            float* src = gpus[g].h_scores + (size_t)row * Sg;
            for (int s = 0; s < Sg; ++s) {
                tmp_row[idx++] = src[s];
            }
        }
        // softmax over idx==T (the full K dimension)
        softmax_inplace(tmp_row, T);

        // now scatter weights back into per-gpu h_scores (reuse the same pinned area)
        idx = 0;
        for (int g = 0; g < num_gpus; ++g) {
            int Sg = S[g];
            float* dst = gpus[g].h_scores + (size_t)row * Sg;
            for (int s = 0; s < Sg; ++s) {
                dst[s] = tmp_row[idx++];
            }
        }
    }

    free(tmp_row);

    // Step 3: copy per-gpu weight buffers (now in h_scores) back to device d_Wlocal
    for (int g = 0; g < num_gpus; ++g) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        size_t bytes_scores = (size_t)T * S[g] * sizeof(float);
        CHECK_CUDA(cudaMemcpy(gpus[g].d_Wlocal, gpus[g].h_scores, bytes_scores, cudaMemcpyHostToDevice));
    }

    // Step 4: compute partial outputs on each GPU: O_partial = W_local (T x Sg) * V_local (Sg x D)
    dim3 blockOut(16, 16);
    for (int g = 0; g < num_gpus; ++g) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        int Sg = S[g];
        dim3 gridOut((D + blockOut.x - 1) / blockOut.x, (T + blockOut.y - 1) / blockOut.y);
        matmul_W_V_kernel<<<gridOut, blockOut>>>(gpus[g].d_Wlocal, gpus[g].d_Vlocal, gpus[g].d_Opartial, T, Sg, D);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        // copy partial back to host
        size_t bytes_Opartial = (size_t)T * D * sizeof(float);
        CHECK_CUDA(cudaMemcpy(gpus[g].h_Opartial, gpus[g].d_Opartial, bytes_Opartial, cudaMemcpyDeviceToHost));
    }

    // Step 5: sum partials across GPUs on host to produce final O
    for (int i = 0; i < T * D; ++i) h_O[i] = 0.0f;
    for (int g = 0; g < num_gpus; ++g) {
        for (int i = 0; i < T * D; ++i) {
            h_O[i] += gpus[g].h_Opartial[i];
        }
    }

    // Print sample output O[0..7] (first 8 values of O flattened row-major)
    printf("\nDistributed FlashAttention Forward Output Sample O[0..7]:\n");
    for (int i = 0; i < 8 && i < T * D; ++i) {
        printf(" %9.6f", h_O[i]);
    }
    printf(" \n");

    // cleanup
    for (int g = 0; g < num_gpus; ++g) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        cudaFree(gpus[g].d_Q);
        cudaFree(gpus[g].d_Klocal);
        cudaFree(gpus[g].d_Vlocal);
        cudaFree(gpus[g].d_scores);
        cudaFree(gpus[g].d_Wlocal);
        cudaFree(gpus[g].d_Opartial);
        cudaFreeHost(gpus[g].h_scores);
        free(gpus[g].h_Opartial);
    }

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);

    return 0;
}
