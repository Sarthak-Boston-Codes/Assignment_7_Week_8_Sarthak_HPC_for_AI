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

// Computing one element
__global__ void matmul_A_Bt_kernel(const float* A, const float* B_local, float* C,
                                   int T, int D, int S_local)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < T && col < S_local) {
        float sum = 0.0f;
     
        for (int d = 0; d < D; ++d) {
            sum += A[row * D + d] * B_local[col * D + d];
        }
        C[row * S_local + col] = sum;
    }
}

// Computing one element
__global__ void matmul_W_V_kernel(const float* W_local, const float* V_local, float* O_partial,
                                  int T, int S_local, int D)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int d   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < T && d < D) {
        float sum = 0.0f;
       
        for (int s = 0; s < S_local; ++s) {
            sum += W_local[row * S_local + s] * V_local[s * D + d];
        }
        O_partial[row * D + d] = sum;
    }
}

// Hosting softmax on a vector 
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

    // Partition tokens across GPUs
    std::vector<int> S(num_gpus);
    int base = T / num_gpus;
    int rem = T % num_gpus;
    for (int g = 0; g < num_gpus; ++g) {
        S[g] = base + (g < rem ? 1 : 0);
    }

    std::vector<int> offset(num_gpus);
    offset[0] = 0;
    for (int g = 1; g < num_gpus; ++g) offset[g] = offset[g-1] + S[g-1];

    size_t bytes_TD = (size_t)T * D * sizeof(float);
    float* h_Q = (float*)malloc(bytes_TD);
    float* h_K = (float*)malloc(bytes_TD);
    float* h_V = (float*)malloc(bytes_TD);
    float* h_O = (float*)malloc(bytes_TD); 

    if (!h_Q || !h_K || !h_V || !h_O) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Initialize random 
    srand(12345);
    for (int i = 0; i < T * D; ++i) {

        h_Q[i] = ((float)(rand() % 1000) / 1000.0f) - 0.5f;
        h_K[i] = ((float)(rand() % 1000) / 1000.0f) - 0.5f;
        h_V[i] = ((float)(rand() % 1000) / 1000.0f) - 0.5f;
        h_O[i] = 0.0f;
    }

    // Per-GPU resources
    struct GPUState {
        int dev;
        float* d_Q;      
        float* d_Klocal; 
        float* d_Vlocal; 
        float* d_scores; 
        float* h_scores; 
        float* d_Wlocal; 
        float* d_Opartial; 
        float* h_Opartial; 
    };

    std::vector<GPUState> gpus(num_gpus);

    // Allocating per GPU memory
    for (int g = 0; g < num_gpus; ++g) {
        int dev = g; 
        gpus[g].dev = dev;
        CHECK_CUDA(cudaSetDevice(dev));

        int Sg = S[g];
        size_t bytes_Klocal = (size_t)Sg * D * sizeof(float);
        size_t bytes_scores = (size_t)T * Sg * sizeof(float);
        size_t bytes_Wlocal = bytes_scores;
        size_t bytes_Opartial = bytes_TD; // T x D

        // Allocating device buffers
        CHECK_CUDA(cudaMalloc(&gpus[g].d_Q, bytes_TD));            
        CHECK_CUDA(cudaMalloc(&gpus[g].d_Klocal, bytes_Klocal));
        CHECK_CUDA(cudaMalloc(&gpus[g].d_Vlocal, bytes_Klocal));  
        CHECK_CUDA(cudaMalloc(&gpus[g].d_scores, bytes_scores));  
        CHECK_CUDA(cudaMalloc(&gpus[g].d_Wlocal, bytes_Wlocal));  
        CHECK_CUDA(cudaMalloc(&gpus[g].d_Opartial, bytes_Opartial)); 

        CHECK_CUDA(cudaMallocHost(&gpus[g].h_scores, bytes_scores));
        gpus[g].h_Opartial = (float*)malloc(bytes_Opartial);
        if (!gpus[g].h_Opartial) { fprintf(stderr, "host malloc partial failed\n"); exit(1); }

        CHECK_CUDA(cudaMemcpy(gpus[g].d_Q, h_Q, bytes_TD, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMemcpy(gpus[g].d_Klocal, h_K + offset[g] * D, bytes_Klocal, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpus[g].d_Vlocal, h_V + offset[g] * D, bytes_Klocal, cudaMemcpyHostToDevice));
    }

 
    // Launch kernels:
    dim3 blockScores(16, 16);
    for (int g = 0; g < num_gpus; ++g) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        int Sg = S[g];
        dim3 grid((Sg + blockScores.x - 1) / blockScores.x, (T + blockScores.y - 1) / blockScores.y);
        matmul_A_Bt_kernel<<<grid, blockScores>>>(gpus[g].d_Q, gpus[g].d_Klocal, gpus[g].d_scores, T, D, Sg);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        size_t bytes_scores = (size_t)T * Sg * sizeof(float);
        CHECK_CUDA(cudaMemcpy(gpus[g].h_scores, gpus[g].d_scores, bytes_scores, cudaMemcpyDeviceToHost));
    }

    // Assembly global scores on host
    float* tmp_row = (float*)malloc(T * sizeof(float));
    if (!tmp_row) { fprintf(stderr, "tmp_row malloc failed\n"); exit(1); }

    
    for (int row = 0; row < T; ++row) {

        int idx = 0;
        for (int g = 0; g < num_gpus; ++g) {
            int Sg = S[g];

            float* src = gpus[g].h_scores + (size_t)row * Sg;
            for (int s = 0; s < Sg; ++s) {
                tmp_row[idx++] = src[s];
            }
        }

        softmax_inplace(tmp_row, T);


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

    // GPU per weight buffers
    for (int g = 0; g < num_gpus; ++g) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        size_t bytes_scores = (size_t)T * S[g] * sizeof(float);
        CHECK_CUDA(cudaMemcpy(gpus[g].d_Wlocal, gpus[g].h_scores, bytes_scores, cudaMemcpyHostToDevice));
    }

    // Computing partial outputs on GPU
    dim3 blockOut(16, 16);
    for (int g = 0; g < num_gpus; ++g) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        int Sg = S[g];
        dim3 gridOut((D + blockOut.x - 1) / blockOut.x, (T + blockOut.y - 1) / blockOut.y);
        matmul_W_V_kernel<<<gridOut, blockOut>>>(gpus[g].d_Wlocal, gpus[g].d_Vlocal, gpus[g].d_Opartial, T, Sg, D);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        size_t bytes_Opartial = (size_t)T * D * sizeof(float);
        CHECK_CUDA(cudaMemcpy(gpus[g].h_Opartial, gpus[g].d_Opartial, bytes_Opartial, cudaMemcpyDeviceToHost));
    }

    // Sum partial across GPU's
    for (int i = 0; i < T * D; ++i) h_O[i] = 0.0f;
    for (int g = 0; g < num_gpus; ++g) {
        for (int i = 0; i < T * D; ++i) {
            h_O[i] += gpus[g].h_Opartial[i];
        }
    }

    printf("\nDistributed FlashAttention Forward Output Sample O[0..7]:\n");
    for (int i = 0; i < 8 && i < T * D; ++i) {
        printf(" %9.6f", h_O[i]);
    }
    printf(" \n");

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
