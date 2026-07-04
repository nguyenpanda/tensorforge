/*
 * hpcsmith/native/01_cuda_gemm.cu
 * ===============================
 * CUDA C++ PyTorch extension implementing General Matrix Multiplication (GEMM).
 *
 * Provides two kernel variants:
 * 1. gemm_naive_kernel: Global memory access where each thread computes one output element.
 * 2. gemm_tiled_kernel: Shared memory tiling implementation reducing global memory bandwidth.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

/**
 * @brief Naive CUDA kernel for matrix multiplication C = A * B.
 *
 * Each thread reads a row of A and a column of B from global memory to compute
 * a single element of matrix C.
 *
 * @param A Pointer to matrix A in global memory (dimensions M x K).
 * @param B Pointer to matrix B in global memory (dimensions K x N).
 * @param C Pointer to output matrix C in global memory (dimensions M x N).
 * @param M Number of rows in matrix A and matrix C.
 * @param N Number of columns in matrix B and matrix C.
 * @param K Inner dimension (columns of A, rows of B).
 */
__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * @brief Tiled CUDA kernel utilizing shared memory for matrix multiplication C = A * B.
 *
 * Tiles matrices A and B into TILE_SIZE x TILE_SIZE sub-blocks in shared memory
 * to reuse loaded data across threads in the block, substantially reducing
 * global memory bandwidth traffic.
 *
 * @param A Pointer to matrix A in global memory (dimensions M x K).
 * @param B Pointer to matrix B in global memory (dimensions K x N).
 * @param C Pointer to output matrix C in global memory (dimensions M x N).
 * @param M Number of rows in matrix A and matrix C.
 * @param N Number of columns in matrix B and matrix C.
 * @param K Inner dimension (columns of A, rows of B).
 */
__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_k_a = t * TILE_SIZE + tx;
        int tiled_k_b = t * TILE_SIZE + ty;

        if (row < M && tiled_k_a < K) {
            As[ty][tx] = A[row * K + tiled_k_a];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (tiled_k_b < K && col < N) {
            Bs[ty][tx] = B[tiled_k_b * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * @brief Host wrapper executing CUDA GEMM kernels on 2D PyTorch float32 tensors.
 *
 * Validates input tensor dimensions, contiguity, device placement, and data type
 * before dispatching the requested CUDA matrix multiplication kernel.
 *
 * @param a First operand tensor of shape (M, K).
 * @param b Second operand tensor of shape (K, N).
 * @param use_tiled If true, invokes gemm_tiled_kernel; otherwise invokes gemm_naive_kernel.
 * @return torch::Tensor Output matrix C of shape (M, N).
 */
torch::Tensor run_gemm(torch::Tensor a, torch::Tensor b, bool use_tiled) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "run_gemm: inputs must be 2D tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "run_gemm: inputs must be float32 tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "run_gemm: inputs must be contiguous tensors");
    TORCH_CHECK(a.device() == b.device(), "run_gemm: inputs must be on the same device");
    TORCH_CHECK(a.is_cuda(), "run_gemm: inputs must be CUDA tensors");
    TORCH_CHECK(a.size(1) == b.size(0), "run_gemm: inner dimensions must match");

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    auto c = torch::empty({M, N}, a.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    if (use_tiled) {
        gemm_tiled_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    } else {
        gemm_naive_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    }

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TensorForge HPC Bridge — Lesson 01: CUDA GEMM";
    m.def(
        "run_gemm",
        &run_gemm,
        "General matrix multiplication using CUDA (naive or tiled).",
        py::arg("a"),
        py::arg("b"),
        py::arg("use_tiled") = false
    );
}
