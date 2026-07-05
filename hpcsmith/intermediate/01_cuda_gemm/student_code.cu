/*
 * student_code.cu — Lesson 01: CUDA GEMM
 * ======================================
 * CUDA C++ PyTorch extension implementing General Matrix Multiplication (GEMM).
 *
 * Provides two kernel variants:
 * 1. gemm_naive_kernel: Global memory access where each thread computes one output element.
 * 2. gemm_tiled_kernel: Shared memory tiling implementation reducing global memory bandwidth.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_SIZE 16

/**
 * @brief Naive CUDA kernel for matrix multiplication C = A * B.
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
    /* TODO: Implement CUDA Kernel */
}

/**
 * @brief Tiled CUDA kernel utilizing shared memory for matrix multiplication C = A * B.
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
    /* TODO: Implement CUDA Kernel */
}

/**
 * @brief Host wrapper executing CUDA GEMM kernels on 2D PyTorch float32 tensors.
 *
 * @param a First operand tensor of shape (M, K).
 * @param b Second operand tensor of shape (K, N).
 * @param use_tiled If true, invokes gemm_tiled_kernel; otherwise invokes gemm_naive_kernel.
 * @return torch::Tensor Output matrix C of shape (M, N).
 */
torch::Tensor run_gemm(torch::Tensor a, torch::Tensor b, bool use_tiled) {
    throw std::runtime_error("TODO: Implement this C++ function");
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
