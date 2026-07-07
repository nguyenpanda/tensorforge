"""
hint_01_cuda_gemm.py
====================
Hints for HPCSmith tier ``intermediate``, lesson ``01_cuda_gemm``.
"""

MODULE = "intermediate/01_cuda_gemm"

HINT = {
    "CudaGemm": {
        "run_naive_gemm": {
            "major": (
                "Write your C++/CUDA logic in student_code.cu. The Python file is only used to compile your code.\n"
                "In student_code.cu, calculate global row and column index from blockIdx and threadIdx.\n"
                "Each thread computes one output element C[row * N + col] by iterating over inner dimension K."
            ),
            "minor": (
                "row = blockIdx.y * blockDim.y + threadIdx.y;\n"
                "col = blockIdx.x * blockDim.x + threadIdx.x;\n"
                "Accumulate A[row * K + k] * B[k * N + col] in a float register before writing to C."
            ),
        },
        "run_tiled_gemm": {
            "major": (
                "Write your C++/CUDA logic in student_code.cu. The Python file is only used to compile your code.\n"
                "Use 2D shared memory arrays __shared__ float As[TILE_SIZE][TILE_SIZE] and Bs[TILE_SIZE][TILE_SIZE].\n"
                "Iterate over tiles across dimension K, loading sub-matrices into shared memory."
            ),
            "minor": (
                "Always call __syncthreads() after loading tile data into shared memory before computing dot products,\n"
                "and again before loading the next tile from global memory."
            ),
        },
    },
}
