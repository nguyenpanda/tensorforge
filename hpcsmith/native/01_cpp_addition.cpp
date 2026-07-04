/*
 * hpcsmith/native/01_cpp_addition.cpp
 * ===================================
 * PyTorch C++ extension: element-wise tensor addition.
 *
 * Lesson: hpcsmith/basic/01_cpp_integration
 *
 * Entry point for the TensorForge HPC Bridge curriculum. Demonstrates the
 * minimal structure required to expose a C++ function to Python via
 * ``torch.utils.cpp_extension.load``. No CMake or manual build step needed;
 * ``CudaJitBackend`` compiles this file on-the-fly at lesson runtime.
 */

#include <torch/extension.h>

/// Returns the element-wise sum of \p a and \p b via ATen's operator+.
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device() == b.device(), "add_tensors: operands must be on the same device");
    TORCH_CHECK(a.dtype() == b.dtype(), "add_tensors: operands must have the same dtype");
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TensorForge HPC Bridge — Lesson 01: C++ tensor addition";
    m.def(
        "add_tensors",
        &add_tensors,
        "Element-wise addition of two tensors via a C++ ATen kernel.",
        py::arg("a"),
        py::arg("b")
    );
}
