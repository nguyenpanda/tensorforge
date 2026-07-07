/*
 * student_code.cpp — Lesson 01: C++ Integration
 * =============================================
 * PyTorch C++ extension: element-wise tensor addition.
 */

#include <torch/extension.h>
#include <stdexcept>

/// Returns the element-wise sum of \p a and \p b via ATen's operator+.
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    //  This is just an example to make sure the C++ integration works.
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
