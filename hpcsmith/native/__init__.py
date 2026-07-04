"""
hpcsmith/native/__init__.py
===========================
Package marker for the TensorForge HPC native-code workspace.

Students place their raw C++ (``*.cpp``) and CUDA (``*.cu``) kernel source files
directly in this directory. The :class:`~forge_core.backends.cuda_backend.CudaJitBackend`
locates and JIT-compiles those files on-the-fly via
``torch.utils.cpp_extension.load``.
"""
