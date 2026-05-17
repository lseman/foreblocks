#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> selective_scan_cuda_fwd(
    torch::Tensor u,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor Dskip
);

std::vector<torch::Tensor> selective_scan_cuda_bwd(
    torch::Tensor u,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor Dskip,
    torch::Tensor grad_y,
    torch::Tensor states
);

std::tuple<torch::Tensor, torch::Tensor> selective_scan_fwd(
    torch::Tensor u,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor Dskip
) {
    TORCH_CHECK(u.is_cuda(), "u must be CUDA");
    TORCH_CHECK(dt.is_cuda(), "dt must be CUDA");
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");
    TORCH_CHECK(C.is_cuda(), "C must be CUDA");
    TORCH_CHECK(Dskip.is_cuda(), "Dskip must be CUDA");
    return selective_scan_cuda_fwd(u, dt, A, B, C, Dskip);
}

std::vector<torch::Tensor> selective_scan_bwd(
    torch::Tensor u,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor Dskip,
    torch::Tensor grad_y,
    torch::Tensor states
) {
    TORCH_CHECK(u.is_cuda(), "u must be CUDA");
    TORCH_CHECK(dt.is_cuda(), "dt must be CUDA");
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");
    TORCH_CHECK(C.is_cuda(), "C must be CUDA");
    TORCH_CHECK(Dskip.is_cuda(), "Dskip must be CUDA");
    TORCH_CHECK(grad_y.is_cuda(), "grad_y must be CUDA");
    TORCH_CHECK(states.is_cuda(), "states must be CUDA");
    return selective_scan_cuda_bwd(u, dt, A, B, C, Dskip, grad_y, states);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selective_scan_fwd", &selective_scan_fwd, "Selective scan forward (CUDA)");
    m.def("selective_scan_bwd", &selective_scan_bwd, "Selective scan backward (CUDA)");
}
