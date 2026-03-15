#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void selective_scan_fwd_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ dt,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ Bpar,
    const scalar_t* __restrict__ Cpar,
    const scalar_t* __restrict__ Dskip,
    scalar_t* __restrict__ y,
    float* __restrict__ states,
    int Bsz,
    int T,
    int D,
    int N
) {
    int bd = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Bsz * D;
    if (bd >= total) return;

    int b = bd / D;
    int d = bd % D;

    extern __shared__ float shmem[];
    float* state = shmem + threadIdx.x * N;

    for (int n = 0; n < N; ++n) {
        state[n] = 0.0f;
    }

    const int u_base = b * T * D + d;
    const int bc_base = ((b * T) * D + d) * N;
    const int state_base = ((b * T) * D + d) * N;

    for (int t = 0; t < T; ++t) {
        int u_idx = u_base + t * D;
        int bc_t_base = bc_base + t * D * N;
        int state_t_base = state_base + t * D * N;

        float u_t = static_cast<float>(u[u_idx]);
        float dt_t = static_cast<float>(dt[u_idx]);
        float out_t = 0.0f;

        for (int n = 0; n < N; ++n) {
            int a_idx = d * N + n;
            int bc_idx = bc_t_base + n;

            float a = static_cast<float>(A[a_idx]);
            float bpar = static_cast<float>(Bpar[bc_idx]);
            float cpar = static_cast<float>(Cpar[bc_idx]);
            float abar = expf(dt_t * a);

            state[n] = abar * state[n] + dt_t * bpar * u_t;
            states[state_t_base + n] = state[n];
            out_t += cpar * state[n];
        }

        out_t += static_cast<float>(Dskip[d]) * u_t;
        y[u_idx] = static_cast<scalar_t>(out_t);
    }
}


template <typename scalar_t>
__global__ void selective_scan_bwd_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ dt,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ Bpar,
    const scalar_t* __restrict__ Cpar,
    const scalar_t* __restrict__ Dskip,
    const scalar_t* __restrict__ grad_y,
    const float* __restrict__ states,
    scalar_t* __restrict__ du,
    scalar_t* __restrict__ ddt,
    scalar_t* __restrict__ dBpar,
    scalar_t* __restrict__ dCpar,
    float* __restrict__ dA_accum,
    float* __restrict__ dDskip_accum,
    int Bsz,
    int T,
    int D,
    int N
) {
    int bd = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Bsz * D;
    if (bd >= total) return;

    int b = bd / D;
    int d = bd % D;

    extern __shared__ float shmem[];
    float* grad_state = shmem + threadIdx.x * N;

    for (int n = 0; n < N; ++n) {
        grad_state[n] = 0.0f;
    }

    const int u_base = b * T * D + d;
    const int bc_base = ((b * T) * D + d) * N;
    const int state_base = ((b * T) * D + d) * N;
    const float dskip = static_cast<float>(Dskip[d]);

    for (int t = T - 1; t >= 0; --t) {
        const int u_idx = u_base + t * D;
        const int bc_t_base = bc_base + t * D * N;
        const int state_t_base = state_base + t * D * N;
        const int prev_state_base = state_t_base - D * N;

        const float u_t = static_cast<float>(u[u_idx]);
        const float dt_t = static_cast<float>(dt[u_idx]);
        const float grad_out_t = static_cast<float>(grad_y[u_idx]);

        float du_acc = grad_out_t * dskip;
        float ddt_acc = 0.0f;

        atomicAdd(&dDskip_accum[d], grad_out_t * u_t);

        for (int n = 0; n < N; ++n) {
            const int bc_idx = bc_t_base + n;
            const float cpar = static_cast<float>(Cpar[bc_idx]);
            const float state_t = states[state_t_base + n];

            dCpar[bc_idx] = static_cast<scalar_t>(grad_out_t * state_t);
            grad_state[n] += grad_out_t * cpar;
        }

        for (int n = 0; n < N; ++n) {
            const int a_idx = d * N + n;
            const int bc_idx = bc_t_base + n;

            const float a = static_cast<float>(A[a_idx]);
            const float bpar = static_cast<float>(Bpar[bc_idx]);
            const float grad_state_t = grad_state[n];
            const float prev_state = (t == 0) ? 0.0f : states[prev_state_base + n];
            const float abar = expf(dt_t * a);
            const float d_abar = grad_state_t * prev_state;
            const float d_temp = d_abar * abar;

            dBpar[bc_idx] = static_cast<scalar_t>(grad_state_t * dt_t * u_t);
            du_acc += grad_state_t * dt_t * bpar;
            ddt_acc += grad_state_t * bpar * u_t + d_temp * a;
            atomicAdd(&dA_accum[a_idx], d_temp * dt_t);

            grad_state[n] = grad_state_t * abar;
        }

        du[u_idx] = static_cast<scalar_t>(du_acc);
        ddt[u_idx] = static_cast<scalar_t>(ddt_acc);
    }
}


std::tuple<torch::Tensor, torch::Tensor> selective_scan_cuda_fwd(
    torch::Tensor u,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor Bpar,
    torch::Tensor Cpar,
    torch::Tensor Dskip
) {
    CHECK_INPUT(u);
    CHECK_INPUT(dt);
    CHECK_INPUT(A);
    CHECK_INPUT(Bpar);
    CHECK_INPUT(Cpar);
    CHECK_INPUT(Dskip);

    TORCH_CHECK(u.dim() == 3, "u must have shape [B,T,D]");
    TORCH_CHECK(dt.dim() == 3, "dt must have shape [B,T,D]");
    TORCH_CHECK(A.dim() == 2, "A must have shape [D,N]");
    TORCH_CHECK(Bpar.dim() == 4, "B must have shape [B,T,D,N]");
    TORCH_CHECK(Cpar.dim() == 4, "C must have shape [B,T,D,N]");
    TORCH_CHECK(Dskip.dim() == 1, "Dskip must have shape [D]");

    int Bsz = u.size(0);
    int T = u.size(1);
    int D = u.size(2);
    int N = A.size(1);

    TORCH_CHECK(
        dt.size(0) == Bsz && dt.size(1) == T && dt.size(2) == D,
        "dt shape mismatch"
    );
    TORCH_CHECK(A.size(0) == D, "A shape mismatch");
    TORCH_CHECK(
        Bpar.size(0) == Bsz && Bpar.size(1) == T && Bpar.size(2) == D && Bpar.size(3) == N,
        "B shape mismatch"
    );
    TORCH_CHECK(
        Cpar.size(0) == Bsz && Cpar.size(1) == T && Cpar.size(2) == D && Cpar.size(3) == N,
        "C shape mismatch"
    );
    TORCH_CHECK(Dskip.size(0) == D, "Dskip shape mismatch");

    auto y = torch::zeros_like(u);
    auto states = torch::zeros({Bsz, T, D, N}, u.options().dtype(torch::kFloat));

    const int threads = 128;
    const int total = Bsz * D;
    const int blocks = (total + threads - 1) / threads;
    const size_t shmem = threads * N * sizeof(float);

    auto stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        u.scalar_type(),
        "selective_scan_cuda_fwd",
        ([&] {
            selective_scan_fwd_kernel<scalar_t><<<blocks, threads, shmem, stream>>>(
                u.data_ptr<scalar_t>(),
                dt.data_ptr<scalar_t>(),
                A.data_ptr<scalar_t>(),
                Bpar.data_ptr<scalar_t>(),
                Cpar.data_ptr<scalar_t>(),
                Dskip.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                states.data_ptr<float>(),
                Bsz, T, D, N
            );
        })
    );

    return std::make_tuple(y, states);
}


std::vector<torch::Tensor> selective_scan_cuda_bwd(
    torch::Tensor u,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor Bpar,
    torch::Tensor Cpar,
    torch::Tensor Dskip,
    torch::Tensor grad_y,
    torch::Tensor states
) {
    CHECK_INPUT(u);
    CHECK_INPUT(dt);
    CHECK_INPUT(A);
    CHECK_INPUT(Bpar);
    CHECK_INPUT(Cpar);
    CHECK_INPUT(Dskip);
    CHECK_INPUT(grad_y);
    CHECK_INPUT(states);

    TORCH_CHECK(states.scalar_type() == torch::kFloat, "states must be float32");

    int Bsz = u.size(0);
    int T = u.size(1);
    int D = u.size(2);
    int N = A.size(1);

    TORCH_CHECK(grad_y.sizes() == u.sizes(), "grad_y shape mismatch");
    TORCH_CHECK(
        states.size(0) == Bsz && states.size(1) == T && states.size(2) == D && states.size(3) == N,
        "states shape mismatch"
    );

    auto du = torch::zeros_like(u);
    auto ddt = torch::zeros_like(dt);
    auto dBpar = torch::zeros_like(Bpar);
    auto dCpar = torch::zeros_like(Cpar);
    auto dA_accum = torch::zeros({D, N}, A.options().dtype(torch::kFloat));
    auto dDskip_accum = torch::zeros({D}, Dskip.options().dtype(torch::kFloat));

    const int threads = 128;
    const int total = Bsz * D;
    const int blocks = (total + threads - 1) / threads;
    const size_t shmem = threads * N * sizeof(float);

    auto stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        u.scalar_type(),
        "selective_scan_cuda_bwd",
        ([&] {
            selective_scan_bwd_kernel<scalar_t><<<blocks, threads, shmem, stream>>>(
                u.data_ptr<scalar_t>(),
                dt.data_ptr<scalar_t>(),
                A.data_ptr<scalar_t>(),
                Bpar.data_ptr<scalar_t>(),
                Cpar.data_ptr<scalar_t>(),
                Dskip.data_ptr<scalar_t>(),
                grad_y.data_ptr<scalar_t>(),
                states.data_ptr<float>(),
                du.data_ptr<scalar_t>(),
                ddt.data_ptr<scalar_t>(),
                dBpar.data_ptr<scalar_t>(),
                dCpar.data_ptr<scalar_t>(),
                dA_accum.data_ptr<float>(),
                dDskip_accum.data_ptr<float>(),
                Bsz, T, D, N
            );
        })
    );

    auto dA = dA_accum.to(A.scalar_type());
    auto dDskip = dDskip_accum.to(Dskip.scalar_type());
    return {du, ddt, dA, dBpar, dCpar, dDskip};
}
