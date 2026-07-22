from __future__ import annotations

import pytest
import torch

import foreblocks.models.transformer.features.patching as patching

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not patching._TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("shape", "patch_len", "stride"),
    [
        ((2, 17, 10), 5, 2),
        ((1, 16, 65), 4, 4),
        ((2, 9, 7), 3, 5),
    ],
)
def test_triton_patch_tokenizer_matches_pytorch_forward_and_backward(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    patch_len: int,
    stride: int,
) -> None:
    torch.manual_seed(11)
    triton_tokenizer = (
        patching.PatchTokenizer(shape[-1], patch_len, stride, pad_end=True)
        .cuda()
        .to(dtype)
    )
    eager_tokenizer = (
        patching.PatchTokenizer(shape[-1], patch_len, stride, pad_end=True)
        .cuda()
        .to(dtype)
    )
    eager_tokenizer.load_state_dict(triton_tokenizer.state_dict())

    triton_input = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    eager_input = triton_input.detach().clone().requires_grad_(True)

    triton_output, triton_info = triton_tokenizer(triton_input)
    monkeypatch.setattr(patching, "_can_use_triton_patchify", lambda _x: False)
    eager_output, eager_info = eager_tokenizer(eager_input)

    tolerance = 2e-2 if dtype != torch.float32 else 1e-5
    torch.testing.assert_close(
        triton_output, eager_output, rtol=tolerance, atol=tolerance
    )
    assert triton_info == eager_info

    gradient = torch.randn_like(triton_output)
    triton_output.backward(gradient)
    eager_output.backward(gradient)
    torch.testing.assert_close(
        triton_input.grad, eager_input.grad, rtol=tolerance, atol=tolerance
    )
    for triton_parameter, eager_parameter in zip(
        triton_tokenizer.parameters(), eager_tokenizer.parameters(), strict=True
    ):
        torch.testing.assert_close(
            triton_parameter.grad,
            eager_parameter.grad,
            rtol=tolerance,
            atol=tolerance,
        )
