"""NVFP4 fake-quant for QAD with the numerics of the deployed model (llm-compressor export,
run by vLLM's FlashInfer CUTLASS kernels):

weights: block scales from the current weight, as the export computes them from the healed
    weight. One global scale per fused GEMM: gate and up of an expert share one (vLLM's w13),
    q, k and v share one (qkv_proj); down and o have their own. fq_weight(w, gamax).
activations: static global scale (calibrated amax of the GEMM input; for experts one value for
    all experts of a layer, the max, as vLLM reduces it), dynamic 16-element block scales.
    fq_act(x, amax).

Both compiled, values only; callers wrap them in a straight-through estimator. Checked against
the dequantized export in check_fq_vs_export.py.
"""

import torch

FP8_MAX, FP4_MAX = 448.0, 6.0


def _round_fp4(a):
    q = torch.where(a < 2, torch.round(a * 2) / 2, torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2))
    return q.clamp(max=FP4_MAX)


def _blocks(xf, gs):
    """xf fp32 [..., n]; gs fp32 broadcastable to [..., 1, 1] in the block view."""
    b = xf.reshape(*xf.shape[:-1], xf.shape[-1] // 16, 16)
    s = (b.abs().amax(-1, keepdim=True) / FP4_MAX / gs).clamp(max=FP8_MAX).to(torch.float8_e4m3fn).float() * gs
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (torch.sign(b) * _round_fp4(b.abs() / s) * s).reshape(xf.shape)


def _fq_weight(w, gamax):
    """w [..., out, in]; gamax fp32 [..., 1, 1] (global amax of the fused GEMM)."""
    gs = (gamax.float() / (FP4_MAX * FP8_MAX)).unsqueeze(-1)
    return _blocks(w.float(), gs).to(w.dtype)


def _fq_act(x, amax):
    """x [..., n]; amax fp32 scalar tensor (static, calibrated)."""
    gs = amax.float() / (FP4_MAX * FP8_MAX)
    return _blocks(x.float(), gs).to(x.dtype)


def matrix_amax(w):
    """Global amax per matrix over the last two dims, keepdim, fp32."""
    return w.detach().abs().amax(dim=(-2, -1), keepdim=True).float()


fq_weight = torch.compile(_fq_weight, dynamic=False)
fq_act = torch.compile(_fq_act, dynamic=True)
fq_weight_eager, fq_act_eager = _fq_weight, _fq_act


if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    w = (torch.randn(96, 1536, 2048, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(131072, 2048, device="cuda", dtype=torch.bfloat16)
    ga, aa = matrix_amax(w), torch.tensor(5.0, device="cuda")
    for name, f, a in (("weight eager", fq_weight_eager, (w, ga)), ("weight compiled", fq_weight, (w, ga)),
                       ("act eager", fq_act_eager, (x, aa)), ("act compiled", fq_act, (x, aa))):
        with torch.no_grad():
            f(*a)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(5):
                f(*a)
            torch.cuda.synchronize()
        print(name, round((time.time() - t) / 5 * 1000, 1), "ms")
    with torch.no_grad():
        for nm, a_, b_ in (("weight", fq_weight(w, ga), fq_weight_eager(w, ga)), ("act", fq_act(x, aa), fq_act_eager(x, aa))):
            d = (a_.float() - b_.float()).abs()
            print(nm, "compiled vs eager differing fraction", float((d > 0).float().mean()))
