"""
exp_ml.py - Danh gia cac ky thuat Hoc may (Project 5) khi mo rong kich thuoc ma tran.

Thi nghiem G (Tich chap 2D - Convolve):
    Mo rong kich thuoc ma tran dau vao tu 28x28 (MNIST) len 64x64, 128x128, 256x256
    voi kich thuoc nhan (kernel) 3x3, 5x5, 7x7.
    So sanh:
      - Naive Loop Convolve (duyet tung o (y, x) bang tensordot)
      - Vectorized Kernel-Slice Convolve (da cai dat trong models.Convolve)
    Do do: thoi gian thuc thi (ms), toc do tang toc (speedup), sai so cuc dai (max_abs_err).

Thi nghiem H (Causal Self-Attention):
    Mo rong chieu dai chuoi (block_size T) tu 16 -> 64 -> 128 -> 256 -> 512 va kich thuoc
    nhung (layer_size C) tu 32 -> 64 -> 128 -> 256 (ma tran chu y kich thuoc T x T).
    Do do: thoi gian forward + backward (ms), thong luong (tokens/s), dung luong ma tran
    attention (KB) minh hoa do phuc tap bac hai O(T^2).
"""
import csv
import os
import sys
import time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(os.path.dirname(HERE), "machinelearning")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, PROJ)

import models  # noqa: E402


def naive_convolve(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out_h = inp.shape[0] - weight.shape[0] + 1
    out_w = inp.shape[1] - weight.shape[1] + 1
    out = torch.zeros((out_h, out_w))
    kh, kw = weight.shape
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = torch.tensordot(inp[i:i + kh, j:j + kw], weight, dims=2)
    return out


def experiment_g():
    rows = []
    sizes = [28, 64, 128, 256]
    kernels = [3, 5, 7]
    torch.manual_seed(42)
    for H in sizes:
        for k in kernels:
            inp = torch.rand((H, H))
            w = torch.rand((k, k))

            # Vectorized (models.Convolve)
            t0 = time.perf_counter()
            for _ in range(5):
                out_vec = models.Convolve(inp, w)
            vec_ms = (time.perf_counter() - t0) * 1000 / 5.0

            # Naive loop (chay 1 lan tren ma tran lon de tranh qua lau)
            t0 = time.perf_counter()
            out_naive = naive_convolve(inp, w)
            naive_ms = (time.perf_counter() - t0) * 1000

            max_err = float(torch.max(torch.abs(out_vec - out_naive)).item())
            speedup = naive_ms / max(vec_ms, 1e-6)
            rows.append(dict(
                matrix_size=f"{H}x{H}",
                kernel_size=f"{k}x{k}",
                output_size=f"{H - k + 1}x{H - k + 1}",
                naive_ms=round(naive_ms, 3),
                vectorized_ms=round(vec_ms, 3),
                speedup=round(speedup, 1),
                max_abs_err=round(max_err, 8),
            ))
            print(f"  [G] Matrix {H:3d}x{H:<3d} Kernel {k}x{k} | "
                  f"Naive: {naive_ms:8.2f}ms | Vectorized: {vec_ms:6.3f}ms | "
                  f"Speedup: {speedup:6.1f}x | Err: {max_err:.1e}")
    return rows


def experiment_h():
    rows = []
    seq_lengths = [16, 64, 128, 256, 512]
    emb_dims = [64, 128, 256]
    batch_size = 8
    torch.manual_seed(42)
    for T in seq_lengths:
        for C in emb_dims:
            attn = models.Attention(layer_size=C, block_size=T)
            x = torch.rand((batch_size, T, C), requires_grad=True)
            # Warmup
            y = attn(x)
            y.sum().backward()

            iters = 5
            t0 = time.perf_counter()
            for _ in range(iters):
                x.grad = None
                y = attn(x)
                y.sum().backward()
            total_ms = (time.perf_counter() - t0) * 1000 / iters
            attn_matrix_kb = (batch_size * T * T * 4) / 1024.0
            tokens_per_sec = (batch_size * T) / (total_ms / 1000.0)
            rows.append(dict(
                batch_size=batch_size,
                seq_len_T=T,
                emb_dim_C=C,
                attn_matrix_shape=f"{batch_size}x{T}x{T}",
                attn_matrix_kb=round(attn_matrix_kb, 1),
                fw_bw_ms=round(total_ms, 2),
                tokens_per_sec=round(tokens_per_sec, 0),
            ))
            print(f"  [H] B={batch_size} T={T:3d} C={C:3d} | "
                  f"AttnMatrix: {attn_matrix_kb:7.1f} KB | "
                  f"FW+BW: {total_ms:6.2f} ms | {tokens_per_sec:8.0f} tok/s")
    return rows


def write_csv(rows, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  -> %s (%d dong)" % (path, len(rows)))


if __name__ == "__main__":
    print("== Thi nghiem G: Mo rong ma tran Tich chap 2D (Convolve) ==")
    write_csv(experiment_g(), "ml_convolve_scaling.csv")
    print("== Thi nghiem H: Mo rong ma tran Causal Self-Attention ==")
    write_csv(experiment_h(), "ml_attention_scaling.csv")
