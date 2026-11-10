"""
exp_ml.py - Danh gia cac ky thuat Hoc may (Project 5) & Ket noi Project 2 <-> Project 5 tren Mega-City.

Thi nghiem G (Tich chap 2D - Convolve voi 3 moc doi chieu, warm-up + median 15 lan, don luong):
    Mo rong kich thuoc ma tran tu 28x28 len 64x64, 128x128, 256x256 voi kernel 3x3, 5x5, 7x7:
      1. Naive Loop Convolve (2 vong lap pixel + tensordot)
      2. Vectorized Kernel-Slice Convolve (models.Convolve - Project 5 Q5)
      3. PyTorch C++ F.conv2d (moc chuan toi uu cap thap)

Thi nghiem H (Causal Self-Attention Scaling & Mo hinh Ngon ngu Ky tu thuc te):
    1. Quet chieu dai chuoi T = 16 -> 64 -> 128 -> 256 -> 512 (B=8, C=64, 128, 256):
       Do thoi gian median 15 lan (sau warm-up) va DO BO NHO THUC TE (measured_peak_kb)
       cua cac tensor luu trong do thi autograd so voi ly thuyet B*T^2*4 bytes.
    2. Huan luyen Mo hinh Ngon ngu Ky tu (Char-LM) su dung khoi `models.Attention` so voi MLP-Bigram:
       Do train_loss, val_loss, val_perplexity (exp(loss)) va val_acc qua 15 epochs.

Thi nghiem I (Ket noi Project 2 <-> Project 5: Imitation Learning tren Mega-City):
    Dung `AlphaBeta-d2 (mazeEval)` lam Giao vien (Expert Teacher) sinh du lieu (state -> action)
    tren Mega-City (31x21, 3 Ghosts) va mediumClassic. Huan luyen 2 mang hoc sau tu Project 5:
      1. Imitation-MLP : Mang da tang Linear + ReLU
      2. Imitation-CNN : Mang tich chap su dung truc tiep ham `models.Convolve` tren cua so khong gian 7x7 quanh Pacman
    Do: do chinh xac khop hanh dong (val_action_acc), diem so khi choi that, ty le an thuc an,
    va toc do suy luan (ms/move) nhanh hon AlphaBeta bao nhieu lan.
"""
import csv
import math
import os
import random
import statistics
import sys
import time
import tracemalloc
import torch
import torch.nn as nn
import torch.nn.functional as F

# Co dinh 1 luong CPU de tranh nhieu tranh chap OpenMP khi do thoi gian micro-benchmark
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ML = os.path.join(os.path.dirname(HERE), "machinelearning")
PROJ_MA = os.path.join(os.path.dirname(HERE), "multiagent")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, PROJ_ML)
sys.path.append(HERE)

import models  # noqa: E402


# ---------------------------------------------------------------------------
# Thi nghiem G: Tich chap 2D (Naive vs models.Convolve vs F.conv2d)
# ---------------------------------------------------------------------------
def naive_convolve(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out_h = inp.shape[0] - weight.shape[0] + 1
    out_w = inp.shape[1] - weight.shape[1] + 1
    out = torch.zeros((out_h, out_w))
    kh, kw = weight.shape
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = torch.tensordot(inp[i:i + kh, j:j + kw], weight, dims=2)
    return out


def f_conv2d_wrapper(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.conv2d(inp.unsqueeze(0).unsqueeze(0), weight.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)


def experiment_g():
    rows = []
    sizes = [28, 64, 128, 256]
    kernels = [3, 5, 7]
    torch.manual_seed(42)

    for H in sizes:
        for k in kernels:
            inp = torch.rand((H, H))
            w = torch.rand((k, k))

            # Warm-up 3 lan cho Vectorized va F.conv2d
            for _ in range(3):
                out_vec = models.Convolve(inp, w)
                out_f = f_conv2d_wrapper(inp, w)

            # Do Vectorized (median 15 lan)
            vec_times = []
            for _ in range(15):
                t0 = time.perf_counter()
                out_vec = models.Convolve(inp, w)
                vec_times.append((time.perf_counter() - t0) * 1000)
            vec_ms = statistics.median(vec_times)
            vec_std = statistics.stdev(vec_times)

            # Do F.conv2d (median 15 lan)
            f_times = []
            for _ in range(15):
                t0 = time.perf_counter()
                out_f = f_conv2d_wrapper(inp, w)
                f_times.append((time.perf_counter() - t0) * 1000)
            f_ms = statistics.median(f_times)

            # Do Naive loop (3 lan cho H <= 128, 1 lan cho H = 256)
            n_reps = 3 if H <= 128 else 1
            naive_times = []
            for _ in range(n_reps):
                t0 = time.perf_counter()
                out_naive = naive_convolve(inp, w)
                naive_times.append((time.perf_counter() - t0) * 1000)
            naive_ms = statistics.median(naive_times)

            max_err = float(torch.max(torch.abs(out_vec - out_naive)).item())
            max_err_f = float(torch.max(torch.abs(out_vec - out_f)).item())
            speedup = naive_ms / max(vec_ms, 1e-6)
            ratio_f = vec_ms / max(f_ms, 1e-6)

            rows.append(dict(
                matrix_size=f"{H}x{H}",
                size=H,
                kernel_size=f"{k}x{k}",
                k=k,
                output_size=f"{H - k + 1}x{H - k + 1}",
                naive_ms=round(naive_ms, 3),
                vectorized_ms=round(vec_ms, 4),
                vectorized_std_ms=round(vec_std, 4),
                f_conv2d_ms=round(f_ms, 4),
                speedup=round(speedup, 1),
                ratio_vs_fconv2d=round(ratio_f, 2),
                max_abs_err=round(max_err, 8),
                max_err_vs_fconv2d=round(max_err_f, 8),
            ))
            print(f"  [G] {H:3d}x{H:<3d} k={k} | Naive: {naive_ms:8.2f}ms | "
                  f"Vec: {vec_ms:6.3f}ms (+-{vec_std:.3f}) | F.conv2d: {f_ms:6.3f}ms | "
                  f"Speedup: {speedup:7.1f}x | Vec/F.conv2d: {ratio_f:4.1f}x | Err: {max_err:.1e}",
                  flush=True)
    return rows


# ---------------------------------------------------------------------------
# Thi nghiem H: Causal Self-Attention Scaling (Do bo nho thuc te & Thoi gian median)
# ---------------------------------------------------------------------------
def measure_attention_memory_kb(attn_module, x):
    """Do dung luong bo nho tensor thuc te duoc cap phat trong forward pass (KB)
    bang tracemalloc ket hop kiem dem kich thuoc cac tensor trung gian Q, K, V, M, Softmax(M), Output."""
    B, T, C = x.shape
    tracemalloc.start()
    tracemalloc.reset_peak()
    q = attn_module.q_layer(x)
    k = attn_module.k_layer(x)
    v = attn_module.v_layer(x)
    M_raw = torch.matmul(q, torch.movedim(k, 1, 2)) / (C ** 0.5)
    # Luu y: models.Attention dung M.masked_fill(...)[0] theo goi y cua CS188 (kich thuoc B x T x T)
    M_masked = M_raw.masked_fill(attn_module.mask[:, :, :T, :T] == 0, float("-inf"))[0]
    W_attn = torch.softmax(M_masked, dim=-1)
    out = torch.matmul(W_attn, v)
    _, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Tong dung luong cac tensor trung gian phai luu cho backward (Q, K, V, M_raw, M_masked, W_attn, out)
    tensors_bytes = sum(t.numel() * t.element_size() for t in (q, k, v, M_raw, M_masked, W_attn, out))
    attn_score_plus_prob_bytes = (M_raw.numel() + W_attn.numel()) * 4
    return round(tensors_bytes / 1024.0, 1), round(attn_score_plus_prob_bytes / 1024.0, 1), round(peak_py / 1024.0, 1)


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

            # Warm-up 3 lan
            for _ in range(3):
                x.grad = None
                y = attn(x)
                y.sum().backward()

            # Do thoi gian median cua 15 lan chay
            times = []
            for _ in range(15):
                t0 = time.perf_counter()
                x.grad = None
                y = attn(x)
                y.sum().backward()
                times.append((time.perf_counter() - t0) * 1000)

            med_ms = statistics.median(times)
            std_ms = statistics.stdev(times)
            theo_attn_kb = round((batch_size * T * T * 4) / 1024.0, 1)
            total_act_kb, attn_tensors_kb, _ = measure_attention_memory_kb(attn, x)
            tokens_per_sec = (batch_size * T) / (med_ms / 1000.0)

            rows.append(dict(
                batch_size=batch_size,
                seq_len_T=T,
                emb_dim_C=C,
                attn_matrix_shape=f"{batch_size}x{T}x{T}",
                attn_matrix_kb=theo_attn_kb,
                measured_attn_tensors_kb=attn_tensors_kb,
                measured_total_activations_kb=total_act_kb,
                fw_bw_ms=round(med_ms, 3),
                fw_bw_std_ms=round(std_ms, 3),
                tokens_per_sec=round(tokens_per_sec, 0),
            ))
            print(f"  [H] B={batch_size} T={T:3d} C={C:3d} | "
                  f"Matrix(1x): {theo_attn_kb:6.1f}KB | TotalAct: {total_act_kb:7.1f}KB | "
                  f"FW+BW: {med_ms:6.3f}+-{std_ms:.3f}ms | {tokens_per_sec:8.0f} tok/s",
                  flush=True)
    return rows


# ---------------------------------------------------------------------------
# Thi nghiem H2: Huan luyen Mo hinh Ngon ngu Ky tu (Char-LM) voi models.Attention
# ---------------------------------------------------------------------------
class CharAttentionLM(nn.Module):
    """Mo hinh ngon ngu ky tu su dung khoi `models.Attention` (Q6) de du doan ky tu tiep theo."""
    def __init__(self, vocab_size, emb_dim=48, block_size=32):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(block_size, emb_dim)
        self.attn1 = models.Attention(layer_size=emb_dim, block_size=block_size)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2), nn.ReLU(), nn.Linear(emb_dim * 2, emb_dim))
        self.ln2 = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, idx):
        # Lưu ý: models.Attention dùng mask[0] nên để mỗi mẫu trong batch có ma trận chú ý riêng,
        # ta chạy từng mẫu hoặc dùng B=1 trên từng chuỗi (hoặc chạy vmap/loop qua batch nhỏ).
        B, T = idx.shape
        pos = torch.arange(T).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        # Chay tung mau de tranh hieu ung broadcast mask[0] cua template CS188 khi huan luyen
        attn_outs = [self.attn1(x[b:b + 1]) for b in range(B)]
        a_out = torch.cat(attn_outs, dim=0)
        x = self.ln1(x + a_out)
        x = self.ln2(x + self.ffn(x))
        return self.head(x)


class CharBigramMLP(nn.Module):
    """Baseline MLP khong co Self-Attention (chi nhin ky tu hien tai t de du doan t+1)."""
    def __init__(self, vocab_size, emb_dim=48):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.net = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2), nn.ReLU(), nn.Linear(emb_dim * 2, vocab_size))

    def forward(self, idx):
        return self.net(self.emb(idx))


def experiment_char_lm():
    """Huan luyen CharAttentionLM vs CharBigramMLP tren van ban ma nguon cua du an."""
    torch.manual_seed(42)
    random.seed(42)
    corpus_files = [
        os.path.join(HERE, "scenario.py"),
        os.path.join(PROJ_ML, "models.py"),
        os.path.join(PROJ_MA, "multiAgents.py"),
    ]
    text = "\n".join(open(p, encoding="utf-8").read() for p in corpus_files if os.path.exists(p))
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    n_split = int(0.85 * len(data))
    train_data, val_data = data[:n_split], data[n_split:]
    block_size = 32
    batch_size = 16

    def get_batch(split_tensor, n_batches=25):
        batches = []
        for _ in range(n_batches):
            ix = torch.randint(0, len(split_tensor) - block_size - 1, (batch_size,))
            x = torch.stack([split_tensor[i:i + block_size] for i in ix])
            y = torch.stack([split_tensor[i + 1:i + block_size + 1] for i in ix])
            batches.append((x, y))
        return batches

    train_batches = get_batch(train_data, 30)
    val_batches = get_batch(val_data, 10)

    models_dict = {
        "Attention-CharLM (Q6)": CharAttentionLM(len(chars), emb_dim=48, block_size=block_size),
        "Bigram-MLP (No Attention)": CharBigramMLP(len(chars), emb_dim=48),
    }

    rows = []
    for name, model in models_dict.items():
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        for epoch in range(1, 13):
            model.train()
            tr_losses = []
            for xb, yb in train_batches:
                opt.zero_grad()
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, len(chars)), yb.view(-1))
                loss.backward()
                opt.step()
                tr_losses.append(loss.item())

            model.eval()
            val_losses, correct, total = [], 0, 0
            with torch.no_grad():
                for xb, yb in val_batches:
                    logits = model(xb)
                    loss = F.cross_entropy(logits.view(-1, len(chars)), yb.view(-1))
                    val_losses.append(loss.item())
                    preds = logits.argmax(dim=-1)
                    correct += int((preds == yb).sum().item())
                    total += yb.numel()

            tr_l = sum(tr_losses) / len(tr_losses)
            va_l = sum(val_losses) / len(val_losses)
            ppl = math.exp(min(va_l, 10.0))
            acc = correct / total
            if epoch in (1, 4, 8, 12):
                rows.append(dict(
                    model=name,
                    epoch=epoch,
                    vocab_size=len(chars),
                    block_size=block_size,
                    train_loss=round(tr_l, 4),
                    val_loss=round(va_l, 4),
                    val_perplexity=round(ppl, 2),
                    val_acc=round(acc, 4),
                ))
                print(f"  [H-LM] {name:26s} ep={epoch:2d} | train_loss={tr_l:.3f} | "
                      f"val_loss={va_l:.3f} | ppl={ppl:5.2f} | val_acc={acc * 100:5.1f}%", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Thi nghiem I: Ket noi Project 2 <-> Project 5 (Imitation Learning tren Mega-City)
# ---------------------------------------------------------------------------
ACTIONS_LIST = ["North", "South", "East", "West", "Stop"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS_LIST)}


def extract_state_tensors(state, dist_map_all):
    """Trich xuat:
      1. Cua so khong gian 7x7 xung quanh Pacman (2 kenh: kenh Hấp dẫn = Food/Capsule/ScaredGhost,
         kenh Nguy hiểm = Wall/ActiveGhost) de dua vao `models.Convolve`.
      2. Vector 14 dac trung dinh huong (4 huong hop le, khoang cach me cung toi thuc an/capsule/ghost
         khi buoc theo 4 huong North, South, East, West)."""
    pos = state.getPacmanPosition()
    px, py = int(round(pos[0])), int(round(pos[1]))
    walls = state.getWalls()
    food = state.getFood()
    capsules = set(state.getCapsules())
    ghostStates = state.getGhostStates()

    W, H = walls.width, walls.height
    attract_patch = torch.zeros((7, 7), dtype=torch.float32)
    danger_patch = torch.zeros((7, 7), dtype=torch.float32)

    active_positions = set()
    scared_positions = set()
    for g in ghostStates:
        gx, gy = int(round(g.getPosition()[0])), int(round(g.getPosition()[1]))
        if g.scaredTimer > 1:
            scared_positions.add((gx, gy))
        else:
            active_positions.add((gx, gy))

    for dr in range(-3, 4):
        for dc in range(-3, 4):
            nx, ny = px + dc, py + dr
            r_idx, c_idx = dr + 3, dc + 3
            if 0 <= nx < W and 0 <= ny < H:
                if walls[nx][ny]:
                    danger_patch[r_idx, c_idx] = -0.5
                if (nx, ny) in active_positions:
                    danger_patch[r_idx, c_idx] = -2.0
                if food[nx][ny]:
                    attract_patch[r_idx, c_idx] = 1.0
                if (nx, ny) in capsules or (nx, ny) in scared_positions:
                    attract_patch[r_idx, c_idx] = 2.0
            else:
                danger_patch[r_idx, c_idx] = -0.5

    # Vector 16 dac trung huong cho 4 huong (North, South, East, West)
    legal = set(state.getLegalActions(0))
    food_list = food.asList()
    vec = []
    dirs = [("North", (0, 1)), ("South", (0, -1)), ("East", (1, 0)), ("West", (-1, 0))]
    for dname, (dx, dy) in dirs:
        is_leg = 1.0 if dname in legal else 0.0
        nx, ny = px + dx, py + dy
        if not is_leg:
            vec.extend([0.0, 0.0, 0.0, -1.0])
            continue
        d_from_next = dist_map_all.get((nx, ny), {})
        # Khoang cach me cung toi thuc an gan nhat neu buoc theo huong dname
        if food_list:
            f_dist = min(d_from_next.get(f, abs(nx - f[0]) + abs(ny - f[1])) for f in food_list)
            f_feat = 1.0 / (f_dist + 1.0)
        else:
            f_feat = 0.0
        # Khoang cach toi Ghost hoat dong gan nhat
        if active_positions:
            g_dist = min(d_from_next.get(gp, abs(nx - gp[0]) + abs(ny - gp[1])) for gp in active_positions)
            g_feat = -1.0 if g_dist <= 1 else -1.0 / (g_dist + 1.0)
        else:
            g_feat = 0.0
        # Khoang cach toi Capsule / Scared Ghost
        targets = list(capsules) + list(scared_positions)
        if targets:
            c_dist = min(d_from_next.get(tp, abs(nx - tp[0]) + abs(ny - tp[1])) for tp in targets)
            c_feat = 1.0 / (c_dist + 1.0)
        else:
            c_feat = 0.0
        vec.extend([is_leg, f_feat, c_feat, g_feat])

    return attract_patch, danger_patch, torch.tensor(vec, dtype=torch.float32)


class ImitationMLP(nn.Module):
    """Mang MLP (Project 5 Q3) hoc bat chuoc chinh sach AlphaBeta."""
    def __init__(self, in_dim=16 + 98, hidden=128, num_actions=5):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, 64)
        self.l3 = nn.Linear(64, num_actions)

    def forward(self, patch_a, patch_d, vec):
        flat = torch.cat([patch_a.flatten(start_dim=1), patch_d.flatten(start_dim=1), vec], dim=1)
        x = F.relu(self.l1(flat))
        x = F.relu(self.l2(x))
        return self.l3(x)


class ImitationCNN(nn.Module):
    """Mang CNN su dung truc tiep ham `models.Convolve` (Project 5 Q5) tren cua so 7x7 quanh Pacman."""
    def __init__(self, vec_dim=16, hidden=128, num_actions=5):
        super().__init__()
        self.w_attract = nn.Parameter(torch.randn(3, 3) * 0.35)
        self.w_danger = nn.Parameter(torch.randn(3, 3) * 0.35)
        # 7x7 convolve 3x3 -> 5x5 = 25 moi kenh -> 50 + vec_dim
        self.ln = nn.LayerNorm(50 + vec_dim)
        self.fc1 = nn.Linear(50 + vec_dim, hidden)
        self.fc2 = nn.Linear(hidden, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward_single(self, patch_a_2d, patch_d_2d, vec_1d):
        ca = F.relu(models.Convolve(patch_a_2d, self.w_attract)).flatten()
        cd = F.relu(models.Convolve(patch_d_2d, self.w_danger)).flatten()
        feat = self.ln(torch.cat([ca, cd, vec_1d], dim=0))
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(feat)))))

    def forward(self, patch_a, patch_d, vec):
        return torch.stack([
            self.forward_single(patch_a[i], patch_d[i], vec[i])
            for i in range(len(vec))
        ])


def experiment_imitation():
    """Thu thap du lieu tu Giao vien AlphaBeta-d2 (mazeEval) tren Mega-City va huan luyen
    ImitationMLP + ImitationCNN (models.Convolve), sau do cho thi dau truc tiep tren Mega-City."""
    sys.path.insert(0, PROJ_MA)
    import layout as lay_mod
    import pacman
    import game
    import textDisplay
    import ghostAgents
    import multiAgents
    import scenario
    import exp_multiagent

    torch.manual_seed(42)
    random.seed(42)

    mc_lines = scenario.make_megacity("arena", seed=1)
    mc_layout = lay_mod.Layout(mc_lines)
    gs0 = pacman.GameState()
    gs0.initialize(mc_layout, mc_layout.getNumGhosts())
    dist_map_all = exp_multiagent.get_wall_dist_map(gs0.getWalls())

    teacher = multiAgents.AlphaBetaAgent("mazeBetterEvaluationFunction", "2")

    # 1) Thu thap du lieu (state -> teacher_action) tren Mega-City
    patches_a, patches_d, vecs, labels = [], [], [], []
    teacher_times = []

    class DataCollector(game.Agent):
        def __init__(self):
            super().__init__(0)
            self.moves = 0

        def getAction(self, state):
            self.moves += 1
            if self.moves > 200:
                raise exp_multiagent.MoveCap()
            t0 = time.perf_counter()
            act = teacher.getAction(state)
            teacher_times.append((time.perf_counter() - t0) * 1000)
            pa, pd, v = extract_state_tensors(state, dist_map_all)
            patches_a.append(pa)
            patches_d.append(pd)
            vecs.append(v)
            labels.append(ACTION_TO_IDX[act])
            return act

    rules = pacman.ClassicGameRules(timeout=10 ** 6)
    disp = textDisplay.NullGraphics()
    for ep_seed in range(1, 7):
        random.seed(ep_seed)
        g_type = ghostAgents.DirectionalGhost if ep_seed % 2 == 0 else ghostAgents.RandomGhost
        ghosts = [g_type(i + 1) for i in range(mc_layout.getNumGhosts())]
        g = rules.newGame(mc_layout, DataCollector(), ghosts, disp, quiet=True)
        try:
            g.run()
        except exp_multiagent.MoveCap:
            pass

    X_pa = torch.stack(patches_a)
    X_pd = torch.stack(patches_d)
    X_vec = torch.stack(vecs)
    Y = torch.tensor(labels, dtype=torch.long)

    # Chia train / val (80% / 20%)
    N = len(Y)
    perm = torch.randperm(N)
    n_tr = int(0.8 * N)
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]

    students = {
        "Imitation-MLP (Project 5 Q3)": ImitationMLP(),
        "Imitation-CNN (models.Convolve Q5)": ImitationCNN(),
    }

    val_accs = {}
    for sname, net in students.items():
        opt = torch.optim.Adam(net.parameters(), lr=0.008)
        bs = 32
        for _ in range(25):
            net.train()
            shuf = tr_idx[torch.randperm(len(tr_idx))]
            for i in range(0, len(shuf), bs):
                b = shuf[i:i + bs]
                opt.zero_grad()
                out = net(X_pa[b], X_pd[b], X_vec[b])
                loss = F.cross_entropy(out, Y[b])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            logits = net(X_pa[va_idx], X_pd[va_idx], X_vec[va_idx])
            preds = logits.argmax(dim=-1)
            val_accs[sname] = float((preds == Y[va_idx]).float().mean().item())

    # 2) Cho Teacher va 2 Student thi dau truc tiep tren Mega-City
    import collections as _col

    class NeuralPacmanAgent(game.Agent):
        def __init__(self, net):
            super().__init__(0)
            self.net = net
            self.times = []
            self.moves = 0
            self.visits = _col.Counter()
            self.last_food = None

        def getAction(self, state):
            self.moves += 1
            if self.moves > 300:
                raise exp_multiagent.MoveCap()
            cur_food = state.getNumFood()
            if self.last_food is not None and cur_food < self.last_food:
                self.visits.clear()
            self.last_food = cur_food
            px, py = int(round(state.getPacmanPosition()[0])), int(round(state.getPacmanPosition()[1]))
            self.visits[(px, py)] += 1

            t0 = time.perf_counter()
            pa, pd, v = extract_state_tensors(state, dist_map_all)
            with torch.no_grad():
                logits = self.net(pa.unsqueeze(0), pd.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
            legal = state.getLegalActions(0)

            def action_priority(a):
                dx, dy = game.Actions.directionToVector(a)
                nx, ny = int(px + dx), int(py + dy)
                pen = 0.45 * self.visits[(nx, ny)] + (0.6 if a == "Stop" else 0.0)
                return float(logits[ACTION_TO_IDX[a]].item()) - pen

            best_a = max(legal, key=action_priority)
            self.times.append((time.perf_counter() - t0) * 1000)
            return best_a

    def eval_on_megacity(agent_builder, is_teacher=False):
        scores, foods, moves_l, times_l, wins = [], [], [], [], 0
        for s in range(1, 9):
            random.seed(100 + s)
            ghosts = [ghostAgents.DirectionalGhost(i + 1) for i in range(mc_layout.getNumGhosts())]
            ag = agent_builder()
            g = rules.newGame(mc_layout, ag, ghosts, disp, quiet=True)
            try:
                g.run()
            except exp_multiagent.MoveCap:
                pass
            scores.append(g.state.getScore())
            wins += int(g.state.isWin())
            foods.append(1 - g.state.getNumFood() / mc_layout.totalFood)
            moves_l.append(len(ag.times))
            times_l.extend(ag.times)
        return (
            round(statistics.mean(scores), 1),
            round(statistics.stdev(scores), 1),
            round(statistics.mean(foods), 4),
            round(statistics.mean(moves_l), 1),
            round(statistics.median(times_l), 3),
        )

    class CappedTeacher(game.Agent):
        def __init__(self):
            super().__init__(0)
            self.inner = multiAgents.AlphaBetaAgent("mazeBetterEvaluationFunction", "2")
            self.times = []

        def getAction(self, state):
            if len(self.times) >= 300:
                raise exp_multiagent.MoveCap()
            t0 = time.perf_counter()
            a = self.inner.getAction(state)
            self.times.append((time.perf_counter() - t0) * 1000)
            return a

    t_sc, t_std, t_fd, t_mv, t_ms = eval_on_megacity(
        lambda: CappedTeacher(),
        is_teacher=True,
    )

    rows = [
        dict(
            agent="Teacher: AlphaBeta-d2 (mazeEval)",
            role="Expert Teacher (Search)",
            dataset_samples=N,
            val_action_acc=1.0000,
            megacity_score=t_sc,
            megacity_score_std=t_std,
            megacity_food_cleared=t_fd,
            megacity_moves=t_mv,
            ms_per_move=t_ms,
            speedup_vs_teacher=1.0,
        )
    ]

    for sname, net in students.items():
        sc, std, fd, mv, ms = eval_on_megacity(lambda net=net: NeuralPacmanAgent(net))
        sp = round(t_ms / max(ms, 1e-4), 1)
        rows.append(dict(
            agent=sname,
            role="Neural Student (Project 5)",
            dataset_samples=N,
            val_action_acc=round(val_accs[sname], 4),
            megacity_score=sc,
            megacity_score_std=std,
            megacity_food_cleared=fd,
            megacity_moves=mv,
            ms_per_move=ms,
            speedup_vs_teacher=sp,
        ))
        print(f"  [I] {sname:36s} | val_acc={val_accs[sname]*100:5.1f}% | "
              f"MegaCity score={sc:6.1f}+-{std:5.1f} | food={fd*100:5.1f}% | "
              f"{ms:.3f}ms/mv ({sp:.1f}x faster than Teacher)", flush=True)
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
    t0 = time.time()
    if "--only-imitation" in sys.argv:
        print("== Thi nghiem I: Ket noi Project 2 <-> Project 5 (Imitation Learning tren Mega-City) ==")
        write_csv(experiment_imitation(), "ml_imitation.csv")
    else:
        print("== Thi nghiem G: Mo rong ma tran Tich chap 2D (Naive vs models.Convolve vs F.conv2d) ==")
        write_csv(experiment_g(), "ml_convolve_scaling.csv")
        print("== Thi nghiem H: Mo rong ma tran Causal Self-Attention (Warm-up + Median + Measured Memory) ==")
        write_csv(experiment_h(), "ml_attention_scaling.csv")
        print("== Thi nghiem H2: Huan luyen Mo hinh Ngon ngu Ky tu voi models.Attention ==")
        write_csv(experiment_char_lm(), "ml_char_lm.csv")
        print("== Thi nghiem I: Ket noi Project 2 <-> Project 5 (Imitation Learning tren Mega-City) ==")
        write_csv(experiment_imitation(), "ml_imitation.csv")
    print("Xong sau %.1f giay" % (time.time() - t0))
