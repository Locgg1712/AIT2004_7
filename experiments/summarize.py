"""
summarize.py - Tong hop toan bo ket qua thuc nghiem A -> I truc tiep tu cac file CSV,
kem do lech chuan (std), khoang tin cay Wilson 95% (95% CI), va he so mu quy luat luy thua
log-log (scaling exponent alpha: Y ~ X^alpha, R^2).
"""
import collections
import csv
import math
import os
import statistics

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def read_csv(name):
    with open(os.path.join(RES_DIR, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_std(vals):
    if not vals:
        return 0.0, 0.0
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def wilson_ci(wins, n, z=1.96):
    """Khoang tin cay Wilson 95% cho ty le nhi phan wins/n (tra ve %)."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return max(0.0, (center - half) * 100.0), min(100.0, (center + half) * 100.0)


def fit_log_log(xs, ys):
    """Hoi quy tuyen tinh log(y) = alpha * log(x) + b. Tra ve (alpha, R^2)."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2:
        return 0.0, 0.0
    lx = [p[0] for p in pts]
    ly = [p[1] for p in pts]
    mx, my = statistics.mean(lx), statistics.mean(ly)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x in lx)
    alpha = num / den if den > 0 else 0.0
    b = my - alpha * mx
    ss_tot = sum((y - my) ** 2 for y in ly)
    ss_res = sum((y - (alpha * x + b)) ** 2 for x, y in pts)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return alpha, r2


def main():
    # -----------------------------------------------------------------------
    # 1. SEARCH SCALING (Exp A)
    # -----------------------------------------------------------------------
    rows_a = read_csv("search_scaling.csv")
    print("=" * 85)
    print("1. EXPERIMENT A: SEARCH SCALING & MEGA-CITY ROUTE (search_scaling.csv)")
    print("=" * 85)
    grp_a = collections.defaultdict(list)
    for r in rows_a:
        grp_a[(r["map"], r["size"], r["terrain"], r["algo"])].append(r)

    size_order = {"37x37": 0, "31x21": 1, "31x31": 2, "61x61": 3, "101x101": 4, "151x151": 5}
    for (mp, sz, ter, alg), rs in sorted(grp_a.items(), key=lambda x: (x[0][0], size_order.get(x[0][1], 9), x[0][2], x[0][3])):
        cost_m, cost_s = mean_std([float(x["cost"]) for x in rs])
        opt_m, _ = mean_std([float(x["opt_ratio"]) for x in rs])
        exp_m, exp_s = mean_std([float(x["expanded"]) for x in rs])
        cov_m, _ = mean_std([float(x["coverage"]) * 100 for x in rs])
        fr_m, _ = mean_std([float(x["peak_frontier"]) for x in rs])
        ms_m, _ = mean_std([float(x["time_ms"]) for x in rs])
        print(f"{mp:14s} {sz:7s} {ter:5s} {alg:21s} | cost={cost_m:6.1f}+-{cost_s:5.1f} "
              f"| opt={opt_m:4.2f}x | exp={exp_m:7.1f}+-{exp_s:5.1f} ({cov_m:4.1f}%) "
              f"| front={fr_m:5.1f} | ms={ms_m:6.2f}")

    print("\n--- Log-Log Scaling Exponents (Expanded ~ N^alpha, PeakFrontier ~ N^beta on Unit Mazes) ---")
    for alg in ["DFS", "BFS", "UCS", "A*-Manhattan", "A*-Euclid", "WA*-Manhattan (w=2.0)"]:
        ns, exps, frs = [], [], []
        for r in rows_a:
            if r["map"] == "maze" and r["terrain"] == "unit" and r["algo"] == alg:
                n_val = int(r["size"].split("x")[0])
                ns.append(n_val)
                exps.append(float(r["expanded"]))
                frs.append(float(r["peak_frontier"]))
        a_exp, r2_exp = fit_log_log(ns, exps)
        a_fr, r2_fr = fit_log_log(ns, frs)
        print(f"  {alg:22s}: Expanded ~ N^{a_exp:.2f} (R^2={r2_exp:.3f}) | PeakFrontier ~ N^{a_fr:.2f} (R^2={r2_fr:.3f})")

    # -----------------------------------------------------------------------
    # 2. MULTI-FOOD SEARCH (Exp B)
    # -----------------------------------------------------------------------
    rows_b = read_csv("search_food.csv")
    print("\n" + "=" * 85)
    print("2. EXPERIMENT B: MULTI-FOOD COMBINATORIAL SCALING (search_food.csv)")
    print("=" * 85)
    grp_b = collections.defaultdict(list)
    for r in rows_b:
        grp_b[(int(r["k"]), r["algo"])].append(r)

    for (k, alg), rs in sorted(grp_b.items()):
        solved_rs = [x for x in rs if int(x["solved"]) == 1]
        sr = f"{len(solved_rs)}/{len(rs)}"
        exp_m, exp_s = mean_std([float(x["expanded"]) for x in rs])
        ms_m, ms_s = mean_std([float(x["time_ms"]) for x in rs])
        pre_m, _ = mean_std([float(x["precompute_ms"]) for x in rs])
        if solved_rs:
            opt_m, _ = mean_std([float(x["opt_ratio"]) for x in solved_rs if x["opt_ratio"]])
            cost_m, _ = mean_std([float(x["cost"]) for x in solved_rs])
            opt_str = f"{opt_m:.2f}x (cost={cost_m:.1f})"
        else:
            opt_str = "TIMEOUT"
        print(f"k={k:2d} | {alg:32s} | solved={sr} | exp={exp_m:9.1f}+-{exp_s:8.1f} "
              f"| total_ms={ms_m:7.1f} (pre={pre_m:4.1f}ms) | {opt_str}")

    # -----------------------------------------------------------------------
    # 3. MULTI-AGENT GAMES & PRUNING (Exp C & D)
    # -----------------------------------------------------------------------
    rows_c = read_csv("multiagent_games.csv")
    print("\n" + "=" * 85)
    print("3. EXPERIMENT C: MULTI-AGENT ADVERSARIAL GAMES (multiagent_games.csv)")
    print("=" * 85)
    grp_c = collections.defaultdict(list)
    for r in rows_c:
        grp_c[(r["map"], r["ghost"], r["agent"])].append(r)

    for (mp, gh, ag), rs in sorted(grp_c.items()):
        n = len(rs)
        wins = sum(int(x["win"]) for x in rs)
        died = sum(int(x["died"]) for x in rs)
        tout = sum(int(x["timeout"]) for x in rs)
        w_lo, w_hi = wilson_ci(wins, n)
        sc_m, sc_s = mean_std([float(x["score"]) for x in rs])
        fd_m, fd_s = mean_std([float(x["food_cleared"]) * 100 for x in rs])
        mv_m, mv_s = mean_std([float(x["moves"]) for x in rs])
        nd_m, _ = mean_std([float(x["nodes_per_move"]) for x in rs])
        ms_m, _ = mean_std([float(x["ms_per_move"]) for x in rs])
        print(f"{mp:26s} | {gh:11s} | {ag:26s} | W/D/T={wins}/{died}/{tout} "
              f"(win={wins*100/n:5.1f}% CI[{w_lo:4.1f},{w_hi:4.1f}]) "
              f"| sc={sc_m:7.1f}+-{sc_s:5.1f} | fd={fd_m:5.1f}%+-{fd_s:4.1f}% "
              f"| mv={mv_m:5.1f} | nd={nd_m:5.1f} | ms={ms_m:5.2f}")

    rows_d = read_csv("multiagent_pruning.csv")
    print("\n" + "=" * 85)
    print("4. EXPERIMENT D: ALPHA-BETA PRUNING & MOVE ORDERING (multiagent_pruning.csv)")
    print("=" * 85)
    grp_d = collections.defaultdict(list)
    for r in rows_d:
        grp_d[(r["map"], int(r["depth"]))].append(r)

    for (mp, d), rs in sorted(grp_d.items()):
        mm_n, _ = mean_std([float(x["minimax_nodes"]) for x in rs])
        ab_n, _ = mean_std([float(x["alphabeta_nodes"]) for x in rs])
        oab_n, _ = mean_std([float(x["ordered_ab_nodes"]) for x in rs])
        pr_m, pr_s = mean_std([float(x["pruning_ratio"]) * 100 for x in rs])
        opr_m, opr_s = mean_std([float(x["ordered_pruning_ratio"]) * 100 for x in rs])
        sp_m, sp_s = mean_std([float(x["speedup"]) for x in rs])
        osp_m, osp_s = mean_std([float(x["ordered_speedup"]) for x in rs])
        mt_m, _ = mean_std([float(x["action_match"]) * 100 for x in rs])
        print(f"{mp:26s} d={d} | MM={mm_n:6.1f} | AB={ab_n:6.1f} (prune={pr_m:4.1f}%+-{pr_s:4.1f}%, "
              f"speedup={sp_m:4.2f}x+-{sp_s:4.2f}) | OrdAB={oab_n:6.1f} (prune={opr_m:4.1f}%+-{opr_s:4.1f}%, "
              f"speedup={osp_m:4.2f}x) | match={mt_m:5.1f}%")

    # -----------------------------------------------------------------------
    # 5. REINFORCEMENT LEARNING MDP & PACMAN (Exp E & F)
    # -----------------------------------------------------------------------
    rows_e = read_csv("rl_mdp_summary.csv")
    print("\n" + "=" * 85)
    print("5. EXPERIMENT E: MDP VALUE ITERATION VS PRIORITIZED SWEEPING (rl_mdp_summary.csv)")
    print("=" * 85)
    grp_e = collections.defaultdict(list)
    for r in rows_e:
        grp_e[(r["size"], r["algo"])].append(r)

    sz_ord = {"5x5": 0, "10x10": 1, "20x20": 2, "30x30": 3, "40x40": 4}
    for (sz, alg), rs in sorted(grp_e.items(), key=lambda x: (sz_ord.get(x[0][0], 9), x[0][1])):
        st_m, _ = mean_std([float(x["n_states"]) for x in rs])
        bk_m, bk_s = mean_std([float(x["backups_to_tol"]) for x in rs])
        qe_m, qe_s = mean_std([float(x["q_evals_to_tol"]) for x in rs])
        sw_m, _ = mean_std([float(x["sweeps_equiv"]) for x in rs])
        ms_m, ms_s = mean_std([float(x["time_ms"]) for x in rs])
        print(f"{sz:6s} ({st_m:6.1f} st) | {alg:20s} | backups={bk_m:7.1f}+-{bk_s:6.1f} ({sw_m:4.1f} sweeps) "
              f"| q_evals={qe_m:8.1f}+-{qe_s:7.1f} | pure_ms={ms_m:6.2f}+-{ms_s:5.2f}")

    rows_f = read_csv("rl_pacman.csv")
    print("\n" + "=" * 85)
    print("6. EXPERIMENT F: PACMAN Q-LEARNING & ZERO-SHOT TRANSFER (rl_pacman.csv)")
    print("=" * 85)
    grp_f = collections.defaultdict(list)
    for r in rows_f:
        grp_f[(r["layout"], r["agent"], int(r["n_train"]))].append(r)

    for (lay, ag, ntr), rs in sorted(grp_f.items()):
        n_total_games = len(rs) * 30
        wins_total = int(round(sum(float(x["win_rate"]) * 30 for x in rs)))
        w_lo, w_hi = wilson_ci(wins_total, n_total_games)
        win_m, win_s = mean_std([float(x["win_rate"]) * 100 for x in rs])
        sc_m, sc_s = mean_std([float(x["avg_score"]) for x in rs])
        fd_m, fd_s = mean_std([float(x["food_cleared"]) * 100 for x in rs])
        pr_m, _ = mean_std([float(x["learned_params"]) for x in rs])
        print(f"{lay:17s} | {ag:28s} | N={ntr:4d} | win={win_m:5.1f}%+-{win_s:4.1f}% CI[{w_lo:4.1f},{w_hi:4.1f}] "
              f"| sc={sc_m:7.1f}+-{sc_s:5.1f} | fd={fd_m:5.1f}% | params={pr_m:8.1f}")

    # -----------------------------------------------------------------------
    # 7. MACHINE LEARNING: CONVOLVE, ATTENTION, CHAR-LM & IMITATION (Exp G, H, I)
    # -----------------------------------------------------------------------
    rows_g = read_csv("ml_convolve_scaling.csv")
    print("\n" + "=" * 85)
    print("7. EXPERIMENT G: 2D CONVOLUTION SCALING (ml_convolve_scaling.csv)")
    print("=" * 85)
    for r in rows_g:
        print(f"{r['matrix_size']:7s} kernel={r['kernel_size']} | Naive={float(r['naive_ms']):8.2f}ms "
              f"| Vec={float(r['vectorized_ms']):6.3f}+-{float(r['vectorized_std_ms']):.3f}ms "
              f"| F.conv2d={float(r['f_conv2d_ms']):6.3f}ms | Speedup={float(r['speedup']):6.1f}x "
              f"| Vec/F.conv2d={float(r['ratio_vs_fconv2d']):4.1f}x | Err={float(r['max_abs_err']):.1e}")

    rows_h = read_csv("ml_attention_scaling.csv")
    print("\n" + "=" * 85)
    print("8. EXPERIMENT H: CAUSAL SELF-ATTENTION SCALING (ml_attention_scaling.csv)")
    print("=" * 85)
    for r in rows_h:
        print(f"B={r['batch_size']} T={int(r['seq_len_T']):3d} C={int(r['emb_dim_C']):3d} "
              f"| MatrixKB={float(r['attn_matrix_kb']):6.1f} | TotalActKB={float(r['measured_total_activations_kb']):7.1f} "
              f"| FW+BW={float(r['fw_bw_ms']):6.3f}+-{float(r['fw_bw_std_ms']):.3f}ms "
              f"| Tok/s={float(r['tokens_per_sec']):8.0f}")

    ts_64 = [float(r["seq_len_T"]) for r in rows_h if int(r["emb_dim_C"]) == 64 and int(r["seq_len_T"]) >= 64]
    ms_64 = [float(r["fw_bw_ms"]) for r in rows_h if int(r["emb_dim_C"]) == 64 and int(r["seq_len_T"]) >= 64]
    act_64 = [float(r["measured_total_activations_kb"]) for r in rows_h if int(r["emb_dim_C"]) == 64]
    ts_all = [float(r["seq_len_T"]) for r in rows_h if int(r["emb_dim_C"]) == 64]
    a_time, r2_t = fit_log_log(ts_64, ms_64)
    a_mem, r2_m = fit_log_log(ts_all, act_64)
    print(f"  Scaling Exponents (C=64): FW+BW Time (T>=64) ~ T^{a_time:.2f} (R^2={r2_t:.3f}) | "
          f"Total Activations ~ T^{a_mem:.2f} (R^2={r2_m:.3f})")

    rows_lm = read_csv("ml_char_lm.csv")
    print("\n" + "=" * 85)
    print("9. EXPERIMENT H2: CHARACTER LANGUAGE MODEL WITH ATTENTION (ml_char_lm.csv)")
    print("=" * 85)
    for r in rows_lm:
        print(f"{r['model']:26s} ep={int(r['epoch']):2d} | train_loss={float(r['train_loss']):.3f} "
              f"| val_loss={float(r['val_loss']):.3f} | ppl={float(r['val_perplexity']):5.2f} "
              f"| val_acc={float(r['val_acc'])*100:5.1f}%")

    rows_im = read_csv("ml_imitation.csv")
    print("\n" + "=" * 85)
    print("10. EXPERIMENT I: PROJECT 2 <-> PROJECT 5 IMITATION LEARNING (ml_imitation.csv)")
    print("=" * 85)
    for r in rows_im:
        print(f"{r['agent']:36s} | val_acc={float(r['val_action_acc'])*100:5.1f}% "
              f"| score={float(r['megacity_score']):6.1f}+-{float(r['megacity_score_std']):5.1f} "
              f"| food={float(r['megacity_food_cleared'])*100:5.1f}% | ms/mv={float(r['ms_per_move']):.3f} "
              f"| speedup={float(r['speedup_vs_teacher']):5.1f}x")


if __name__ == "__main__":
    main()
