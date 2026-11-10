import csv
import os
import collections

res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

with open(os.path.join(res_dir, "search_scaling.csv"), encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
grp = collections.defaultdict(list)
for r in rows:
    if r["map"] == "maze":
        grp[(r["size"], r["terrain"], r["algo"])].append(r)
print("=== SEARCH SCALING ===")
for (sz, ter, alg), rs in sorted(grp.items()):
    n = len(rs)
    opt_r = sum(float(x["opt_ratio"]) for x in rs) / n
    exp = sum(int(x["expanded"]) for x in rs) / n
    cov = sum(float(x["coverage"]) for x in rs) / n
    fr = sum(int(x["peak_frontier"]) for x in rs) / n
    ms = sum(float(x["time_ms"]) for x in rs) / n
    print(f"{sz:8s} {ter:6s} {alg:14s} | opt_ratio={opt_r:.2f} | exp={exp:7.1f} | cov={cov:.3f} | front={fr:6.1f} | ms={ms:6.2f}")

with open(os.path.join(res_dir, "multiagent_pruning.csv"), encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
grp = collections.defaultdict(list)
for r in rows:
    grp[(r["map"], r["depth"])].append(r)
print("\n=== MULTIAGENT PRUNING ===")
for (m, d), rs in sorted(grp.items()):
    n = len(rs)
    mm = sum(int(x["minimax_nodes"]) for x in rs) / n
    ab = sum(int(x["alphabeta_nodes"]) for x in rs) / n
    pr = sum(float(x["pruning_ratio"]) * 100 for x in rs) / n
    sp = sum(float(x["speedup"]) for x in rs) / n
    mt = sum(int(x["action_match"]) * 100 for x in rs) / n
    print(f"{m:34s} d={d} | mm_nodes={mm:7.1f} | ab_nodes={ab:7.1f} | prune={pr:5.1f}% | speedup={sp:4.2f}x | match={mt:5.1f}%")

with open(os.path.join(res_dir, "multiagent_games.csv"), encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
grp = collections.defaultdict(list)
for r in rows:
    grp[(r["map"], r["ghost"], r["agent"])].append(r)
print("\n=== MULTIAGENT GAMES ===")
for (m, g, a), rs in sorted(grp.items()):
    n = len(rs)
    win = sum(int(x["win"]) * 100 for x in rs) / n
    sc = sum(float(x["score"]) for x in rs) / n
    fd = sum(float(x["food_cleared"]) * 100 for x in rs) / n
    nd = sum(float(x["nodes_per_move"]) for x in rs) / n
    ms = sum(float(x["ms_per_move"]) for x in rs) / n
    print(f"{m[:18]:18s} | {g:11s} | {a:26s} | win={win:5.1f}% | score={sc:7.1f} | food={fd:5.1f}% | nodes/mv={nd:6.1f} | ms/mv={ms:6.2f}")

with open(os.path.join(res_dir, "rl_pacman.csv"), encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
grp = collections.defaultdict(list)
for r in rows:
    grp[(r["layout"], r["agent"], int(r["n_train"]))].append(r)
print("\n=== RL PACMAN ===")
for (lay, ag, ntr), rs in sorted(grp.items()):
    n = len(rs)
    win = sum(float(x["win_rate"]) * 100 for x in rs) / n
    sc = sum(float(x["avg_score"]) for x in rs) / n
    fd = sum(float(x["food_cleared"]) * 100 for x in rs) / n
    pr = sum(int(x["learned_params"]) for x in rs) / n
    print(f"{lay:14s} | {ag:40s} | N={ntr:4d} | win={win:5.1f}% | score={sc:7.1f} | food={fd:5.1f}% | params={pr:7.1f}")
