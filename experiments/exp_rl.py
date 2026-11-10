"""
exp_rl.py - Danh gia cac thuat toan quy hoach & hoc tang cuong (Project 3) tren kich ban lon.

Thi nghiem E (quy hoach co mo hinh - MDP):
    So sanh ValueIterationAgent va PrioritizedSweepingValueIterationAgent tren
        - DiscountGrid 5x5 (ban do goc cua do an)
        - Gridworld lon 10x10, 20x20, 30x30 (co tuong, ho bay -10, loi ra +1 va +10)
    Do do: so phep cap nhat trang thai (Bellman backup) va thoi gian de dat sai so
    gia tri max|V - V*| < 1e-3, cung do trung khop chinh sach (policy agreement) va
    duong cong sai so theo so phep cap nhat.

Thi nghiem F (hoc khong mo hinh tren Pacman):
    So sanh QLearningAgent (bang tra - tabular) va ApproximateQAgent (xap xi tuyen tinh
    voi SimpleExtractor) tren
        - smallGrid (7x7, 1 ma, 2 thuc an)     : ban do goc cua Q8
        - mediumClassic (20x11, 2 ma, 97 thuc an): khong gian trang thai ~10^30
    Do do: ty le thang va diem trung binh cua 50 van kiem tra sau N = 50, 200, 500, 1000, 2000
    van huan luyen; kich thuoc mo hinh (so muc Q da hoc vs 4 trong so dac trung).

Chay:  python exp_rl.py
"""
import csv
import os
import random
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(os.path.dirname(HERE), "reinforcement")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, PROJ)
sys.path.append(HERE)
os.chdir(PROJ)

import gridworld              # noqa: E402
import layout as lay_mod      # noqa: E402
import pacman                 # noqa: E402
import textDisplay            # noqa: E402
import ghostAgents            # noqa: E402
import valueIterationAgents   # noqa: E402
import qlearningAgents        # noqa: E402
import util                   # noqa: E402
import scenario               # noqa: E402


# ---------------------------------------------------------------------------
# Thi nghiem E: Value Iteration vs Prioritized Sweeping
# ---------------------------------------------------------------------------
def build_mdp(size, seed, discount=0.9, noise=0.2, living=-0.04):
    if size == 5:
        gw = gridworld.getDiscountGrid()
    else:
        gw = gridworld.Gridworld(scenario.make_gridworld(size, size, seed))
    gw.setNoise(noise)
    gw.setLivingReward(living)
    return gw, discount


def reference_vi(mdp, discount, tol=1e-6, max_iters=300):
    """Chay Value Iteration den hoi tu chat de lay V* va chinh sach toi uu pi*."""
    agent = valueIterationAgents.ValueIterationAgent(mdp, discount, iterations=0)
    non_term = [s for s in mdp.getStates() if not mdp.isTerminal(s)]
    for _ in range(max_iters):
        new_v = util.Counter()
        diff = 0.0
        for s in non_term:
            a = agent.computeActionFromValues(s)
            if a is not None:
                new_v[s] = agent.computeQValueFromValues(s, a)
                diff = max(diff, abs(new_v[s] - agent.values[s]))
        agent.values = new_v
        if diff < tol:
            break
    opt_actions = {}
    for s in non_term:
        acts = mdp.getPossibleActions(s)
        qs = {a: agent.computeQValueFromValues(s, a) for a in acts}
        m = max(qs.values())
        opt_actions[s] = {a for a, q in qs.items() if abs(q - m) <= 1e-5}
    return agent.values, opt_actions, non_term


def metrics_vs_ref(agent, v_star, opt_actions, non_term):
    max_err = max(abs(agent.values[s] - v_star[s]) for s in non_term)
    agree = sum(1 for s in non_term if agent.computeActionFromValues(s) in opt_actions[s])
    return max_err, agree / len(non_term)


def run_mdp_convergence(size, seed, tol=1e-3):
    mdp, discount = build_mdp(size, seed)
    v_star, opt_actions, non_term = reference_vi(mdp, discount)
    n_states = len(non_term)
    checkpoints = {50, 100, 250, 500, 1000, 2000, 4000, 8000, 15000}
    curve_rows = []

    # 1) Value Iteration (moi vong quet cap nhat tat ca n_states trang thai)
    vi = valueIterationAgents.ValueIterationAgent(mdp, discount, iterations=0)
    backups = 0
    vi_hit_backups = vi_hit_ms = None
    t0 = time.perf_counter()
    for it in range(1, 301):
        new_v = util.Counter()
        for s in non_term:
            a = vi.computeActionFromValues(s)
            if a is not None:
                new_v[s] = vi.computeQValueFromValues(s, a)
        vi.values = new_v
        backups += n_states
        err, ag = metrics_vs_ref(vi, v_star, opt_actions, non_term)
        if any(backups - n_states < c <= backups for c in checkpoints):
            curve_rows.append(dict(size=size, seed=seed, algo="ValueIteration",
                                   backups=backups, max_err=round(err, 6), policy_agree=round(ag, 4)))
        if vi_hit_backups is None and err < tol:
            vi_hit_backups = backups
            vi_hit_ms = (time.perf_counter() - t0) * 1000
        if vi_hit_backups is not None and backups >= max(checkpoints):
            break
    if vi_hit_backups is None:
        vi_hit_backups = backups
        vi_hit_ms = (time.perf_counter() - t0) * 1000

    # 2) Prioritized Sweeping: chay 1 luot duy nhat va theo doi sai so tung buoc
    import collections
    ps = valueIterationAgents.ValueIterationAgent(mdp, discount, iterations=0)
    t0 = time.perf_counter()
    predecessors = collections.defaultdict(set)
    for state in mdp.getStates():
        if not mdp.isTerminal(state):
            for action in mdp.getPossibleActions(state):
                for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)

    pq = util.PriorityQueue()
    for state in non_term:
        actions = mdp.getPossibleActions(state)
        if actions:
            maxQ = max(ps.computeQValueFromValues(state, a) for a in actions)
            pq.update(state, -abs(ps.values[state] - maxQ))

    ps_hit_backups = ps_hit_ms = None
    theta = 1e-5
    for k in range(1, 15001):
        if pq.isEmpty():
            if ps_hit_backups is None:
                ps_hit_backups = k
                ps_hit_ms = (time.perf_counter() - t0) * 1000
            break
        state = pq.pop()
        if not mdp.isTerminal(state):
            actions = mdp.getPossibleActions(state)
            if actions:
                ps.values[state] = max(ps.computeQValueFromValues(state, a) for a in actions)
        for p in predecessors[state]:
            if not mdp.isTerminal(p):
                actions = mdp.getPossibleActions(p)
                if actions:
                    maxQ_p = max(ps.computeQValueFromValues(p, a) for a in actions)
                    diff = abs(ps.values[p] - maxQ_p)
                    if diff > theta:
                        pq.update(p, -diff)
        if k in checkpoints or (ps_hit_backups is None and k % 25 == 0):
            err, ag = metrics_vs_ref(ps, v_star, opt_actions, non_term)
            if k in checkpoints:
                curve_rows.append(dict(size=size, seed=seed, algo="PrioritizedSweeping",
                                       backups=k, max_err=round(err, 6), policy_agree=round(ag, 4)))
            if ps_hit_backups is None and err < tol:
                ps_hit_backups = k
                ps_hit_ms = (time.perf_counter() - t0) * 1000
    if ps_hit_backups is None:
        ps_hit_backups = 15000
        ps_hit_ms = (time.perf_counter() - t0) * 1000

    summary = [
        dict(size="%dx%d" % (size, size), seed=seed, n_states=n_states,
             algo="ValueIteration", backups_to_tol=vi_hit_backups,
             sweeps_equiv=round(vi_hit_backups / n_states, 1),
             time_ms=round(vi_hit_ms, 2)),
        dict(size="%dx%d" % (size, size), seed=seed, n_states=n_states,
             algo="PrioritizedSweeping", backups_to_tol=ps_hit_backups,
             sweeps_equiv=round(ps_hit_backups / n_states, 1),
             time_ms=round(ps_hit_ms, 2)),
    ]
    print("  [E] %2dx%-2d seed=%d (%3d trang thai) | VI: %6d cap nhat (%4.0f vong, %6.1fms) | "
          "PS: %5d cap nhat (%4.1f vong, %6.1fms) -> giam %.1fx"
          % (size, size, seed, n_states, vi_hit_backups, vi_hit_backups / n_states, vi_hit_ms,
             ps_hit_backups, ps_hit_backups / n_states, ps_hit_ms,
             vi_hit_backups / max(1, ps_hit_backups)), flush=True)
    return summary, curve_rows


def experiment_e():
    summary, curves = [], []
    s, c = run_mdp_convergence(5, seed=0)
    summary += s
    curves += c
    for sz in (10, 20, 30):
        for seed in (1, 2, 3):
            s, c = run_mdp_convergence(sz, seed)
            summary += s
            curves += c
    return summary, curves


# ---------------------------------------------------------------------------
# Thi nghiem F: Tabular Q-Learning vs Approximate Q-Learning tren Pacman
# ---------------------------------------------------------------------------
TRAIN_BUDGETS = [50, 200, 500, 1000]
N_TEST = 20


def run_pacman_rl(args):
    layout_name, agent_kind, n_train, seed = args
    random.seed(seed)
    layout = lay_mod.getLayout(layout_name)
    ghosts = [ghostAgents.RandomGhost(i + 1) for i in range(layout.getNumGhosts())]
    if agent_kind == "Tabular Q-Learning":
        agent = qlearningAgents.PacmanQAgent(numTraining=n_train)
    else:
        agent = qlearningAgents.ApproximateQAgent(extractor="SimpleExtractor", numTraining=n_train)

    rules = pacman.ClassicGameRules(timeout=30)
    disp = textDisplay.NullGraphics()
    t0 = time.perf_counter()
    for ep in range(n_train):
        g = rules.newGame(layout, agent, ghosts, disp, quiet=True)
        g.run()
    train_s = time.perf_counter() - t0

    # Dam bao che do kiem tra (epsilon=0, alpha=0) ngay ca khi n_train=0
    agent.epsilon = 0.0
    agent.alpha = 0.0
    scores, wins, cleared = [], 0, []
    for ep in range(N_TEST):
        g = rules.newGame(layout, agent, ghosts, disp, quiet=True)
        g.run()
        scores.append(g.state.getScore())
        wins += int(g.state.isWin())
        cleared.append(1 - g.state.getNumFood() / layout.totalFood)

    n_params = (len(agent.qValues) if agent_kind == "Tabular Q-Learning"
                else len(agent.getWeights()))
    return dict(layout=layout_name, agent=agent_kind, n_train=n_train, seed=seed,
                win_rate=round(wins / N_TEST, 4),
                avg_score=round(sum(scores) / N_TEST, 1),
                food_cleared=round(sum(cleared) / N_TEST, 4),
                learned_params=n_params,
                train_seconds=round(train_s, 2))


def experiment_f():
    jobs = [(lay, ag, n, s)
            for lay in ("smallGrid", "mediumClassic")
            for ag in ("Tabular Q-Learning", "Approximate Q-Learning (SimpleExtractor)")
            for n in TRAIN_BUDGETS
            for s in (1, 2)]
    with Pool(min(8, os.cpu_count() or 2)) as pool:
        rows = []
        for r in pool.imap_unordered(run_pacman_rl, jobs):
            rows.append(r)
            print("  [F] %-14s %-42s N=%4d seed=%d | win=%5.1f%% score=%7.1f params=%6d (%.1fs)"
                  % (r["layout"], r["agent"], r["n_train"], r["seed"],
                     r["win_rate"] * 100, r["avg_score"], r["learned_params"], r["train_seconds"]),
                  flush=True)
    rows.sort(key=lambda r: (r["layout"], r["agent"], r["n_train"], r["seed"]))
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
    t = time.time()
    print("== Thi nghiem E: Value Iteration vs Prioritized Sweeping ==")
    s, c = experiment_e()
    write_csv(s, "rl_mdp_summary.csv")
    write_csv(c, "rl_mdp_curves.csv")
    print("== Thi nghiem F: Tabular vs Approximate Q-Learning tren Pacman ==")
    write_csv(experiment_f(), "rl_pacman.csv")
    print("Xong sau %.1f giay" % (time.time() - t))
