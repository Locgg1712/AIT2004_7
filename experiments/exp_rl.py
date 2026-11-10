"""
exp_rl.py - Danh gia cac thuat toan quy hoach & hoc tang cuong (Project 3) tren kich ban lon & Mega-City.

Thi nghiem E (quy hoach co mo hinh - MDP):
    So sanh ValueIterationAgent va PrioritizedSweepingValueIterationAgent tren:
        - DiscountGrid 5x5 (ban do goc cua do an)
        - Gridworld lon 10x10, 20x20, 30x30, 40x40 (seeds 1..3, co tuong, ho bay -10, loi ra +1 va +10)
    Do do:
        - backups_to_tol : so lan cap nhat gia tri trang thai V(s) de dat max|V - V*| < 1e-3
        - q_evals_to_tol : tong so lan tinh Q(s,a) (bao gom ca kiem tra tien nhiem cua Prioritized Sweeping)
        - pure_time_ms   : thoi gian thuat toan thuan tuy (tach biet hoan toan khoi chi phi do metrics_vs_ref)

Thi nghiem F (hoc khong mo hinh tren Pacman & Chuyen giao Zero-Shot sang Mega-City):
    So sanh 4 tac tu:
        1. Tabular Q-Learning               : Bang tra Q(s,a) roi rac
        2. ApproxQ (IdentityExtractor)      : Xap xi tuyen tinh nhung dung dac trung dinh danh (doi chung)
        3. ApproxQ (SimpleExtractor)        : Xap xi tuyen tinh voi 4 dac trung co ban
        4. ApproxQ (MegaCityExtractor)      : Xap xi tuyen tinh voi 7 dac trung nang cao (capsule, scared ghost, ...)
    Tren 3 boi canh (N = 50, 200, 500, 1000 van huan luyen, N_TEST = 30 van kiem tra, 3 seeds):
        - smallGrid (7x7, 1 Ghost)
        - mediumClassic (20x11, 2 Ghosts)
        - MegaCity-Transfer (31x21, 3 Ghosts): Huan luyen tren mediumClassic, kiem tra ZERO-SHOT tren Mega-City!
"""
import collections
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
import game                   # noqa: E402
import textDisplay            # noqa: E402
import ghostAgents            # noqa: E402
import valueIterationAgents   # noqa: E402
import qlearningAgents        # noqa: E402
import featureExtractors      # noqa: E402
import util                   # noqa: E402
import scenario               # noqa: E402


# ---------------------------------------------------------------------------
# Custom Feature Extractor cho Mega-City & Dau truong nhieu Ghost
# ---------------------------------------------------------------------------
class MegaCityExtractor(featureExtractors.FeatureExtractor):
    """Bo trich xuat 7 dac trung cho ban do lon co nhieu Ghost va Capsule:
      - bias
      - #-of-active-ghosts-1-step-away
      - #-of-active-ghosts-2-steps-away
      - scared-ghost-proximity (san Ghost khi an Capsule)
      - eats-capsule-or-food
      - closest-food
      - closest-capsule-when-threatened"""

    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        capsules = state.getCapsules()
        ghostStates = state.getGhostStates()

        features = util.Counter()
        features["bias"] = 1.0

        x, y = state.getPacmanPosition()
        dx, dy = game.Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        active_ghosts = []
        scared_ghosts = []
        for g in ghostStates:
            if g.scaredTimer > 1:
                scared_ghosts.append(g)
            else:
                active_ghosts.append(g)

        # 1) Ghost nguy hiem o khoang cach 1 buoc va 2 buoc
        g_1step = sum((next_x, next_y) in game.Actions.getLegalNeighbors(g.getPosition(), walls)
                      for g in active_ghosts)
        g_2step = sum(util.manhattanDistance((next_x, next_y), g.getPosition()) <= 2
                      for g in active_ghosts)
        features["#-of-ghosts-1-step-away"] = float(g_1step)
        features["#-of-ghosts-2-steps-away"] = float(g_2step) / 4.0

        # 2) San Ghost dang hoac so (scared)
        if scared_ghosts and g_1step == 0:
            min_scared = min(util.manhattanDistance((next_x, next_y), g.getPosition())
                             for g in scared_ghosts)
            features["scared-ghost-proximity"] = 1.0 / (min_scared + 1.0)

        # 3) An thuc an hoac vien nang luong (Capsule)
        if not g_1step:
            if (next_x, next_y) in capsules:
                features["eats-food"] = 1.5
            elif food[next_x][next_y]:
                features["eats-food"] = 1.0

        # 4) Khoang cach me cung toi thuc an gan nhat
        dist = featureExtractors.closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        # 5) Huong toi Capsule khi co Ghost o gan
        if capsules and active_ghosts:
            min_cap = min(util.manhattanDistance((next_x, next_y), c) for c in capsules)
            features["closest-capsule"] = float(min_cap) / (walls.width + walls.height)

        features.divideAll(10.0)
        return features


qlearningAgents.MegaCityExtractor = MegaCityExtractor


# ---------------------------------------------------------------------------
# Thi nghiem E: Value Iteration vs Prioritized Sweeping (Do ca Backups & Q-evals)
# ---------------------------------------------------------------------------
def build_mdp(size, seed, discount=0.9, noise=0.2, living=-0.04):
    if size == 5:
        gw = gridworld.getDiscountGrid()
    else:
        gw = gridworld.Gridworld(scenario.make_gridworld(size, size, seed))
    gw.setNoise(noise)
    gw.setLivingReward(living)
    return gw, discount


def reference_vi(mdp, discount, tol=1e-7, max_iters=400):
    """Chay Value Iteration den hoi tu chat de lay V* va chinh sach toi uu pi*."""
    agent = valueIterationAgents.ValueIterationAgent(mdp, discount, iterations=0)
    non_term = [s for s in mdp.getStates() if not mdp.isTerminal(s)]
    for _ in range(max_iters):
        new_v = util.Counter()
        diff = 0.0
        for s in non_term:
            acts = mdp.getPossibleActions(s)
            if acts:
                new_v[s] = max(agent.computeQValueFromValues(s, a) for a in acts)
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


def max_err_vs_ref(agent, v_star, non_term):
    return max(abs(agent.values[s] - v_star[s]) for s in non_term)


def metrics_vs_ref(agent, v_star, opt_actions, non_term):
    max_err = max(abs(agent.values[s] - v_star[s]) for s in non_term)
    agree = sum(1 for s in non_term if agent.computeActionFromValues(s) in opt_actions[s])
    return max_err, agree / len(non_term)


def run_mdp_convergence(size, seed, tol=1e-3):
    mdp, discount = build_mdp(size, seed)
    v_star, opt_actions, non_term = reference_vi(mdp, discount)
    n_states = len(non_term)
    checkpoints = [50, 100, 250, 500, 1000, 2000, 4000, 8000, 15000, 25000, 40000]
    curve_rows = []

    # 1) Value Iteration (tach rieng thoi gian thuat toan khoi metrics_vs_ref)
    vi = valueIterationAgents.ValueIterationAgent(mdp, discount, iterations=0)
    backups = 0
    q_evals = 0
    pure_ms = 0.0
    vi_hit_backups = vi_hit_qevals = vi_hit_ms = None
    next_cp_idx = 0

    for it in range(1, 351):
        t0 = time.perf_counter()
        new_v = util.Counter()
        for s in non_term:
            acts = mdp.getPossibleActions(s)
            if acts:
                q_evals += len(acts)
                new_v[s] = max(vi.computeQValueFromValues(s, a) for a in acts)
        vi.values = new_v
        backups += n_states
        pure_ms += (time.perf_counter() - t0) * 1000

        err = max_err_vs_ref(vi, v_star, non_term)
        if next_cp_idx < len(checkpoints) and backups >= checkpoints[next_cp_idx]:
            err, ag = metrics_vs_ref(vi, v_star, opt_actions, non_term)
            while next_cp_idx < len(checkpoints) and backups >= checkpoints[next_cp_idx]:
                cp = checkpoints[next_cp_idx]
                curve_rows.append(dict(size=size, seed=seed, algo="ValueIteration",
                                       backups=cp, q_evals=q_evals,
                                       max_err=round(err, 6), policy_agree=round(ag, 4)))
                next_cp_idx += 1
        if vi_hit_backups is None and err < tol:
            vi_hit_backups = backups
            vi_hit_qevals = q_evals
            vi_hit_ms = pure_ms
        if vi_hit_backups is not None and (next_cp_idx >= len(checkpoints) or err < 1e-6):
            if next_cp_idx < len(checkpoints):
                err, ag = metrics_vs_ref(vi, v_star, opt_actions, non_term)
                while next_cp_idx < len(checkpoints):
                    cp = checkpoints[next_cp_idx]
                    curve_rows.append(dict(size=size, seed=seed, algo="ValueIteration",
                                           backups=cp, q_evals=q_evals,
                                           max_err=round(err, 6), policy_agree=round(ag, 4)))
                    next_cp_idx += 1
            break

    if vi_hit_backups is None:
        vi_hit_backups = backups
        vi_hit_qevals = q_evals
        vi_hit_ms = pure_ms

    # 2) Prioritized Sweeping (do ca pop backups va tong so q_evals tinh lai tien nhiem)
    ps = valueIterationAgents.ValueIterationAgent(mdp, discount, iterations=0)
    ps_q_evals = 0
    t0 = time.perf_counter()
    predecessors = collections.defaultdict(set)
    for state in non_term:
        for action in mdp.getPossibleActions(state):
            for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
                if prob > 0:
                    predecessors[nextState].add(state)

    pq = util.PriorityQueue()
    for state in non_term:
        actions = mdp.getPossibleActions(state)
        if actions:
            ps_q_evals += len(actions)
            maxQ = max(ps.computeQValueFromValues(state, a) for a in actions)
            pq.update(state, -abs(ps.values[state] - maxQ))
    ps_pure_ms = (time.perf_counter() - t0) * 1000

    ps_hit_backups = ps_hit_qevals = ps_hit_ms = None
    theta = 1e-5
    max_k = checkpoints[-1]
    cp_set = set(checkpoints)

    for k in range(1, max_k + 1):
        t0 = time.perf_counter()
        if pq.isEmpty():
            ps_pure_ms += (time.perf_counter() - t0) * 1000
            if ps_hit_backups is None:
                ps_hit_backups = k
                ps_hit_qevals = ps_q_evals
                ps_hit_ms = ps_pure_ms
            err, ag = metrics_vs_ref(ps, v_star, opt_actions, non_term)
            for cp in checkpoints:
                if cp >= k:
                    curve_rows.append(dict(size=size, seed=seed, algo="PrioritizedSweeping",
                                           backups=cp, q_evals=ps_q_evals,
                                           max_err=round(err, 6), policy_agree=round(ag, 4)))
            break

        state = pq.pop()
        actions = mdp.getPossibleActions(state)
        if actions:
            ps_q_evals += len(actions)
            ps.values[state] = max(ps.computeQValueFromValues(state, a) for a in actions)

        for p in predecessors[state]:
            p_acts = mdp.getPossibleActions(p)
            if p_acts:
                ps_q_evals += len(p_acts)
                maxQ_p = max(ps.computeQValueFromValues(p, a) for a in p_acts)
                diff = abs(ps.values[p] - maxQ_p)
                if diff > theta:
                    pq.update(p, -diff)
        ps_pure_ms += (time.perf_counter() - t0) * 1000

        if ps_hit_backups is None and k % 25 == 0:
            err = max_err_vs_ref(ps, v_star, non_term)
            if err < tol:
                ps_hit_backups = k
                ps_hit_qevals = ps_q_evals
                ps_hit_ms = ps_pure_ms

        if k in cp_set:
            err, ag = metrics_vs_ref(ps, v_star, opt_actions, non_term)
            curve_rows.append(dict(size=size, seed=seed, algo="PrioritizedSweeping",
                                   backups=k, q_evals=ps_q_evals,
                                   max_err=round(err, 6), policy_agree=round(ag, 4)))
            if ps_hit_backups is not None and err < 1e-5:
                for cp in checkpoints:
                    if cp > k:
                        curve_rows.append(dict(size=size, seed=seed, algo="PrioritizedSweeping",
                                               backups=cp, q_evals=ps_q_evals,
                                               max_err=round(err, 6), policy_agree=round(ag, 4)))
                break

    if ps_hit_backups is None:
        ps_hit_backups = max_k
        ps_hit_qevals = ps_q_evals
        ps_hit_ms = ps_pure_ms

    summary = [
        dict(size="%dx%d" % (size, size), seed=seed, n_states=n_states,
             algo="ValueIteration",
             backups_to_tol=vi_hit_backups,
             q_evals_to_tol=vi_hit_qevals,
             sweeps_equiv=round(vi_hit_backups / n_states, 1),
             time_ms=round(vi_hit_ms, 2)),
        dict(size="%dx%d" % (size, size), seed=seed, n_states=n_states,
             algo="PrioritizedSweeping",
             backups_to_tol=ps_hit_backups,
             q_evals_to_tol=ps_hit_qevals,
             sweeps_equiv=round(ps_hit_backups / n_states, 1),
             time_ms=round(ps_hit_ms, 2)),
    ]
    print("  [E] %2dx%-2d s=%d (%4d st) | VI: %6d bkp / %7d Q (%.1fms) | "
          "PS: %5d bkp / %7d Q (%.1fms) -> bkp %.1fx, Q %.2fx"
          % (size, size, seed, n_states,
             vi_hit_backups, vi_hit_qevals, vi_hit_ms,
             ps_hit_backups, ps_hit_qevals, ps_hit_ms,
             vi_hit_backups / max(1, ps_hit_backups),
             vi_hit_qevals / max(1, ps_hit_qevals)), flush=True)
    return summary, curve_rows


def experiment_e():
    summary, curves = [], []
    s, c = run_mdp_convergence(5, seed=0)
    summary += s
    curves += c
    for sz in (10, 20, 30, 40):
        for seed in (1, 2, 3):
            s, c = run_mdp_convergence(sz, seed)
            summary += s
            curves += c
    return summary, curves


# ---------------------------------------------------------------------------
# Thi nghiem F: Tabular vs Identity vs Simple vs MegaCityExtractor + Transfer
# ---------------------------------------------------------------------------
TRAIN_BUDGETS = [50, 200, 500, 1000]
N_TEST = 30
RL_SEEDS = [1, 2, 3]
RL_AGENTS = (
    "Tabular Q-Learning",
    "ApproxQ (IdentityExtractor)",
    "ApproxQ (SimpleExtractor)",
    "ApproxQ (MegaCityExtractor)",
)


class _StepCap(Exception):
    pass


class _CappedAgent(game.Agent):
    """Gioi han 500 buoc moi van test tren ban do lon de tranh ket vo han khi chua hoc duoc."""
    def __init__(self, base, max_steps=500):
        super().__init__(0)
        self.base = base
        self.max_steps = max_steps
        self.steps = 0

    def registerInitialState(self, state):
        self.steps = 0
        if hasattr(self.base, "registerInitialState"):
            self.base.registerInitialState(state)

    def getAction(self, state):
        self.steps += 1
        if self.steps > self.max_steps:
            raise _StepCap()
        return self.base.getAction(state)

    def final(self, state):
        if hasattr(self.base, "final"):
            self.base.final(state)


def make_rl_agent(agent_kind, n_train):
    if agent_kind == "Tabular Q-Learning":
        return qlearningAgents.PacmanQAgent(numTraining=n_train)
    if agent_kind == "ApproxQ (IdentityExtractor)":
        return qlearningAgents.ApproximateQAgent(extractor="IdentityExtractor", numTraining=n_train)
    if agent_kind == "ApproxQ (SimpleExtractor)":
        return qlearningAgents.ApproximateQAgent(extractor="SimpleExtractor", numTraining=n_train)
    if agent_kind == "ApproxQ (MegaCityExtractor)":
        return qlearningAgents.ApproximateQAgent(extractor="MegaCityExtractor", numTraining=n_train)
    raise ValueError(agent_kind)


def _eval_trained_agent(agent, test_layout, n_test, max_steps=500):
    test_ghosts = [ghostAgents.RandomGhost(i + 1) for i in range(test_layout.getNumGhosts())]
    rules = pacman.ClassicGameRules(timeout=30)
    disp = textDisplay.NullGraphics()
    scores, wins, cleared, moves_list = [], 0, [], []
    test_wrapper = _CappedAgent(agent, max_steps=max_steps)

    for _ in range(n_test):
        g = rules.newGame(test_layout, test_wrapper, test_ghosts, disp, quiet=True)
        try:
            g.run()
        except _StepCap:
            pass
        scores.append(g.state.getScore())
        wins += int(g.state.isWin())
        cleared.append(1 - g.state.getNumFood() / test_layout.totalFood)
        moves_list.append(test_wrapper.steps)
    return wins, scores, cleared, moves_list


def run_pacman_rl(args):
    scenario_name, agent_kind, n_train, seed = args
    random.seed(seed * 100 + n_train)

    if scenario_name == "smallGrid":
        train_layout = lay_mod.getLayout("smallGrid")
    elif scenario_name == "mediumClassic":
        train_layout = lay_mod.getLayout("mediumClassic")
    else:
        raise ValueError(scenario_name)

    agent = make_rl_agent(agent_kind, n_train)
    train_ghosts = [ghostAgents.RandomGhost(i + 1) for i in range(train_layout.getNumGhosts())]

    rules = pacman.ClassicGameRules(timeout=30)
    disp = textDisplay.NullGraphics()
    t0 = time.perf_counter()
    for _ in range(n_train):
        g = rules.newGame(train_layout, agent, train_ghosts, disp, quiet=True)
        g.run()
    train_s = time.perf_counter() - t0

    # Chuyen sang che do kiem tra (tat hoc & kham pha)
    agent.epsilon = 0.0
    agent.alpha = 0.0
    n_params = (len(agent.qValues) if agent_kind == "Tabular Q-Learning"
                else len(agent.getWeights()))

    wins, scores, cleared, moves_list = _eval_trained_agent(agent, train_layout, N_TEST, max_steps=500)
    out_rows = [
        dict(
            layout=scenario_name,
            agent=agent_kind,
            n_train=n_train,
            seed=seed,
            win_rate=round(wins / N_TEST, 4),
            avg_score=round(sum(scores) / N_TEST, 1),
            food_cleared=round(sum(cleared) / N_TEST, 4),
            avg_moves=round(sum(moves_list) / N_TEST, 1),
            learned_params=n_params,
            train_seconds=round(train_s, 2),
        )
    ]

    # Neu huan luyen tren mediumClassic, danh gia ZERO-SHOT chinh tac tu vua huan luyen tren MegaCity 31x21 (3 Ghosts)!
    if scenario_name == "mediumClassic":
        mc_layout = lay_mod.Layout(scenario.make_megacity("arena", seed=1))
        random.seed(seed * 1000 + n_train)
        w_mc, s_mc, c_mc, m_mc = _eval_trained_agent(agent, mc_layout, N_TEST, max_steps=500)
        out_rows.append(
            dict(
                layout="MegaCity-Transfer",
                agent=agent_kind,
                n_train=n_train,
                seed=seed,
                win_rate=round(w_mc / N_TEST, 4),
                avg_score=round(sum(s_mc) / N_TEST, 1),
                food_cleared=round(sum(c_mc) / N_TEST, 4),
                avg_moves=round(sum(m_mc) / N_TEST, 1),
                learned_params=n_params,
                train_seconds=round(train_s, 2),
            )
        )
    return out_rows


def experiment_f():
    jobs = [
        (lay, ag, n, s)
        for lay in ("smallGrid", "mediumClassic")
        for ag in RL_AGENTS
        for n in TRAIN_BUDGETS
        for s in RL_SEEDS
    ]
    rows = []
    with Pool(min(6, os.cpu_count() or 1)) as pool:
        for res_list in pool.imap_unordered(run_pacman_rl, jobs):
            for r in res_list:
                rows.append(r)
                print("  [F] %-17s %-28s N=%-4d s=%d | win=%5.1f%% score=%7.1f food=%5.1f%% params=%d"
                      % (r["layout"], r["agent"], r["n_train"], r["seed"],
                         r["win_rate"] * 100, r["avg_score"], r["food_cleared"] * 100,
                         r["learned_params"]), flush=True)
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
    print("== Thi nghiem F: Tabular vs Identity vs Simple vs MegaCityExtractor + Transfer ==")
    write_csv(experiment_f(), "rl_pacman.csv")
    print("Xong sau %.1f giay" % (time.time() - t))
