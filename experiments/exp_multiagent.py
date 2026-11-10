"""
exp_multiagent.py - Danh gia cac tac tu doi khang (Project 2) tren dau truong Pacman lon.

Thi nghiem C (thi dau): Reflex, AlphaBeta d2 (scoreEval), AlphaBeta d2 (betterEval),
    Expectimax d2 (betterEval) choi tren
        - mediumClassic (20x11, 2 ma)          : ban do goc cua do an (moc so sanh)
        - arena 41x21, 3 ma (kich ban moi)     : ban do lon, nhieu ma hon
    voi 2 loai ma: RandomGhost (ngau nhien) va DirectionalGhost (duoi bat Pacman).

Thi nghiem D (hieu qua cat tia): tren cac trang thai lay mau tu van dau that,
    so sanh Minimax va AlphaBeta cung do sau: so nut sinh ra, thoi gian, va kiem tra
    hai thuat toan chon cung mot hanh dong.

Do do (metrics):
    win_rate        ty le thang (kem khoang tin cay Wilson 95%)
    score           diem trung binh (+- do lech chuan)
    food_cleared    ty le thuc an da an duoc (muc do hoan thanh nhiem vu)
    moves           so nuoc di cua Pacman moi van
    ms_per_move     thoi gian quyet dinh trung binh / nuoc (p95, max)
    nodes_per_move  so trang thai sinh ra (generateSuccessor) trung binh / nuoc
    pruning_ratio   1 - nodes(AlphaBeta) / nodes(Minimax)
    action_match    ty le AlphaBeta chon cung hanh dong voi Minimax (ky vong 100%)

Chay:  python exp_multiagent.py [--games N]
"""
import argparse
import csv
import os
import random
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(os.path.dirname(HERE), "multiagent")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, PROJ)
sys.path.append(HERE)
os.chdir(PROJ)

import layout as lay_mod      # noqa: E402
import pacman                 # noqa: E402
import game                   # noqa: E402
import textDisplay            # noqa: E402
import ghostAgents            # noqa: E402
import multiAgents            # noqa: E402
import scenario               # noqa: E402

# --- Dem so trang thai sinh ra ma khong sua code cua do an ---------------------
NODES = [0]
_orig_gen = pacman.GameState.generateSuccessor


def _counting_gen(self, agentIndex, action):
    NODES[0] += 1
    return _orig_gen(self, agentIndex, action)


pacman.GameState.generateSuccessor = _counting_gen

MAX_MOVES = 600


class MoveCap(Exception):
    pass


class Instrumented(game.Agent):
    """Boc tac tu Pacman de do thoi gian va so nut cho tung nuoc di."""

    def __init__(self, inner):
        super().__init__(0)
        self.inner = inner
        self.times, self.nodes = [], []

    def registerInitialState(self, state):
        if hasattr(self.inner, "registerInitialState"):
            self.inner.registerInitialState(state)

    def getAction(self, state):
        if len(self.times) >= MAX_MOVES:
            raise MoveCap()
        NODES[0] = 0
        t0 = time.perf_counter()
        a = self.inner.getAction(state)
        self.times.append((time.perf_counter() - t0) * 1000)
        self.nodes.append(NODES[0])
        return a


AGENTS = {
    "Reflex": lambda: multiAgents.ReflexAgent(),
    "AlphaBeta-d2 (scoreEval)": lambda: multiAgents.AlphaBetaAgent("scoreEvaluationFunction", "2"),
    "AlphaBeta-d2 (betterEval)": lambda: multiAgents.AlphaBetaAgent("betterEvaluationFunction", "2"),
    "Expectimax-d2 (betterEval)": lambda: multiAgents.ExpectimaxAgent("betterEvaluationFunction", "2"),
}
GHOSTS = {
    "Random": ghostAgents.RandomGhost,
    "Directional": ghostAgents.DirectionalGhost,
}


def get_maps():
    return {
        "mediumClassic (goc, 20x11, 2 ma)": lay_mod.getLayout("mediumClassic").layoutText,
        "Arena 41x21, 3 ma (moi)": scenario.make_classic_arena(20, 10, 3, seed=1),
    }


def play(args):
    map_name, agent_name, ghost_name, seed = args
    lines = get_maps()[map_name]
    layout = lay_mod.Layout(lines)
    random.seed(seed)
    pac = Instrumented(AGENTS[agent_name]())
    ghosts = [GHOSTS[ghost_name](i + 1) for i in range(layout.getNumGhosts())]
    rules = pacman.ClassicGameRules(timeout=10 ** 6)
    g = rules.newGame(layout, pac, ghosts, textDisplay.NullGraphics(), quiet=True)
    outcome = None
    t0 = time.perf_counter()
    try:
        g.run()
    except MoveCap:
        outcome = "timeout"
    state = g.state
    if outcome is None:
        outcome = "win" if state.isWin() else "lose"
    total_food = layout.totalFood
    times = sorted(pac.times) or [0.0]
    return dict(map=map_name, agent=agent_name, ghost=ghost_name, seed=seed,
                outcome=outcome, win=int(outcome == "win"), score=state.getScore(),
                moves=len(pac.times),
                food_cleared=round(1 - state.getNumFood() / total_food, 4),
                ms_per_move=round(sum(times) / len(times), 3),
                p95_ms=round(times[int(0.95 * (len(times) - 1))], 3),
                max_ms=round(times[-1], 3),
                nodes_per_move=round(sum(pac.nodes) / max(1, len(pac.nodes)), 1),
                game_seconds=round(time.perf_counter() - t0, 2))


# --- Thi nghiem D: hieu qua cat tia alpha-beta ---------------------------------
def sample_states(lines, n, seed=0, every=7):
    """Lay mau trang thai tu mot van dau that (AlphaBeta betterEval vs DirectionalGhost)."""
    layout = lay_mod.Layout(lines)
    random.seed(seed)
    collected = []

    class Recorder(game.Agent):
        def __init__(self):
            super().__init__(0)
            self.inner = multiAgents.AlphaBetaAgent("betterEvaluationFunction", "2")
            self.t = 0

        def getAction(self, state):
            if self.t % every == 0 and len(collected) < n:
                collected.append(state)
            self.t += 1
            if len(collected) >= n:
                raise MoveCap()
            return self.inner.getAction(state)

    ghosts = [ghostAgents.DirectionalGhost(i + 1) for i in range(layout.getNumGhosts())]
    rules = pacman.ClassicGameRules(timeout=10 ** 6)
    g = rules.newGame(layout, Recorder(), ghosts, textDisplay.NullGraphics(), quiet=True)
    try:
        g.run()
    except MoveCap:
        pass
    return collected


def measure(agent, state):
    NODES[0] = 0
    t0 = time.perf_counter()
    a = agent.getAction(state)
    return a, NODES[0], (time.perf_counter() - t0) * 1000


def pruning_job(args):
    map_name, depth, n_states = args
    lines = get_maps()[map_name]
    rows = []
    for i, st in enumerate(sample_states(lines, n_states)):
        mm = multiAgents.MinimaxAgent("betterEvaluationFunction", str(depth))
        ab = multiAgents.AlphaBetaAgent("betterEvaluationFunction", str(depth))
        ex = multiAgents.ExpectimaxAgent("betterEvaluationFunction", str(depth))
        a1, n1, t1 = measure(mm, st)
        a2, n2, t2 = measure(ab, st)
        _, n3, t3 = measure(ex, st)
        rows.append(dict(map=map_name, depth=depth, state_id=i,
                         minimax_nodes=n1, alphabeta_nodes=n2, expectimax_nodes=n3,
                         minimax_ms=round(t1, 2), alphabeta_ms=round(t2, 2),
                         expectimax_ms=round(t3, 2),
                         pruning_ratio=round(1 - n2 / n1, 4),
                         speedup=round(t1 / max(t2, 1e-6), 3),
                         action_match=int(a1 == a2)))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    opt = ap.parse_args()
    t = time.time()
    maps = list(get_maps())
    jobs = [(m, a, gname, s) for m in maps for a in AGENTS for gname in GHOSTS
            for s in range(opt.games)]
    prune_jobs = [(maps[0], 2, 20), (maps[0], 3, 10), (maps[1], 2, 20)]
    print("== Thi nghiem C (%d van) + D (cat tia) voi %d tien trinh ==" % (len(jobs), opt.workers))
    with Pool(opt.workers) as pool:
        prune_async = pool.map_async(pruning_job, prune_jobs)
        rows = []
        for r in pool.imap_unordered(play, jobs):
            rows.append(r)
            print("  [C] %-34s %-27s %-11s seed=%-2d %-7s score=%6.0f moves=%4d %.1fs" % (
                r["map"], r["agent"], r["ghost"], r["seed"], r["outcome"], r["score"],
                r["moves"], r["game_seconds"]), flush=True)
        prune_rows = [x for part in prune_async.get() for x in part]
    rows.sort(key=lambda r: (r["map"], r["agent"], r["ghost"], r["seed"]))
    write_csv(rows, "multiagent_games.csv")
    write_csv(prune_rows, "multiagent_pruning.csv")
    print("Xong sau %.1f giay" % (time.time() - t))
