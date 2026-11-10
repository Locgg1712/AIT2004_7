"""
exp_multiagent.py - Danh gia cac tac tu doi khang (Project 2) tren Mega-City & Dau truong lon.

Thi nghiem C (thi dau doi khang):
    So sanh 6 tac tu tren 3 ban do (mediumClassic 20x11, MegaCity 31x21 3 Ghost, Arena 41x21 3 Ghost)
    voi 2 loai Ghost (RandomGhost va DirectionalGhost), 10 hat giong (seeds) moi o:
      1. Reflex                         : Tac tu phan xa 1 buoc (Project 2 Q1)
      2. AlphaBeta-d2 (scoreEval)       : Alpha-Beta voi ham diem tho
      3. AlphaBeta-d2 (betterEval)      : Alpha-Beta voi betterEvaluationFunction (Manhattan, terminal +-inf)
      4. Expectimax-d2 (betterEval)     : Expectimax voi betterEvaluationFunction (Manhattan, terminal +-inf)
      5. AlphaBeta-d2 (mazeEval)        : Alpha-Beta voi khoang cach me cung (BFS cache) & terminal huu han (+-10000)
      6. Expectimax-d2 (mazeEval)       : Expectimax voi khoang cach me cung (BFS cache) & terminal huu han (+-10000)

Thi nghiem D (hieu qua cat tia Alpha-Beta & Move Ordering - Chay don luong doc lap, median 5 lan):
    Tren cac trang thai lay mau tu van dau that tren mediumClassic (d=2, d=3), MegaCity (d=2), Arena (d=2):
    So sanh Minimax, AlphaBeta chuan, AlphaBeta + Move Ordering (sap xep nuoc di theo 1-ply eval), va Expectimax.
"""
import argparse
from collections import deque
import csv
import os
import random
import statistics
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

# --- Dem so trang thai sinh ra (generateSuccessor) khong sua code bai nop ------
NODES = [0]
_orig_gen = pacman.GameState.generateSuccessor


def _counting_gen(self, agentIndex, action):
    NODES[0] += 1
    return _orig_gen(self, agentIndex, action)


pacman.GameState.generateSuccessor = _counting_gen

MAX_MOVES = 600


class MoveCap(Exception):
    pass


# --- Khoang cach me cung (BFS co cache theo id(walls)) & Ham luong gia nang cap -
_MAZE_DIST_CACHE = {}


def get_wall_dist_map(walls):
    """Tinh truoc khoang cach BFS giua moi cap o trong cho mot luoi tuong (chi tinh 1 lan/ban do, tra cuu O(1))."""
    cached = getattr(walls, "_dist_map", None)
    if cached is not None:
        return cached
    key = (walls.width, walls.height)
    if key in _MAZE_DIST_CACHE:
        walls._dist_map = _MAZE_DIST_CACHE[key]
        return walls._dist_map
    W, H = walls.width, walls.height
    free = [(x, y) for x in range(W) for y in range(H) if not walls[x][y]]
    all_dists = {}
    for src in free:
        d = {src: 0}
        q = deque([src])
        while q:
            x, y = q.popleft()
            nd = d[(x, y)] + 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and not walls[nx][ny] and (nx, ny) not in d:
                    d[(nx, ny)] = nd
                    q.append((nx, ny))
        all_dists[src] = d
    _MAZE_DIST_CACHE[key] = all_dists
    walls._dist_map = all_dists
    return all_dists


def mazeBetterEvaluationFunction(currentGameState):
    """Ham luong gia nang cap cho ban do me cung lon:
      1. Dung gia tri ket thuc HUU HAN (+10000 / -10000) thay vi +-inf de Expectimax
         tinh dung ky vong rui ro xac suat thay vi bi -inf nuot mat toan bo trung binh.
      2. Dung khoang cach me cung that (BFS cache) thay vi Manhattan de khong bi ket sau tuong."""
    if currentGameState.isWin():
        return 10000.0 + currentGameState.getScore()
    if currentGameState.isLose():
        return -10000.0 + currentGameState.getScore()

    pos = currentGameState.getPacmanPosition()
    pos_int = (int(round(pos[0])), int(round(pos[1])))
    walls = currentGameState.getWalls()
    dist_map = get_wall_dist_map(walls).get(pos_int, {})

    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = float(currentGameState.getScore())

    if foodList:
        minFoodDist = min(dist_map.get(f, util.manhattanDistance(pos_int, f)) for f in foodList)
        score += 12.0 / (minFoodDist + 1.0)
        score -= 4.0 * len(foodList)

    if capsules:
        minCapDist = min(dist_map.get(c, util.manhattanDistance(pos_int, c)) for c in capsules)
        score += 6.0 / (minCapDist + 1.0)
        score -= 20.0 * len(capsules)

    for ghostState in ghostStates:
        gx, gy = ghostState.getPosition()
        g_int = (int(round(gx)), int(round(gy)))
        dist = dist_map.get(g_int, util.manhattanDistance(pos_int, g_int))
        if ghostState.scaredTimer > 0:
            if ghostState.scaredTimer > dist:
                score += 60.0 / (dist + 1.0)
        else:
            if dist <= 1:
                score -= 800.0
            elif dist <= 3:
                score -= 15.0 / (dist + 1.0)
            else:
                score -= 2.0 / (dist + 1.0)

    return score


import util  # noqa: E402
multiAgents.mazeBetterEvaluationFunction = mazeBetterEvaluationFunction


class OrderedAlphaBetaAgent(multiAgents.MultiAgentSearchAgent):
    """Alpha-Beta ket hop Move Ordering (sap xep nuoc di o nut goc va nut Pacman theo 1-ply eval)
    de tang ty le cat tia ma van dam bao 100% cung gia tri Minimax."""

    def getAction(self, gameState):
        numAgents = gameState.getNumAgents()

        def alphabeta(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                v = float("-inf")
                # Move ordering cho Pacman o tang tren
                if depth == 0 and len(legalActions) > 1:
                    succs = [(a, state.generateSuccessor(agentIndex, a)) for a in legalActions]
                    succs.sort(key=lambda item: self.evaluationFunction(item[1]), reverse=True)
                    for _, successor in succs:
                        v = max(v, alphabeta(successor, nextAgent, nextDepth, alpha, beta))
                        if v > beta:
                            return v
                        alpha = max(alpha, v)
                else:
                    for action in legalActions:
                        successor = state.generateSuccessor(agentIndex, action)
                        v = max(v, alphabeta(successor, nextAgent, nextDepth, alpha, beta))
                        if v > beta:
                            return v
                        alpha = max(alpha, v)
                return v
            else:
                v = float("inf")
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    v = min(v, alphabeta(successor, nextAgent, nextDepth, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        bestAction = None
        bestScore = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        nextAgent = 1 % numAgents
        nextDepth = 1 if nextAgent == 0 else 0

        # Sap xep cac nuoc di o goc theo 1-ply evaluation
        root_succs = [(a, gameState.generateSuccessor(0, a)) for a in gameState.getLegalActions(0)]
        root_succs.sort(key=lambda item: self.evaluationFunction(item[1]), reverse=True)

        for action, successor in root_succs:
            score = alphabeta(successor, nextAgent, nextDepth, alpha, beta)
            if score > bestScore or bestAction is None:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction


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
    "AlphaBeta-d2 (mazeEval)": lambda: multiAgents.AlphaBetaAgent("mazeBetterEvaluationFunction", "2"),
    "Expectimax-d2 (mazeEval)": lambda: multiAgents.ExpectimaxAgent("mazeBetterEvaluationFunction", "2"),
}
GHOSTS = {
    "Random": ghostAgents.RandomGhost,
    "Directional": ghostAgents.DirectionalGhost,
}


def get_maps():
    return {
        "mediumClassic (20x11, 2 ma)": lay_mod.getLayout("mediumClassic").layoutText,
        "MegaCity (31x21, 3 ma)": scenario.make_megacity("arena", seed=1),
        "Arena (41x21, 3 ma)": scenario.make_classic_arena(20, 10, 3, seed=1),
    }


def play(args):
    map_name, agent_name, ghost_name, seed = args
    lines = get_maps()[map_name]
    layout = lay_mod.Layout(lines)
    # Tien xu ly bang khoang cach me cung cho ban do nay truoc khi bat dau do thoi gian
    gs_init = pacman.GameState()
    gs_init.initialize(layout, layout.getNumGhosts())
    get_wall_dist_map(gs_init.getWalls())

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
                outcome=outcome,
                win=int(outcome == "win"),
                died=int(outcome == "lose"),
                timeout=int(outcome == "timeout"),
                score=state.getScore(),
                moves=len(pac.times),
                food_cleared=round(1 - state.getNumFood() / total_food, 4),
                ms_per_move=round(statistics.median(times), 3),
                p95_ms=round(times[int(0.95 * (len(times) - 1))], 3),
                max_ms=round(times[-1], 3),
                nodes_per_move=round(sum(pac.nodes) / max(1, len(pac.nodes)), 1),
                game_seconds=round(time.perf_counter() - t0, 2))


# --- Thi nghiem D: Hieu qua cat tia Alpha-Beta (Chay don luong doc lap) --------
def sample_states(lines, n, seed=0, every=6):
    """Lay mau trang thai tu van dau that (AlphaBeta mazeEval vs DirectionalGhost)."""
    layout = lay_mod.Layout(lines)
    random.seed(seed)
    collected = []

    class Recorder(game.Agent):
        def __init__(self):
            super().__init__(0)
            self.inner = multiAgents.AlphaBetaAgent("mazeBetterEvaluationFunction", "2")
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


def measure_median(agent, state, repeats=5):
    """Chay 1 lan warm-up va lay trung vi (median) cua `repeats` lan do doc lap."""
    NODES[0] = 0
    action = agent.getAction(state)
    nodes = NODES[0]
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        agent.getAction(state)
        times.append((time.perf_counter() - t0) * 1000)
    return action, nodes, statistics.median(times)


def run_pruning_sequential(prune_jobs):
    """Chay tuan tu tren 1 tien trinh duy nhat de khong bi nhieu CPU giua cac worker."""
    all_rows = []
    maps_dict = get_maps()
    for map_name, depth, n_states in prune_jobs:
        lines = maps_dict[map_name]
        states = sample_states(lines, n_states)
        repeats = 3 if depth >= 3 else 5
        for i, st in enumerate(states):
            mm = multiAgents.MinimaxAgent("betterEvaluationFunction", str(depth))
            ab = multiAgents.AlphaBetaAgent("betterEvaluationFunction", str(depth))
            oab = OrderedAlphaBetaAgent("betterEvaluationFunction", str(depth))
            ex = multiAgents.ExpectimaxAgent("betterEvaluationFunction", str(depth))

            a1, n1, t1 = measure_median(mm, st, repeats=repeats)
            a2, n2, t2 = measure_median(ab, st, repeats=repeats)
            a3, n3, t3 = measure_median(oab, st, repeats=repeats)
            _, n4, t4 = measure_median(ex, st, repeats=repeats)

            all_rows.append(dict(
                map=map_name, depth=depth, state_id=i,
                minimax_nodes=n1, alphabeta_nodes=n2, ordered_ab_nodes=n3, expectimax_nodes=n4,
                minimax_ms=round(t1, 3), alphabeta_ms=round(t2, 3),
                ordered_ab_ms=round(t3, 3), expectimax_ms=round(t4, 3),
                pruning_ratio=round(1 - n2 / max(1, n1), 4),
                ordered_pruning_ratio=round(1 - n3 / max(1, n1), 4),
                speedup=round(t1 / max(t2, 1e-6), 3),
                ordered_speedup=round(t1 / max(t3, 1e-6), 3),
                action_match=int(a1 == a2),
            ))
        print(f"  [D] {map_name:28s} d={depth} ({len(states)} states) completed.")
    return all_rows


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
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    opt = ap.parse_args()
    t = time.time()
    maps = list(get_maps())
    jobs = [(m, a, gname, s) for m in maps for a in AGENTS for gname in GHOSTS
            for s in range(1, opt.games + 1)]
    print("== Thi nghiem C (%d van) voi %d tien trinh ==" % (len(jobs), opt.workers))
    with Pool(opt.workers) as pool:
        rows = []
        for r in pool.imap_unordered(play, jobs):
            rows.append(r)
            print("  [C] %-26s %-26s %-11s s=%-2d %-7s sc=%6.0f mv=%3d fd=%.0f%%" % (
                r["map"], r["agent"], r["ghost"], r["seed"], r["outcome"], r["score"],
                r["moves"], r["food_cleared"] * 100), flush=True)
    rows.sort(key=lambda r: (r["map"], r["agent"], r["ghost"], r["seed"]))
    write_csv(rows, "multiagent_games.csv")

    print("== Thi nghiem D (Cat tia Alpha-Beta & Move Ordering - Chay don luong doc lap) ==")
    prune_jobs = [
        (maps[0], 2, 20),
        (maps[0], 3, 10),
        (maps[1], 2, 20),
        (maps[2], 2, 20),
    ]
    prune_rows = run_pruning_sequential(prune_jobs)
    write_csv(prune_rows, "multiagent_pruning.csv")
    print("Xong sau %.1f giay" % (time.time() - t))
