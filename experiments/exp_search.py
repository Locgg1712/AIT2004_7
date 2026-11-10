"""
exp_search.py - Danh gia cac thuat toan tim kiem (Project 1) tren kich ban me cung lon & Mega-City.

Thi nghiem A (mo rong quy mo & dam lay):
    DFS, BFS, UCS, A*-Manhattan, A*-Euclid, WA*-Manhattan (w=2.0) tren:
      - bigMaze (37x37, moc chuan CS188)
      - MegaCity-Route (31x21, thanh pho thong nhat co/khong co dam lay)
      - Me cung braided 31x31, 61x61, 101x101, 151x151 (seeds 1..5, unit & swamp cost=5).

Thi nghiem B (bai toan da muc tieu FoodSearchProblem):
    So sanh 6 phuong phap tren me cung 31x31 voi k = 3, 5, 7, 9, 11 vien thuc an (seeds 1..3):
      1. UCS                              : Tim kiem chi phi dong nhat khong heuristic
      2. A* + foodHeuristic               : Heuristic max d_maze(pos, f) tinh BFS on-the-fly (Project 1 Q7)
      3. A* + foodHeuristic (precomputed) : Cung heuristic max d_maze(pos, f) nhung tinh truoc BFS tu k vien
      4. A* + MSTHeuristic (precomputed)  : Heuristic min_f d(pos,f) + MST(Food) (admissible & consistent chat hon)
      5. WA* + MST (w=2.0)                : Weighted A* (w=2.0) danh doi ty le toi uu lay toc do
      6. ClosestDot (greedy)              : Tham lam BFS toi vien gan nhat (Project 1 Q8)
"""
from collections import deque
import contextlib
import csv
import io
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(os.path.dirname(HERE), "search")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, PROJ)
sys.path.append(HERE)
os.chdir(PROJ)

import layout as lay_mod      # noqa: E402
import pacman                 # noqa: E402
import search                 # noqa: E402
import searchAgents           # noqa: E402
import util                   # noqa: E402
import scenario               # noqa: E402

# ---------------------------------------------------------------------------
# Theo doi kich thuoc frontier lon nhat (peak_frontier) khong can sua search.py
# ---------------------------------------------------------------------------
_CONTAINERS = []


def _tracked(base, attr):
    class Tracked(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._peak = 0
            _CONTAINERS.append(self)

        def push(self, *a, **k):
            super().push(*a, **k)
            n = len(getattr(self, attr))
            if n > self._peak:
                self._peak = n
    Tracked.__name__ = base.__name__
    return Tracked


util.Stack = _tracked(util.Stack, "list")
util.Queue = _tracked(util.Queue, "list")
util.PriorityQueue = _tracked(util.PriorityQueue, "heap")


def reset_tracking():
    _CONTAINERS.clear()


def peak_frontier():
    return _CONTAINERS[0]._peak if _CONTAINERS else 0


def make_state(lines):
    gs = pacman.GameState()
    gs.initialize(lay_mod.Layout(lines), 0)
    return gs


def weighted_astar(problem, heuristic, w=2.0):
    """Weighted A* voi f(n) = g(n) + w * h(n)."""
    return search.astar(problem, lambda s, p=None: w * heuristic(s, p))


# ---------------------------------------------------------------------------
# Thi nghiem A: Mo rong quy mo me cung & Dia hinh dam lay
# ---------------------------------------------------------------------------
ALGOS_A = {
    "DFS": lambda p: search.dfs(p),
    "BFS": lambda p: search.bfs(p),
    "UCS": lambda p: search.ucs(p),
    "A*-Manhattan": lambda p: search.astar(p, searchAgents.manhattanHeuristic),
    "A*-Euclid": lambda p: search.astar(p, searchAgents.euclideanHeuristic),
    "WA*-Manhattan (w=2.0)": lambda p: weighted_astar(p, searchAgents.manhattanHeuristic, w=2.0),
}
SIZES_A = [(15, 15), (30, 30), (50, 50), (75, 75)]   # -> 31x31, 61x61, 101x101, 151x151
SEEDS_A = [1, 2, 3, 4, 5]


def run_position(lines, terrain, algo_name):
    gs = make_state(lines)
    swamps = scenario.swamp_positions(lines) if terrain == "swamp" else set()
    cost_fn = (lambda pos: scenario.SWAMP_COST if pos in swamps else 1)
    goal = gs.getFood().asList()[0]
    problem = searchAgents.PositionSearchProblem(gs, costFn=cost_fn, goal=goal,
                                                 warn=False, visualize=False)
    reset_tracking()
    t0 = time.perf_counter()
    actions = ALGOS_A[algo_name](problem)
    dt = (time.perf_counter() - t0) * 1000
    cost = problem.getCostOfActions(actions)
    n_open = sum(1 for line in lines for ch in line if ch != "%")
    return dict(steps=len(actions), cost=cost, expanded=problem._expanded,
                peak_frontier=peak_frontier(), time_ms=round(dt, 2), n_open=n_open,
                solved=int(cost < 999999 and len(actions) > 0))


def experiment_a():
    rows = []
    instances = []
    big = lay_mod.getLayout("bigMaze")
    instances.append(("bigMaze (goc)", "37x37", 0, "unit", big.layoutText))
    for seed in (1, 2, 3):
        for terrain in ("unit", "swamp"):
            mc_lines = scenario.make_megacity("route", seed=seed)
            instances.append(("MegaCity-Route", "31x21", seed, terrain, mc_lines))
    for (cw, ch) in SIZES_A:
        for seed in SEEDS_A:
            for terrain in ("unit", "swamp"):
                lines = scenario.make_search_maze(cw, ch, seed, swamp=(terrain == "swamp"))
                size = "%dx%d" % (len(lines[0]), len(lines))
                instances.append(("maze", size, seed, terrain, lines))
    for name, size, seed, terrain, lines in instances:
        res = {a: run_position(lines, terrain, a) for a in ALGOS_A}
        opt = res["UCS"]["cost"]
        for a, r in res.items():
            r.update(map=name, size=size, seed=seed, terrain=terrain, algo=a,
                     optimal_cost=opt,
                     opt_ratio=round(r["cost"] / opt, 4),
                     coverage=round(r["expanded"] / r["n_open"], 4),
                     penetrance=round(r["steps"] / max(1, r["expanded"]), 4))
            rows.append(r)
        print("  [A] %-14s %-8s seed=%d %-5s | " % (name, size, seed, terrain) +
              "  ".join("%s:%d/%d" % (a.split()[0], res[a]["cost"], res[a]["expanded"]) for a in ALGOS_A))
    return rows


# ---------------------------------------------------------------------------
# Thi nghiem B: Bai toan nhieu thuc an (Multi-Food) & Tien xu ly khoang cach + MST
# ---------------------------------------------------------------------------
FOOD_KS = [3, 5, 7, 9, 11]
SEEDS_B = [1, 2, 3]
MAX_EXPANDED = 300_000
MAX_SECONDS = 60.0


class BudgetExceeded(Exception):
    pass


class BudgetFoodProblem(searchAgents.FoodSearchProblem):
    """FoodSearchProblem co gioi han so nut mo rong va thoi gian."""

    def __init__(self, gs):
        super().__init__(gs)
        self._t0 = time.perf_counter()

    def getSuccessors(self, state):
        if self._expanded >= MAX_EXPANDED or time.perf_counter() - self._t0 > MAX_SECONDS:
            raise BudgetExceeded()
        return super().getSuccessors(state)


def precompute_food_bfs(gs):
    """Chay BFS 1 lan tu moi vien thuc an ban dau den tat ca cac o trong me cung.
    Tra ve dict: food_pos -> dict(cell_pos -> distance)."""
    walls = gs.getWalls()
    foods = gs.getFood().asList()
    dist_map = {}
    for f in foods:
        d = {f: 0}
        q = deque([f])
        while q:
            x, y = q.popleft()
            nd = d[(x, y)] + 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not walls[nx][ny] and (nx, ny) not in d:
                    d[(nx, ny)] = nd
                    q.append((nx, ny))
        dist_map[f] = d
    return dist_map


def make_precomputed_max_heuristic(dist_map):
    """Heuristic max_f d_maze(pos, f) su dung bang khoang cach da tinh truoc (O(k) moi nut)."""
    def h(state, problem=None):
        pos, foodGrid = state
        foods = foodGrid.asList()
        if not foods:
            return 0
        return max(dist_map[f][pos] for f in foods)
    return h


def make_precomputed_mst_heuristic(dist_map):
    """Heuristic min_f d_maze(pos, f) + MST(remaining_foods) voi cache theo frozenset(foods).
    Vua admissible, vua consistent, va chat hon max_f d_maze(pos, f)."""
    mst_cache = {}

    def mst_weight(foods_tuple):
        if len(foods_tuple) <= 1:
            return 0
        if foods_tuple in mst_cache:
            return mst_cache[foods_tuple]
        # Thuat toan Prim tren k dinh thuc an
        visited = {foods_tuple[0]}
        unvisited = set(foods_tuple[1:])
        total = 0
        while unvisited:
            best_d = 999999
            best_u = None
            for v in visited:
                dv = dist_map[v]
                for u in unvisited:
                    if dv[u] < best_d:
                        best_d = dv[u]
                        best_u = u
            visited.add(best_u)
            unvisited.remove(best_u)
            total += best_d
        mst_cache[foods_tuple] = total
        return total

    def h(state, problem=None):
        pos, foodGrid = state
        foods = tuple(foodGrid.asList())
        if not foods:
            return 0
        nearest = min(dist_map[f][pos] for f in foods)
        return nearest + mst_weight(foods)
    return h


_ANYFOOD_EXPANDED = [0]
_orig_succ = searchAgents.AnyFoodSearchProblem.getSuccessors


def _counting_succ(self, state):
    _ANYFOOD_EXPANDED[0] += 1
    return _orig_succ(self, state)


searchAgents.AnyFoodSearchProblem.getSuccessors = _counting_succ

ALGOS_B = (
    "UCS",
    "A* + foodHeuristic",
    "A* + foodHeuristic (precomputed)",
    "A* + MSTHeuristic (precomputed)",
    "WA* + MST (w=2.0)",
    "ClosestDot (greedy)",
)


def run_food(args):
    k, seed, algo = args
    lines = scenario.make_food_maze(15, 15, k, seed)
    gs = make_state(lines)
    row = dict(k=k, seed=seed, algo=algo, size="%dx%d" % (len(lines[0]), len(lines)),
               precompute_ms=0.0, search_ms=0.0, time_ms=0.0)
    reset_tracking()
    t0 = time.perf_counter()
    try:
        if algo == "ClosestDot (greedy)":
            _ANYFOOD_EXPANDED[0] = 0
            agent = searchAgents.ClosestDotSearchAgent.__new__(searchAgents.ClosestDotSearchAgent)
            with contextlib.redirect_stdout(io.StringIO()):
                agent.registerInitialState(gs)
            actions, expanded, peak = agent.actions, _ANYFOOD_EXPANDED[0], None
            cost = len(actions)
            search_ms = (time.perf_counter() - t0) * 1000
            pre_ms = 0.0
        else:
            problem = BudgetFoodProblem(gs)
            if algo == "UCS":
                pre_ms = 0.0
                ts = time.perf_counter()
                actions = search.ucs(problem)
                search_ms = (time.perf_counter() - ts) * 1000
            elif algo == "A* + foodHeuristic":
                pre_ms = 0.0
                ts = time.perf_counter()
                actions = search.astar(problem, searchAgents.foodHeuristic)
                search_ms = (time.perf_counter() - ts) * 1000
            elif algo == "A* + foodHeuristic (precomputed)":
                tp = time.perf_counter()
                dmap = precompute_food_bfs(gs)
                h_fn = make_precomputed_max_heuristic(dmap)
                pre_ms = (time.perf_counter() - tp) * 1000
                reset_tracking()
                ts = time.perf_counter()
                actions = search.astar(problem, h_fn)
                search_ms = (time.perf_counter() - ts) * 1000
            elif algo == "A* + MSTHeuristic (precomputed)":
                tp = time.perf_counter()
                dmap = precompute_food_bfs(gs)
                h_fn = make_precomputed_mst_heuristic(dmap)
                pre_ms = (time.perf_counter() - tp) * 1000
                reset_tracking()
                ts = time.perf_counter()
                actions = search.astar(problem, h_fn)
                search_ms = (time.perf_counter() - ts) * 1000
            elif algo == "WA* + MST (w=2.0)":
                tp = time.perf_counter()
                dmap = precompute_food_bfs(gs)
                h_fn = make_precomputed_mst_heuristic(dmap)
                pre_ms = (time.perf_counter() - tp) * 1000
                reset_tracking()
                ts = time.perf_counter()
                actions = weighted_astar(problem, h_fn, w=2.0)
                search_ms = (time.perf_counter() - ts) * 1000
            expanded, peak = problem._expanded, peak_frontier()
            cost = problem.getCostOfActions(actions)
        row.update(solved=1, cost=cost, expanded=expanded, peak_frontier=peak,
                   precompute_ms=round(pre_ms, 2), search_ms=round(search_ms, 2),
                   time_ms=round(pre_ms + search_ms, 2))
    except BudgetExceeded:
        tot_ms = (time.perf_counter() - t0) * 1000
        row.update(solved=0, cost=None, expanded=MAX_EXPANDED, peak_frontier=peak_frontier(),
                   precompute_ms=0.0, search_ms=round(tot_ms, 2), time_ms=round(tot_ms, 2))
    return row


def experiment_b():
    jobs = [(k, s, a) for k in FOOD_KS for s in SEEDS_B for a in ALGOS_B]
    with Pool(min(6, os.cpu_count() or 1)) as pool:
        rows = pool.map(run_food, jobs)
    # Chi phi toi uu = chi phi cua cac thuat toan toi uu (UCS / A*) neu giai duoc
    opt_algos = {"UCS", "A* + foodHeuristic", "A* + foodHeuristic (precomputed)", "A* + MSTHeuristic (precomputed)"}
    for k in FOOD_KS:
        for s in SEEDS_B:
            grp = [r for r in rows if r["k"] == k and r["seed"] == s]
            opts = [r["cost"] for r in grp if r["algo"] in opt_algos and r["solved"]]
            opt = min(opts) if opts else None
            for r in grp:
                r["optimal_cost"] = opt
                r["opt_ratio"] = round(r["cost"] / opt, 4) if (opt and r["solved"]) else None
            print("  [B] k=%-2d seed=%d | " % (k, s) + "  ".join(
                "%s:%s/%s(%.2fs)" % (r["algo"][:12], r["cost"], r["expanded"], r["time_ms"] / 1000)
                for r in grp))
    return rows


def write_csv(rows, name):
    os.makedirs(RESULTS, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("  -> %s (%d dong)" % (path, len(rows)))


if __name__ == "__main__":
    t = time.time()
    print("== Thi nghiem A: mo rong quy mo me cung ==")
    write_csv(experiment_a(), "search_scaling.csv")
    print("== Thi nghiem B: an nhieu thuc an ==")
    write_csv(experiment_b(), "search_food.csv")
    print("Xong sau %.1f giay" % (time.time() - t))
