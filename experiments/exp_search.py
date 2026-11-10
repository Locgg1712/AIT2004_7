"""
exp_search.py - Danh gia cac thuat toan tim kiem (Project 1) tren kich ban me cung lon.

Thi nghiem A (mo rong quy mo): DFS, BFS, UCS, A*-Manhattan, A*-Euclid tren me cung
    31x31 -> 151x151, voi 2 che do chi phi: dong nhat (unit) va co dam lay (swamp, chi phi 5).
Thi nghiem B (nhieu thuc an): UCS, A* + foodHeuristic, ClosestDot (tham lam) tren
    me cung 31x31 voi k = 3..11 cham thuc an, co gioi han ngan sach tinh toan.

Do do (metrics):
    cost            tong chi phi duong di tim duoc
    opt_ratio       cost / chi phi toi uu (=1.0 la toi uu)
    expanded        so nut duoc mo rong (goi getSuccessors)
    coverage        expanded / so o trong (ty le khong gian bi kham pha)
    penetrance      so buoc cua loi giai / expanded (cang gan 1 cang "tap trung")
    peak_frontier   kich thuoc lon nhat cua frontier (uoc luong bo nho)
    time_ms         thoi gian chay (ms)
    solved          tim duoc loi giai trong ngan sach hay khong

Chay:  python exp_search.py            (ket qua ghi vao experiments/results/)
"""
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

import layout as lay_mod      # noqa: E402  (module cua Project 1)
import pacman                 # noqa: E402
import search                 # noqa: E402
import searchAgents           # noqa: E402
import util                   # noqa: E402
import scenario               # noqa: E402

# ---------------------------------------------------------------------------
# Do kich thuoc frontier lon nhat ma KHONG sua search.py: boc cac cau truc du lieu
# util.Stack / Queue / PriorityQueue. Frontier ngoai cung la container tao dau tien.
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


# ---------------------------------------------------------------------------
# Thi nghiem A
# ---------------------------------------------------------------------------
ALGOS_A = {
    "DFS": lambda p: search.dfs(p),
    "BFS": lambda p: search.bfs(p),
    "UCS": lambda p: search.ucs(p),
    "A*-Manhattan": lambda p: search.astar(p, searchAgents.manhattanHeuristic),
    "A*-Euclid": lambda p: search.astar(p, searchAgents.euclideanHeuristic),
}
SIZES_A = [(15, 15), (30, 30), (50, 50), (75, 75)]   # so "o" me cung -> luoi (2n+1)x(2n+1)
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
    # Ban do goc cua do an lam moc so sanh
    big = lay_mod.getLayout("bigMaze")
    instances.append(("bigMaze (goc)", "37x37", 0, "unit", big.layoutText))
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
              "  ".join("%s:%d/%d" % (a, res[a]["cost"], res[a]["expanded"]) for a in ALGOS_A))
    return rows


# ---------------------------------------------------------------------------
# Thi nghiem B
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


_ANYFOOD_EXPANDED = [0]
_orig_succ = searchAgents.AnyFoodSearchProblem.getSuccessors


def _counting_succ(self, state):
    _ANYFOOD_EXPANDED[0] += 1
    return _orig_succ(self, state)


searchAgents.AnyFoodSearchProblem.getSuccessors = _counting_succ


def run_food(args):
    k, seed, algo = args
    lines = scenario.make_food_maze(15, 15, k, seed)
    gs = make_state(lines)
    row = dict(k=k, seed=seed, algo=algo, size="%dx%d" % (len(lines[0]), len(lines)))
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
        else:
            problem = BudgetFoodProblem(gs)
            if algo == "UCS":
                actions = search.ucs(problem)
            else:
                actions = search.astar(problem, searchAgents.foodHeuristic)
            expanded, peak = problem._expanded, peak_frontier()
            cost = problem.getCostOfActions(actions)
        row.update(solved=1, cost=cost, expanded=expanded, peak_frontier=peak)
    except BudgetExceeded:
        row.update(solved=0, cost=None, expanded=MAX_EXPANDED, peak_frontier=peak_frontier())
    row["time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return row


def experiment_b():
    jobs = [(k, s, a) for k in FOOD_KS for s in SEEDS_B
            for a in ("UCS", "A* + foodHeuristic", "ClosestDot (greedy)")]
    with Pool(min(6, os.cpu_count() or 1)) as pool:
        rows = pool.map(run_food, jobs)
    # chi phi toi uu = chi phi cua thuat toan toi uu (UCS / A*) neu giai duoc
    for k in FOOD_KS:
        for s in SEEDS_B:
            grp = [r for r in rows if r["k"] == k and r["seed"] == s]
            opts = [r["cost"] for r in grp if r["algo"] != "ClosestDot (greedy)" and r["solved"]]
            opt = min(opts) if opts else None
            for r in grp:
                r["optimal_cost"] = opt
                r["opt_ratio"] = round(r["cost"] / opt, 4) if (opt and r["solved"]) else None
            print("  [B] k=%-2d seed=%d | " % (k, s) + "  ".join(
                "%s:%s/%s(%.1fs)" % (r["algo"].split()[0], r["cost"], r["expanded"], r["time_ms"] / 1000)
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
