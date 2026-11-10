"""
scenario.py - Sinh kich ban "Thanh pho me cung" (Pacman Mega-City) co kich thuoc lon.

Kich ban gom 3 lop bai toan, moi lop dung de danh gia mot nhom ky thuat da hoc:

  1. Tim duong (Project 1 - Search):
     Me cung lon (toi 101x101 o) co nhieu vong lap (braided maze) va vung "dam lay" '~'
     co chi phi di chuyen cao (5 thay vi 1). Pacman phai di tu goc duoi-phai den goc
     tren-trai. Bien the "nhieu thuc an" yeu cau an het k cham thuc an.

  2. Doi khang (Project 2 - Multi-Agent Search):
     Ban do kieu Pacman co dien lon (41x21) voi nhieu ma (ghost), nhieu vien nang luong.

  3. Hoc tang cuong (Project 3 - Reinforcement Learning):
     Gridworld lon (toi 30x30) co tuong, ho bay (-10), loi ra nho (+1) gan diem xuat phat
     va loi ra lon (+10) o xa, voi nhieu chuyen dong (noise).

Tat ca deu sinh ngau nhien co dinh hat giong (seed) de ket qua tai lap duoc.
Chay truc tiep file nay de sinh cac file .lay mau vao thu muc experiments/layouts.
"""
import os
import random
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT_DIR = os.path.join(HERE, "layouts")

SWAMP_CHAR = "~"     # Layout.py bo qua ky tu la -> o trong; ta tu doc lai de tinh chi phi
SWAMP_COST = 5


# ---------------------------------------------------------------------------
# Sinh me cung co ban
# ---------------------------------------------------------------------------
def carve_maze(cell_w, cell_h, rng):
    """Me cung hoan hao (perfect maze) bang thuat toan recursive backtracker.
    Tra ve luoi ky tu kich thuoc (2*cell_h+1) x (2*cell_w+1), '%' la tuong."""
    W, H = 2 * cell_w + 1, 2 * cell_h + 1
    g = [["%"] * W for _ in range(H)]
    start = (0, 0)
    g[1][1] = " "
    visited = {start}
    stack = [start]
    while stack:
        cx, cy = stack[-1]
        nbrs = [(cx + dx, cy + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= cx + dx < cell_w and 0 <= cy + dy < cell_h
                and (cx + dx, cy + dy) not in visited]
        if not nbrs:
            stack.pop()
            continue
        nx, ny = rng.choice(nbrs)
        g[2 * cy + 1 + (ny - cy)][2 * cx + 1 + (nx - cx)] = " "
        g[2 * ny + 1][2 * nx + 1] = " "
        visited.add((nx, ny))
        stack.append((nx, ny))
    return g


def braid(g, rng, p_deadend=0.5, p_extra=0.05):
    """Tao vong lap: pha bo mot phan ngo cut va mot so buc tuong ngau nhien.
    Vong lap lam xuat hien nhieu duong di -> DFS khong con tim duoc duong ngan nhat."""
    H, W = len(g), len(g[0])

    def is_open(r, c):
        return 0 < r < H - 1 and 0 < c < W - 1 and g[r][c] != "%"

    # 1) Pha ngo cut
    for r in range(1, H - 1, 2):
        for c in range(1, W - 1, 2):
            walls = [(dr, dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if g[r + dr][c + dc] == "%"]
            if len(walls) == 3 and rng.random() < p_deadend:
                cand = [(dr, dc) for dr, dc in walls
                        if 0 < r + 2 * dr < H - 1 and 0 < c + 2 * dc < W - 1]
                if cand:
                    dr, dc = rng.choice(cand)
                    g[r + dr][c + dc] = " "
    # 2) Pha them tuong ngau nhien giua hai o trong
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if g[r][c] != "%":
                continue
            horiz = r % 2 == 1 and c % 2 == 0 and is_open(r, c - 1) and is_open(r, c + 1)
            vert = r % 2 == 0 and c % 2 == 1 and is_open(r - 1, c) and is_open(r + 1, c)
            if (horiz or vert) and rng.random() < p_extra:
                g[r][c] = " "
    return g


def open_cells(g):
    return [(r, c) for r in range(len(g)) for c in range(len(g[0])) if g[r][c] != "%"]


def to_text(g):
    return ["".join(row) for row in g]


def save_layout(lines, name):
    os.makedirs(LAYOUT_DIR, exist_ok=True)
    path = os.path.join(LAYOUT_DIR, name + ".lay")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# 1) Kich ban tim duong
# ---------------------------------------------------------------------------
def add_swamps(g, rng, target_frac=0.25, protect=()):
    """Rai cac vung dam lay hinh dom (blob) cho den khi phu ~target_frac so o trong."""
    cells = open_cells(g)
    target = int(target_frac * len(cells))
    protect = set(protect)
    marked = 0
    H, W = len(g), len(g[0])
    while marked < target:
        r0, c0 = rng.choice(cells)
        radius = rng.randint(2, max(3, min(H, W) // 12))
        for r in range(max(1, r0 - radius), min(H - 1, r0 + radius + 1)):
            for c in range(max(1, c0 - radius), min(W - 1, c0 + radius + 1)):
                if (abs(r - r0) + abs(c - c0) <= radius and g[r][c] == " "
                        and (r, c) not in protect and rng.random() < 0.85):
                    g[r][c] = SWAMP_CHAR
                    marked += 1
                    if marked >= target:
                        return g
    return g


def make_search_maze(cell_w, cell_h, seed, swamp=False, p_deadend=0.5, p_extra=0.06):
    """Me cung lon 1 dich: P o goc duoi-phai, '.' o goc tren-trai."""
    rng = random.Random(seed)
    g = braid(carve_maze(cell_w, cell_h, rng), rng, p_deadend, p_extra)
    H, W = len(g), len(g[0])
    start, goal = (H - 2, W - 2), (1, 1)
    if swamp:
        add_swamps(g, rng, 0.25, protect=(start, goal))
    g[start[0]][start[1]] = "P"
    g[goal[0]][goal[1]] = "."
    return to_text(g)


def make_food_maze(cell_w, cell_h, n_food, seed):
    """Me cung vua phai voi k cham thuc an (bai toan FoodSearchProblem)."""
    rng = random.Random(seed)
    g = braid(carve_maze(cell_w, cell_h, rng), rng, p_deadend=0.6, p_extra=0.08)
    cells = open_cells(g)
    rng.shuffle(cells)
    pr, pc = cells[0]
    g[pr][pc] = "P"
    for r, c in cells[1:1 + n_food]:
        g[r][c] = "."
    return to_text(g)


def swamp_positions(lines):
    """Tap toa do (x, y) theo quy uoc Pacman (y=0 o duoi) cua cac o dam lay."""
    H = len(lines)
    return {(x, H - 1 - row) for row, line in enumerate(lines)
            for x, ch in enumerate(line) if ch == SWAMP_CHAR}


# ---------------------------------------------------------------------------
# 2) Kich ban doi khang
# ---------------------------------------------------------------------------
def make_classic_arena(cell_w, cell_h, n_ghosts, seed, food_frac=0.45, n_capsules=4):
    """Ban do Pacman lon kieu co dien: it ngo cut, thuc an rai rac, nhieu ma o giua."""
    rng = random.Random(seed)
    g = braid(carve_maze(cell_w, cell_h, rng), rng, p_deadend=1.0, p_extra=0.18)
    H, W = len(g), len(g[0])
    cells = open_cells(g)
    # Pacman: o trong gan giua canh duoi
    pac = min(cells, key=lambda rc: (abs(rc[0] - (H - 2)) * 3 + abs(rc[1] - W // 2)))
    # Ma: cac o gan tam ban do
    center = sorted(cells, key=lambda rc: abs(rc[0] - H // 2) + abs(rc[1] - W // 2))
    ghosts = [rc for rc in center if rc != pac][:n_ghosts]
    # Vien nang luong: gan 4 goc
    corners = [(1, 1), (1, W - 2), (H - 2, 1), (H - 2, W - 2)]
    caps = []
    for cr, cc in corners[:n_capsules]:
        caps.append(min(cells, key=lambda rc: abs(rc[0] - cr) + abs(rc[1] - cc)))
    taken = {pac, *ghosts, *caps}
    for r, c in cells:
        if (r, c) not in taken and rng.random() < food_frac:
            g[r][c] = "."
    for r, c in caps:
        g[r][c] = "o"
    for r, c in ghosts:
        g[r][c] = "G"
    g[pac[0]][pac[1]] = "P"
    return to_text(g)


# ---------------------------------------------------------------------------
# 3) Kich ban hoc tang cuong (Gridworld)
# ---------------------------------------------------------------------------
def make_gridworld(width, height, seed, wall_frac=0.18, pit_frac=0.03):
    """Gridworld lon. Tra ve danh sach hang (hang dau la hang tren cung) cho gridworld.makeGrid.
    'S' = xuat phat (goc duoi-trai), +10 = loi ra lon (goc tren-phai),
    +1 = loi ra nho gan xuat phat, -10 = ho bay. Dam bao moi o trong deu den duoc tu S."""
    for attempt in range(100):
        rng = random.Random(seed * 1000 + attempt)
        g = [[" "] * width for _ in range(height)]
        for r in range(height):
            for c in range(width):
                if rng.random() < wall_frac:
                    g[r][c] = "#"
        s = (height - 1, 0)
        big = (0, width - 1)
        small = (height - 1 - max(1, height // 5), max(1, width // 5))
        for (r, c) in (s, big, small):
            g[r][c] = " "
        # Ho bay
        free = [(r, c) for r in range(height) for c in range(width)
                if g[r][c] == " " and (r, c) not in (s, big, small)]
        rng.shuffle(free)
        for r, c in free[:max(1, int(pit_frac * width * height))]:
            g[r][c] = -10
        g[s[0]][s[1]] = "S"
        g[big[0]][big[1]] = 10
        g[small[0]][small[1]] = 1

        # Kiem tra lien thong (o thoat/ho bay la diem cuoi, khong di xuyen qua)
        seen = {s}
        q = deque([s])
        while q:
            r, c = q.popleft()
            if isinstance(g[r][c], int):
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and g[nr][nc] != "#" and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        if big in seen and small in seen:
            for r in range(height):
                for c in range(width):
                    if g[r][c] != "#" and (r, c) not in seen:
                        g[r][c] = "#"
            return g
    raise RuntimeError("Khong sinh duoc gridworld lien thong")


def gridworld_to_text(g):
    sym = {10: "E", 1: "e", -10: "x"}
    return ["".join(sym.get(v, v) if isinstance(v, int) else v for v in row) for row in g]


if __name__ == "__main__":
    samples = {
        "maze_61x61": make_search_maze(30, 30, seed=1),
        "maze_101x101_swamp": make_search_maze(50, 50, seed=1, swamp=True),
        "food_31x31_k8": make_food_maze(15, 15, 8, seed=1),
        "arena_41x21_3ghosts": make_classic_arena(20, 10, 3, seed=1),
    }
    for name, lines in samples.items():
        path = save_layout(lines, name)
        print("%-24s %3dx%-3d -> %s" % (name, len(lines[0]), len(lines), path))
    print("\nVi du arena_41x21_3ghosts:")
    print("\n".join(samples["arena_41x21_3ghosts"]))
    print("\nVi du gridworld 20x20 (E=+10, e=+1, x=-10, #=tuong):")
    print("\n".join(gridworld_to_text(make_gridworld(20, 20, seed=1))))
