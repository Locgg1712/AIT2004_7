# CS188 AI Projects (1, 2, 3, 5) & Thực Nghiệm Mở Rộng Trên Ma Trận Cỡ Lớn

Tài liệu `README.md` này trình bày toàn diện **cách giải quyết và hiện thực chi tiết từng thuật toán** trong 4 bài tập lớn của môn Trí tuệ Nhân tạo (UC Berkeley CS188: **Search**, **Multi-Agent Search**, **Reinforcement Learning**, **Machine Learning**), đồng thời giải thích chi tiết **cách thiết kế kịch bản ma trận cỡ lớn, hệ thống độ đo (evaluation metrics) và kết quả thực nghiệm đánh giá hiệu năng**.

---

## Mục Lục

1. [Cấu trúc thư mục & Điểm số Autograder](#1-cấu-trúc-thư-mục--điểm-số-autograder)
2. [Project 1: Search — Giải thích chi tiết Thuật toán & Heuristic (`search/`)](#2-project-1-search--giải-thích-chi-tiết-thuật-toán--heuristic-search)
3. [Project 2: Multi-Agent Search — Tìm kiếm Đối kháng & Hàm Lượng giá (`multiagent/`)](#3-project-2-multi-agent-search--tìm-kiếm-đối-kháng--hàm-lượng-giá-multiagent)
4. [Project 3: Reinforcement Learning — Quy hoạch động & Học tăng cường (`reinforcement/`)](#4-project-3-reinforcement-learning--quy-hoạch-động--học-tăng-cường-reinforcement)
5. [Project 5: Machine Learning — Mạng Nơ-ron, RNN, Tích chập & Self-Attention (`machinelearning/`)](#5-project-5-machine-learning--mạng-nơ-ron-rnn-tích-chập--self-attention-machinelearning)
6. [Xây dựng Kịch bản Ma trận lớn & Cơ chế Đo lường (`experiments/`)](#6-xây-dựng-kịch-bản-ma-trận-lớn--cơ-chế-đo-lường-experiments)
7. [Độ đo Đánh giá (Metrics) & Tổng hợp Kết quả Thực nghiệm](#7-độ-đo-đánh-giá-metrics--tổng-hợp-kết-quả-thực-nghiệm)
8. [Hướng dẫn chạy Kiểm thử & Tái lập Thực nghiệm](#8-hướng-dẫn-chạy-kiểm-thử--tái-lập-thực-nghiệm)

---

## 1. Cấu trúc thư mục & Điểm số Autograder

Toàn bộ dự án gồm 4 bài tập cốt lõi và 1 bộ khung thực nghiệm mở rộng (`experiments/`):

```text
d:\AIT2004_7\
├── README.md                         # Tài liệu giải thích tổng thể & hướng dẫn chi tiết (file này)
├── .gitignore                        # Whitelist chỉ cho phép git theo dõi các file bài nộp chuẩn
├── search/                           # Project 1: Search (Điểm Autograder: 26/25)
│   ├── search.py                     # DFS, BFS, UCS, A* (Graph Search)
│   └── searchAgents.py               # CornersProblem, cornersHeuristic, foodHeuristic, ClosestDot
├── multiagent/                       # Project 2: Multi-Agent Search (Điểm Autograder: 25/25)
│   └── multiAgents.py                # Reflex, Minimax, AlphaBeta, Expectimax, betterEvaluationFunction
├── reinforcement/                    # Project 3: Reinforcement Learning (Điểm Autograder: 26/25)
│   ├── valueIterationAgents.py       # Value Iteration đồng bộ & Prioritized Sweeping
│   ├── qlearningAgents.py            # Tabular Q-Learning, Epsilon-Greedy, Approximate Q-Learning
│   └── analysis.py                   # Thiết kế tham số MDP (Discount, Noise, LivingReward)
├── machinelearning/                  # Project 5: Machine Learning (Điểm Autograder: 25/25)
│   ├── models.py                     # Perceptron, MLP, RNN, Vectorized Convolve 2D, Causal Attention
│   ├── losses.py                     # MSE Loss & Cross-Entropy Loss
│   └── train.py                      # Vòng lặp huấn luyện tối ưu hóa
└── experiments/                      # Kịch bản mở rộng ma trận lớn & đánh giá định lượng
    ├── scenario.py                   # Sinh mê cung Braided 151x151, Đầm lầy, Arena 41x21, MDP 30x30
    ├── exp_search.py                 # Thực nghiệm A & B (Mở rộng mê cung & Bùng nổ tổ hợp Multi-Food)
    ├── exp_multiagent.py             # Thực nghiệm C & D (Cắt tỉa Alpha-Beta & Đấu trường 3 Ghost)
    ├── exp_rl.py                     # Thực nghiệm E & F (Tốc độ hội tụ Bellman & Khái quát hóa Q-Learning)
    ├── exp_ml.py                     # Thực nghiệm G & H (Vector hóa Convolve & Bậc hai Self-Attention)
    ├── summarize.py                  # Script tổng hợp kết quả từ CSV
    ├── REPORT.md                     # Báo cáo phân tích chi tiết số liệu thực nghiệm
    ├── EXPLANATION_GUIDE.md          # Bản hướng dẫn chuyên sâu về code & thực nghiệm
    ├── layouts/                      # Các file bản đồ mê cung lớn (.lay) đã sinh
    └── results/                      # Các file dữ liệu kết quả thực nghiệm (.csv)
```

---

## 2. Project 1: Search — Giải thích chi tiết Thuật toán & Heuristic (`search/`)

### 2.1. Khung tìm kiếm đồ thị tổng quát (`search/search.py`)

Cả 4 thuật toán trong [`search/search.py`](file:///d:/AIT2004_7/search/search.py) đều được cài đặt theo mô hình **Graph Search** (lưu tập trạng thái đã mở rộng để không bị lặp vô hạn trên đồ thị có chu trình). Điểm khác biệt cốt lõi nằm ở **cấu trúc dữ liệu của hàng đợi biên (`frontier`)** và **cách quản lý chi phí**:

| Thuật toán | Hàm trong `search.py` | Cấu trúc `frontier` | Thứ tự ưu tiên lấy nút ra | Đảm bảo tối ưu? |
| :--- | :--- | :--- | :--- | :--- |
| **DFS** | [`depthFirstSearch`](file:///d:/AIT2004_7/search/search.py#L75-L107) | `util.Stack()` (LIFO) | Nút mới nhất (sâu nhất) | **Không** |
| **BFS** | [`breadthFirstSearch`](file:///d:/AIT2004_7/search/search.py#L109-L129) | `util.Queue()` (FIFO) | Nút nông nhất (ít bước đi nhất) | Chỉ khi chi phí mọi bước bằng nhau ($c=1$) |
| **UCS** | [`uniformCostSearch`](file:///d:/AIT2004_7/search/search.py#L131-L156) | `util.PriorityQueue()` | Chi phí tích lũy nhỏ nhất $g(n)$ | **Có** (với mọi chi phí dương $c > 0$) |
| **A\*** | [`aStarSearch`](file:///d:/AIT2004_7/search/search.py#L165-L192) | `util.PriorityQueue()` | Tổng chi phí ước lượng $f(n) = g(n) + h(n)$ | **Có** (khi $h(n)$ admissible & consistent) |

#### Các chi tiết kỹ thuật quan trọng khi cài đặt:
1. **Kiểm tra `visited` lúc `pop()` thay vì lúc `push()`:**
   Trong `depthFirstSearch` và `breadthFirstSearch`, ta chỉ kiểm tra `if state in visited: continue` và `visited.add(state)` **ngay sau khi lấy nút ra khỏi `frontier`** (trước khi gọi `problem.getSuccessors(state)`). Điều này tuân thủ đúng định nghĩa chuẩn của CS188: một nút chỉ được coi là *đã đóng (closed/expanded)* khi nó thực sự được lấy ra để sinh các nút con, giúp số nút mở rộng (`_expanded`) khớp tuyệt đối với autograder.
2. **Cơ chế chống mở rộng lặp với chi phí cao hơn trong `UCS` và `A*`:**
   Một trạng thái `s` có thể được đẩy vào `PriorityQueue` nhiều lần từ các đường đi khác nhau với chi phí $g_1 > g_2$. Trong [`uniformCostSearch`](file:///d:/AIT2004_7/search/search.py#L135-L154) và [`aStarSearch`](file:///d:/AIT2004_7/search/search.py#L170-L190), ta sử dụng hai từ điển `best_cost` và `expanded_cost`:
   - Khi lấy `(state, actions, cost)` từ `PriorityQueue`: nếu `state` đã từng được mở rộng với chi phí nhỏ hơn hoặc bằng (`cost >= expanded_cost[state]`), ta bỏ qua ngay.
   - Khi duyệt nút con `successor` với `new_cost = cost + step_cost`: chỉ đẩy vào `frontier` nếu `successor` chưa có trong `best_cost` hoặc `new_cost < best_cost[successor]`.

---

### 2.2. Biểu diễn trạng thái & Thiết kế Heuristic (`search/searchAgents.py`)

#### 1. Bài toán 4 góc — [`CornersProblem`](file:///d:/AIT2004_7/search/searchAgents.py#L274-L359) (Q5)
- **Cách chọn không gian trạng thái:** Ta chỉ cần biết vị trí hiện tại của Pacman $(x, y)$ và tập hợp các góc nào trong 4 góc đã được đi qua. Do đó, mỗi trạng thái được biểu diễn bằng một tuple bất biến:
  $$\text{state} = \big((x, y),\; \text{frozenset}(\text{visited\_corners})\big)$$
- **Lý do dùng `frozenset`:** Các thuật toán tìm kiếm cần lưu `state` vào tập `visited = set()` hoặc làm khóa của `dict`. Kiểu `set` thông thường là mutable nên không băm được (`TypeError: unhashable type: 'set'`), trong khi `frozenset` vừa bất biến, vừa băm được, vừa hỗ trợ toán tử hợp `visited | frozenset((next_position,))` rất gọn nhẹ.
- Kích thước không gian trạng thái giảm xuống chỉ còn tối đa $16 \times |\text{số ô trống}|$.

#### 2. Heuristic cho bài toán 4 góc — [`cornersHeuristic`](file:///d:/AIT2004_7/search/searchAgents.py#L361-L398) (Q6)
- **Ý tưởng:** Với tập các góc chưa đi qua `remaining` (tối đa $4$ góc $\implies$ tối đa $4! = 24$ thứ tự đi), ta duyệt tất cả các hoán vị $\pi \in \text{permutations}(\text{remaining})$ và tính tổng khoảng cách mê cung thực tế (`mazeDistance`) nối từ vị trí hiện tại của Pacman lần lượt qua các góc theo thứ tự $\pi$:
  $$h(s) = \min_{\pi \in \text{Perm}(\text{remaining})} \left[ d_{\text{maze}}(\text{pos}, \pi_1) + \sum_{i=1}^{|\text{remaining}|-1} d_{\text{maze}}(\pi_i, \pi_{i+1}) \right]$$
- **Kỹ thuật Cache (`problem.heuristicInfo`):** Vì khoảng cách mê cung giữa 2 tọa độ $(p_1, p_2)$ không đổi trong suốt ván chơi, ta lưu `problem.heuristicInfo[tuple(sorted((point1, point2)))] = mazeDistance(...)`. Nhờ vậy, BFS tính khoảng cách chỉ chạy vài lần ở các nút đầu tiên, sau đó mọi phép tra cứu đều là $O(1)$.
- **Tính chất:** Đây là cận dưới chính xác tuyệt đối ($h(s) = h^*(s)$), vừa **admissible** vừa **consistent**.

#### 3. Heuristic ăn toàn bộ thức ăn — [`foodHeuristic`](file:///d:/AIT2004_7/search/searchAgents.py#L462-L504) (Q7)
- **Thách thức:** Khi số viên thức ăn $k$ lớn (ví dụ $k=15 \dots 30$), việc duyệt $k!$ hoán vị là bất khả thi.
- **Công thức thiết kế:** Ta lấy **khoảng cách mê cung thật lớn nhất từ vị trí hiện tại của Pacman tới viên thức ăn xa nhất còn lại**:
  $$h(\text{pos}, \text{FoodGrid}) = \max_{f \in \text{FoodGrid}} d_{\text{maze}}(\text{pos}, f)$$
  kèm theo lưu cache từng cặp `tuple(sorted((position, food_position)))` trong `problem.heuristicInfo`.
- **Tại sao vừa Admissible vừa Consistent?**
  - *Admissible:* Để ăn hết mọi viên thức ăn còn lại, Pacman bắt buộc phải đi tới viên thức ăn xa nhất $f^*$. Chi phí đi từ vị trí hiện tại tới riêng viên $f^*$ đã là $d_{\text{maze}}(\text{pos}, f^*)$, nên tổng hành trình ăn hết tất cả các viên không bao giờ nhỏ hơn giá trị này.
  - *Consistent:* Khi Pacman bước 1 bước từ $u$ sang $v$ (chi phí bước $= 1$), khoảng cách mê cung tới bất kỳ viên thức ăn $f$ nào giảm tối đa $1$ đơn vị: $d_{\text{maze}}(u, f) \le 1 + d_{\text{maze}}(v, f)$. Do đó $\max_f d_{\text{maze}}(u, f) \le 1 + \max_f d_{\text{maze}}(v, f)$.

#### 4. Tìm kiếm tham lam viên gần nhất — [`ClosestDotSearchAgent`](file:///d:/AIT2004_7/search/searchAgents.py#L506-L570) (Q8)
- Cài đặt [`AnyFoodSearchProblem.isGoalState`](file:///d:/AIT2004_7/search/searchAgents.py#L562-L569) kiểm tra xem ô hiện tại `(x, y)` có thức ăn hay không (`return self.food[x][y]`).
- Hàm [`findPathToClosestDot`](file:///d:/AIT2004_7/search/searchAgents.py#L523-L534) gọi `search.breadthFirstSearch(AnyFoodSearchProblem(gameState))` để luôn tìm đường ngắn nhất tới viên thức ăn gần nhất ở mỗi chặng.

---

## 3. Project 2: Multi-Agent Search — Tìm kiếm Đối kháng & Hàm Lượng giá (`multiagent/`)

### 3.1. Quy ước Độ sâu (Depth / Ply) trong Cây Trò chơi Đa tác tử

Trong [`multiagent/multiAgents.py`](file:///d:/AIT2004_7/multiagent/multiAgents.py), bàn cờ có `numAgents = gameState.getNumAgents()` tác tử:
- `agentIndex == 0`: **Pacman** (tác tử `MAX`).
- `agentIndex` từ `1` đến `numAgents - 1`: **Các Ghost** (tác tử `MIN` hoặc `CHANCE`).
- Một đơn vị độ sâu (`depth`) được định nghĩa là **1 ply đầy đủ** (Pacman đi 1 bước và tất cả các Ghost lần lượt đi 1 bước). Do đó, cách chuyển lượt đệ quy chuẩn trong cả 3 thuật toán là:
  ```python
  nextAgent = (agentIndex + 1) % numAgents
  nextDepth = depth + 1 if nextAgent == 0 else depth
  ```
  Đệ quy dừng khi `state.isWin() or state.isLose() or depth == self.depth`, trả về giá trị từ `self.evaluationFunction(state)`.

### 3.2. Giải thích chi tiết từng Tác tử và Hàm Lượng giá

1. **[`ReflexAgent.evaluationFunction`](file:///d:/AIT2004_7/multiagent/multiAgents.py#L55-L103) (Q1):**
   - Đánh giá trạng thái kế tiếp `successorGameState = currentGameState.generatePacmanSuccessor(action)`.
   - Cộng điểm thưởng tỷ lệ nghịch với khoảng cách Manhattan tới viên thức ăn gần nhất: `score += 10.0 / (minFoodDist + 1.0)`.
   - Duyệt qua các Ghost:
     - Nếu Ghost đang hoảng sợ (`ghostState.scaredTimer > 0`): thưởng điểm săn đuổi `score += 20.0 / (dist + 1.0)`.
     - Nếu Ghost đang hoạt động bình thường: nếu `dist <= 1` trả về ngay `float('-inf')` để cấm đi vào ô chết; nếu ở xa hơn thì phạt nhẹ `score -= 2.0 / (dist + 1.0)`.
   - Phạt hành động đứng yên `if action == Directions.STOP: score -= 10.0` để Pacman luôn chủ động di chuyển.

2. **[`MinimaxAgent`](file:///d:/AIT2004_7/multiagent/multiAgents.py#L135-L206) (Q2):**
   - Tại nút `agentIndex == 0` (Pacman), lấy $\max$ qua các hành động hợp lệ.
   - Tại nút `agentIndex >= 1` (Ghost), lấy $\min$ qua các hành động hợp lệ (giả định mọi Ghost đều chơi tối ưu để ép điểm Pacman thấp nhất).
   - Ở hàm gốc `getAction`, sử dụng điều kiện `if score > bestScore or bestAction is None:` để đảm bảo ngay cả khi mọi hướng đi đều dẫn đến thua bắt buộc (`score == -inf`), Pacman vẫn chọn một hành động hợp lệ đầu tiên thay vì trả về `None`.

3. **[`AlphaBetaAgent`](file:///d:/AIT2004_7/multiagent/multiAgents.py#L208-L265) (Q3):**
   - Truyền thêm hai tham số cận $(\alpha, \beta)$ vào hàm đệ quy `alphabeta(state, agentIndex, depth, alpha, beta)`:
     - Tại nút **MAX** (`agentIndex == 0`): cập nhật `v = max(v, alphabeta(...))`. Nếu **`v > beta`**, cắt tỉa ngay (`return v`); ngược lại cập nhật `alpha = max(alpha, v)`.
     - Tại nút **MIN** (`agentIndex >= 1`): cập nhật `v = min(v, alphabeta(...))`. Nếu **`v < alpha`**, cắt tỉa ngay (`return v`); ngược lại cập nhật `beta = min(beta, v)`.
   - **Lưu ý quan trọng:** Phải dùng bất đẳng thức ngặt `v > beta` và `v < alpha` (không cắt khi `v == beta` hay `v == alpha`) theo đúng yêu cầu của CS188 để không bỏ sót các nút hòa điểm khi kiểm tra thứ tự mở rộng cây.

4. **[`ExpectimaxAgent`](file:///d:/AIT2004_7/multiagent/multiAgents.py#L267-L320) (Q4):**
   - Thay vì giả định Ghost luôn chọn nước đi tệ nhất cho Pacman (`min`), nút Ghost (`agentIndex >= 1`) tính **trung bình cộng (kỳ vọng toán học)** trên tất cả các hành động hợp lệ của Ghost:
     ```python
     totalValue = 0.0
     for action in legalActions:
         successor = state.generateSuccessor(agentIndex, action)
         totalValue += expectimax(successor, nextAgent, nextDepth)
     return totalValue / len(legalActions)
     ```

5. **Hàm lượng giá nâng cao — [`betterEvaluationFunction`](file:///d:/AIT2004_7/multiagent/multiAgents.py#L322-L370) (Q5):**
   - Được thiết kế như một tổ hợp tuyến tính các đặc trưng phi tuyến của trạng thái `currentGameState`:
     $$\text{Eval}(s) = \text{Score}(s) + \frac{10}{d_{\text{minFood}} + 1} - 4 N_{\text{food}} + \frac{5}{d_{\text{minCap}} + 1} - 20 N_{\text{cap}} + \sum_{g \in \text{Ghosts}} \phi(g)$$
   - **Ý nghĩa của việc kết hợp lực hút khoảng cách và phạt số lượng còn lại:**
     - Thành phần $\frac{10}{d_{\text{minFood}} + 1}$ đóng vai trò "la bàn" hướng Pacman về phía viên thức ăn gần nhất khi ở xa.
     - Thành phần $-4 N_{\text{food}}$ và $-20 N_{\text{cap}}$ đảm bảo mỗi khi Pacman thực sự ăn 1 viên thức ăn hoặc 1 viên năng lượng (Capsule), giá trị hàm mục tiêu tăng vọt một lượng cố định ($+4$ hoặc $+20$ cộng với điểm nội tại của game), triệt tiêu hoàn toàn hiện tượng Pacman đứng ngập ngừng cạnh viên thức ăn cuối cùng của một cụm.
     - Khi ăn được Capsule (`ghostState.scaredTimer > 0`), hệ số thưởng săn Ghost tăng mạnh lên $+50.0 / (dist + 1.0)$ để Pacman chủ động truy đuổi Ghost kiếm $+200$ điểm thưởng.

---

## 4. Project 3: Reinforcement Learning — Quy hoạch động & Học tăng cường (`reinforcement/`)

### 4.1. Quy hoạch động trên MDP (`reinforcement/valueIterationAgents.py`)

1. **[`ValueIterationAgent`](file:///d:/AIT2004_7/reinforcement/valueIterationAgents.py#L34-L120) (Q1):**
   - **Tính giá trị $Q(s, a)$ ([`computeQValueFromValues`](file:///d:/AIT2004_7/reinforcement/valueIterationAgents.py#L82-L92)):**
     $$Q_k(s, a) = \sum_{s'} T(s, a, s') \big[ R(s, a, s') + \gamma V_k(s') \big]$$
   - **Trích xuất hành động tối ưu ([`computeActionFromValues`](file:///d:/AIT2004_7/reinforcement/valueIterationAgents.py#L94-L110)):**
     $$\pi_{k}(s) = \arg\max_{a \in A(s)} Q_k(s, a)$$
     (trả về `None` nếu `mdp.isTerminal(state)` hoặc không có hành động hợp lệ).
   - **Cập nhật đồng bộ theo lô ([`runValueIteration`](file:///d:/AIT2004_7/reinforcement/valueIterationAgents.py#L62-L73)):** Ở mỗi vòng lặp, khởi tạo một bảng `newValues = util.Counter()` mới hoàn toàn, tính $V_{k+1}(s) = \max_a Q_k(s,a)$ dựa trên `self.values` cũ, rồi cuối vòng lặp mới gán `self.values = newValues`.

2. **[`PrioritizedSweepingValueIterationAgent`](file:///d:/AIT2004_7/reinforcement/valueIterationAgents.py#L122-L174) (Q4):**
   - Thay vì quét mù quáng toàn bộ không gian trạng thái $S$ ở mỗi vòng lặp, thuật toán tập trung cập nhật các trạng thái có **sai số Bellman lớn nhất**:
     1. **Lập danh sách tiền nhiệm (`predecessors`):** Tìm mọi trạng thái $p$ có thể đi tới $s$ với xác suất $T(p, a, s) > 0$.
     2. **Đưa vào hàng đợi ưu tiên (`pq`):** Tính $\text{diff}(s) = |V(s) - \max_a Q(s, a)|$ cho mọi trạng thái không kết thúc và gọi `pq.update(state, -diff)` (dùng `-diff` vì `PriorityQueue` là min-heap).
     3. **Vòng lặp cập nhật có chọn lọc:** Trong `self.iterations` bước, lấy trạng thái `state` ở đỉnh `pq`, cập nhật `self.values[state]`, sau đó chỉ tính lại `diff` cho các trạng thái tiền nhiệm `p in predecessors[state]`. Nếu `diff > self.theta` thì mới đẩy `p` vào `pq`.

### 4.2. Tinh chỉnh tham số MDP (`reinforcement/analysis.py`)

- **[`question2`](file:///d:/AIT2004_7/reinforcement/analysis.py#L22-L25):** Đặt `answerDiscount = 0.9, answerNoise = 0.0` để loại bỏ rủi ro trượt xuống vực phạt $-100$ trên cầu `BridgeGrid`.
- **[`question3a` – `question3e`](file:///d:/AIT2004_7/reinforcement/analysis.py#L27-L60):** Sử dụng hai tham số điều khiển trực giao trên `DiscountGrid`:
  - **Chọn đích gần ($+1$) hay đích xa ($+10$):** Điều khiển bằng hệ số chiết khấu $\gamma$ (`0.2` cho đích gần ở `3a, 3b`; `0.9` cho đích xa ở `3c, 3d`).
  - **Chọn đường ngắn sát vực hay đường vòng an toàn:** Điều khiển bằng độ nhiễu `noise` (`0.0` để đi sát vực ở `3a, 3c`; `0.2` để đi đường vòng ở `3b, 3d`).
  - **Tránh mọi ô kết thúc (`3e`):** Đặt `answerDiscount = 1.0, answerNoise = 0.0, answerLivingReward = 10.0` để phần thưởng sống sót mỗi bước lớn hơn việc kết thúc.
- **[`question7`](file:///d:/AIT2004_7/reinforcement/analysis.py#L62-L63):** Trả về `'NOT POSSIBLE'` vì 50 tập huấn luyện là quá ít để Q-Learning vừa khám phá hết cây cầu dài vừa hội tụ chính sách tối ưu với xác suất $>99\%$.

### 4.3. Học tăng cường không mô hình (`reinforcement/qlearningAgents.py`)

1. **[`QLearningAgent`](file:///d:/AIT2004_7/reinforcement/qlearningAgents.py#L21-L127) (Q5, Q6):**
   - Lưu giá trị $Q(s,a)$ trong bảng băm `self.qValues = util.Counter()`.
   - Phá vỡ thế hòa ngẫu nhiên trong `computeActionFromQValues` bằng `random.choice(bestActions)`.
   - Chọn hành động khám phá $\varepsilon$-greedy trong `getAction` bằng `util.flipCoin(self.epsilon)`.
   - Cập nhật mẫu chuyển dịch $(s, a, s', r)$ trong `update`:
     $$Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha \big[ r + \gamma \max_{a'} Q(s', a') \big]$$

2. **[`ApproximateQAgent`](file:///d:/AIT2004_7/reinforcement/qlearningAgents.py#L161-L205) (Q8):**
   - Biểu diễn hàm giá trị bằng tích vô hướng giữa vector trọng số `self.weights` và vector đặc trưng `features = self.featExtractor.getFeatures(state, action)`:
     $$Q(s, a) = \sum_{i} w_i f_i(s, a)$$
   - Cập nhật trọng số theo quy tắc TD-error:
     $$\delta = \big(r + \gamma \max_{a'} Q(s', a')\big) - Q(s, a), \qquad w_i \leftarrow w_i + \alpha \cdot \delta \cdot f_i(s, a)$$

---

## 5. Project 5: Machine Learning — Mạng Nơ-ron, RNN, Tích chập & Self-Attention (`machinelearning/`)

Toàn bộ các mô hình học sâu trong [`machinelearning/models.py`](file:///d:/AIT2004_7/machinelearning/models.py), hàm mất mát trong [`machinelearning/losses.py`](file:///d:/AIT2004_7/machinelearning/losses.py) và quy trình huấn luyện trong [`machinelearning/train.py`](file:///d:/AIT2004_7/machinelearning/train.py) được thiết kế chỉ sử dụng đúng các hàm PyTorch được phép import của đề bài:

| Câu | Mô hình (`models.py`) | Kiến trúc chi tiết | Hàm Loss (`losses.py`) | Cấu hình Huấn luyện (`train.py`) |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | [`PerceptronModel`](file:///d:/AIT2004_7/machinelearning/models.py#L23-L78) | Trọng số $w \in \mathbb{R}^{1 \times D}$, dự đoán $\text{sign}(x \cdot w)$ | Cập nhật trực tiếp $w \leftarrow w + yx$ | `batch_size=1`, lặp tới khi không còn mẫu sai |
| **Q2** | [`RegressionModel`](file:///d:/AIT2004_7/machinelearning/models.py#L81-L109) | MLP: $1 \to 128 \text{ (ReLU)} \to 128 \text{ (ReLU)} \to 1$ | [`regression_loss`](file:///d:/AIT2004_7/machinelearning/losses.py#L3-L14) (`mse_loss`) | `Adam(lr=0.01)`, `batch_size=32`, dừng khi loss $< 0.002$ |
| **Q3** | [`DigitClassificationModel`](file:///d:/AIT2004_7/machinelearning/models.py#L112-L156) | MLP: $784 \to 256 \text{ (ReLU)} \to 128 \text{ (ReLU)} \to 10$ | [`digitclassifier_loss`](file:///d:/AIT2004_7/machinelearning/losses.py#L18-L32) (`cross_entropy`) | `Adam(lr=0.001)`, `batch_size=128`, dừng khi val acc $\ge 97.8\%$ |
| **Q4** | [`LanguageIDModel`](file:///d:/AIT2004_7/machinelearning/models.py#L159-L218) | RNN: $h_t = \text{ReLU}(W_x x_t + W_h h_{t-1})$, hidden $256 \to 5$ | [`languageid_loss`](file:///d:/AIT2004_7/machinelearning/losses.py#L35-L50) (`cross_entropy`) | `Adam(lr=0.001)`, `batch_size=64`, `movedim(x,1,0)`, dừng khi val acc $\ge 85\%$ |
| **Q5** | [`Convolve`](file:///d:/AIT2004_7/machinelearning/models.py#L222-L249) & [`DigitConvolutionalModel`](file:///d:/AIT2004_7/machinelearning/models.py#L252-L287) | Tích chập $3\times 3$ trượt lát cắt $\to 676 \to 128 \text{ (ReLU)} \to 10$ | [`digitconvolution_Loss`](file:///d:/AIT2004_7/machinelearning/losses.py#L53-L67) (`cross_entropy`) | `Adam(lr=0.005)`, `batch_size=32`, dừng khi val acc $\ge 88\%$ |
| **Q6** | [`Attention`](file:///d:/AIT2004_7/machinelearning/models.py#L290-L332) | Causal Scaled Dot-Product Self-Attention $\text{ softmax}\big(\frac{QK^T}{\sqrt{d_k}} + M_{\text{mask}}\big)V$ | Cross-Entropy (GPT-2 LM) | Kiểm thử trực tiếp và huấn luyện mô hình ngôn ngữ |

#### Hai điểm nhấn kỹ thuật quan trọng trong `models.py`:
1. **Vector hóa phép tích chập 2D trong [`Convolve`](file:///d:/AIT2004_7/machinelearning/models.py#L222-L249):**
   Thay vì dùng 2 vòng lặp duyệt qua từng tọa độ điểm ảnh $(y, x)$ của ma trận đầu ra (tốn $O(H \cdot W)$ lời gọi hàm Python), ta chỉ lặp qua kích thước rất nhỏ của bộ lọc $(i, j) \in [0, k_h) \times [0, k_w)$ và cộng dồn trực tiếp các lát cắt tensor:
   ```python
   out_h = input_tensor_dimensions[0] - weight_dimensions[0] + 1
   out_w = input_tensor_dimensions[1] - weight_dimensions[1] + 1
   Output_Tensor = zeros((out_h, out_w))
   for i in range(weight_dimensions[0]):
       for j in range(weight_dimensions[1]):
           Output_Tensor = Output_Tensor + input[i:i + out_h, j:j + out_w] * weight[i, j]
   ```
   Cách làm này giảm số phép toán cấp Python từ hàng chục nghìn xuống chỉ còn $k_h \times k_w = 9$ phép toán tensor, giúp huấn luyện CNN nhanh gấp hàng nghìn lần.
2. **Cơ chế Causal Masking trong [`Attention.forward`](file:///d:/AIT2004_7/machinelearning/models.py#L311-L332):**
   ```python
   q = self.q_layer(input)
   k = self.k_layer(input)
   v = self.v_layer(input)
   M = matmul(q, movedim(k, 1, 2)) / (self.layer_size ** 0.5)
   M = M.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))[0]
   return matmul(softmax(M, dim=-1), v)
   ```
   Phép `movedim(k, 1, 2)` chuyển vị ma trận Key từ $(B, T, C)$ sang $(B, C, T)$ để nhân ma trận `matmul(q, ...)` tạo ra tensor điểm chú ý $(B, T, T)$. Sau khi điền $-\infty$ vào nửa trên đường chéo chính bằng `masked_fill`, hàm `softmax(M, dim=-1)` biến các vị trí tương lai thành trọng số $0$.

---

## 6. Xây dựng Kịch bản Ma trận lớn & Cơ chế Đo lường (`experiments/`)

Để đánh giá sâu sự hiệu quả và giới hạn của từng kỹ thuật khi mở rộng quy mô bài toán, thư mục [`experiments/`](file:///d:/AIT2004_7/experiments/scenario.py) thiết lập **4 môi trường ma trận cỡ lớn**:

1. **Mê cung đa đường đi (Braided Maze) từ $31 \times 31$ đến $151 \times 151$ kèm Địa hình Đầm lầy ([`generate_braided_maze`](file:///d:/AIT2004_7/experiments/scenario.py#L23-L99)):**
   - Sinh mê cung bằng Randomized DFS kết hợp đục mở thêm $\beta = 25\%$ tường chắn nội bộ để tạo ra hàng nghìn đường đi thay thế có độ dài khác nhau giữa góc $(1,1)$ và $(N-2, N-2)$.
   - Phủ ngẫu nhiên $30\%$ ô đầm lầy (`~`) có chi phí đi qua $c = 5$ (trong khi đường thường $c = 1$) thông qua lớp [`SwampPositionSearchProblem`](file:///d:/AIT2004_7/experiments/exp_search.py#L31-L65) để kiểm tra sự khác biệt giữa tìm kiếm theo số bước (`BFS`) và tìm kiếm theo chi phí (`UCS`, `A*`).
2. **Mê cung Đa mục tiêu ($31 \times 31$ với $k \in \{3, 5, 7, 9, 11\}$ viên thức ăn — [`generate_food_maze`](file:///d:/AIT2004_7/experiments/scenario.py#L102-L128)):**
   - Tăng dần số lượng mục tiêu $k$ để khảo sát sự bùng nổ tổ hợp $O(V \cdot 2^k)$ của không gian trạng thái `FoodSearchProblem`.
3. **Đấu trường đối kháng cỡ lớn $41 \times 21$ với 3 Ghost ([`arena_41x21_3ghosts.lay`](file:///d:/AIT2004_7/experiments/layouts/arena_41x21_3ghosts.lay)):**
   - Mở rộng diện tích bàn cờ gấp ~2.3 lần `mediumClassic` với 3 Ghost săn đuổi (`DirectionalGhost` / `RandomGhost`) và 4 viên Capsule ở 4 góc.
4. **Mạng lưới MDP ngẫu nhiên từ $5 \times 5$ đến $30 \times 30$ ([`generate_mdp_grid`](file:///d:/AIT2004_7/experiments/scenario.py#L205-L246)):**
   - Lưới $30 \times 30$ với $\sim 790$ trạng thái sống, xác suất trượt nhiễu $0.2$, nhiều bẫy phạt $-10$ và đích thưởng $+10, +5, +2$.

---

## 7. Độ đo Đánh giá (Metrics) & Tổng hợp Kết quả Thực nghiệm

> Toàn bộ bảng số liệu đầy đủ và phân tích chi tiết được trình bày trong [`experiments/REPORT.md`](file:///d:/AIT2004_7/experiments/REPORT.md). Dưới đây là tóm tắt các độ đo cốt lõi và kết quả định lượng nổi bật:

### 7.1. Bảng định nghĩa các Độ đo (Evaluation Metrics)

| Miền bài toán | Độ đo (Metric) | Công thức | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| **Search** | **Node Expansion Ratio** ($\eta_{\text{exp}}$) | $\frac{N_{\text{expanded}}}{|S_{\text{free}}|}$ | Tỷ lệ diện tích mê cung bị thuật toán mở rộng. |
| **Search** | **Suboptimality Ratio** ($\rho_{\text{cost}}$) | $\frac{C(\pi)}{C(\pi^*)}$ | Tỷ lệ chi phí đường đi tìm được so với nghiệm tối ưu $\pi^*$ ($\rho_{\text{cost}} = 1.00$ là tối ưu). |
| **Multi-Agent** | **Pruning Efficiency** ($\eta_{\text{prune}}$) | $1 - \frac{\text{Calls}_{\alpha\beta}}{\text{Calls}_{\text{Minimax}}}$ | Tỷ lệ nút trên cây trò chơi được cắt tỉa bởi Alpha-Beta. |
| **Multi-Agent** | **Speedup & Exact Match** | $\frac{T_{\text{Minimax}}}{T_{\alpha\beta}}$, $\mathbb{I}(a_{\alpha\beta} = a_{\text{Minimax}})$ | Hệ số tăng tốc thời gian suy nghĩ và tỷ lệ giữ nguyên nước đi tối ưu. |
| **Reinforcement** | **Backup Savings Ratio** | $\frac{\text{Backups}_{\text{Standard VI}}}{\text{Backups}_{\text{PrioritizedSweeping}}}$ | Tỷ lệ tiết kiệm số phép cập nhật Bellman để đạt $\max_s |V(s)-V^*(s)| < 10^{-3}$. |
| **Machine Learning** | **Vectorization Speedup** ($S_{\text{conv}}$) | $\frac{T_{\text{naive}}(H,W,k)}{T_{\text{vectorized}}(H,W,k)}$ | Hệ số tăng tốc của tích chập trượt lát cắt tensor so với vòng lặp điểm ảnh. |

---

### 7.2. Kết quả chính qua 8 Thực nghiệm (A – H)

1. **Thực nghiệm A — Tìm kiếm đường đi trên mê cung $31 \times 31 \to 151 \times 151$ ([`exp_search.py`](file:///d:/AIT2004_7/experiments/exp_search.py)):**
   - Trên mê cung $151 \times 151$ chi phí đồng nhất, `DFS` trả về đường đi **dài gấp $6.08\times$ tối ưu** ($1{,}828.4$ vs $300.8$), trong khi `A*-Manhattan` đạt tối ưu tuyệt đối ($1.00\times$) và giảm tỷ lệ nút mở rộng từ $89.9\%$ (`UCS`) xuống **$64.6\%$**.
   - Khi có **địa hình đầm lầy ($c=5$)** trên mê cung $151 \times 151$, `BFS` mất tính tối ưu (**độ lệch $\rho_{\text{cost}} = 1.23\times$**, chi phí $531.6$ vs $430.8$) vì đi xuyên qua đầm lầy để giảm số bước, còn `UCS` và `A*-Manhattan` vòng tránh đầm lầy và giữ vững $\rho_{\text{cost}} = 1.00\times$.

2. **Thực nghiệm B — Bùng nổ tổ hợp bài toán Multi-Food ($31 \times 31, k=3 \dots 11$):**
   - `UCS` bị **Timeout ($>300{,}000$ nút)** ngay khi $k \ge 9$.
   - `A* + foodHeuristic` giảm **75.5% số nút duyệt** ở $k=7$ ($1{,}137$ vs $4{,}644$ nút) và giải thành công $100\%$ tới $k=11$ ($58{,}959$ nút).
   - `ClosestDotSearchAgent` (tham lam BFS) giải $k=11$ chỉ trong **$0.004\text{ s}$** ($353$ nút, nhanh hơn `A*` gần $900\times$) với chi phí chỉ cao hơn tối ưu **$10\%$** ($\rho_{\text{cost}} = 1.10\times$).

3. **Thực nghiệm C & D — Cắt tỉa Alpha-Beta & Đấu trường đối kháng $41 \times 21$ (3 Ghosts) ([`exp_multiagent.py`](file:///d:/AIT2004_7/experiments/exp_multiagent.py)):**
   - `AlphaBeta` khớp quyết định với `Minimax` **100.0%** trên mọi độ sâu, đồng thời cắt bỏ **38.0% số nút** ở độ sâu $d=3$ (tăng tốc **$2.08\times$**) và cắt **23.5% số nút** trên `Arena 41x21` (3 Ghost, tăng tốc **$1.58\times$**).
   - Trên đấu trường lớn `Arena 41x21` với 3 `DirectionalGhost` hung hãn, **`Expectimax-d2 + betterEval`** đạt điểm cao nhất (**$+551.4$**, dọn **$38.8\%$** thức ăn), vượt xa `AlphaBeta-d2` (**$-5.4$**, $22.3\%$ thức ăn) do `AlphaBeta` quá bi quan trước 3 Ghost.

4. **Thực nghiệm E & F — Hội tụ MDP $30 \times 30$ & Khái quát hóa Q-Learning ([`exp_rl.py`](file:///d:/AIT2004_7/experiments/exp_rl.py)):**
   - Trên MDP $30 \times 30$ ($786$ trạng thái), `Prioritized Sweeping` chỉ cần **$11{,}900$ phép cập nhật Bellman** để hội tụ ($\varepsilon < 10^{-3}$) so với **$50{,}166$ phép cập nhật** của `Standard VI` (**tiết kiệm $4.2\times$**).
   - Trên `mediumClassic` ($\sim 10^{30}$ trạng thái), `Tabular Q-Learning` học $182{,}412$ trạng thái sau $1{,}000$ ván vẫn đạt **Win Rate $0.0\%$** ($-401.0$ điểm), trong khi **`Approximate Q-Learning`** với đúng **4 trọng số đặc trưng** đạt **Win Rate $92.5\%$** ($+1{,}241.2$ điểm) chỉ sau **50 ván huấn luyện** (và đạt **$95.0\%$** ở 200 ván).

5. **Thực nghiệm G & H — Tăng tốc Vector hóa Convolve & Bậc hai Self-Attention ([`exp_ml.py`](file:///d:/AIT2004_7/experiments/exp_ml.py)):**
   - Cài đặt vector hóa `models.Convolve` đạt tốc độ **$0.90\text{ ms}$** so với **$2{,}078.31\text{ ms}$** của vòng lặp pixel trên ma trận $128 \times 128$ ($k=5$), tức **tăng tốc $2{,}313.8\times$** với sai số số học $< 10^{-6}$.
   - Khi tăng chiều dài chuỗi $T$ lên $32\times$ ($16 \to 512$), bộ nhớ ma trận chú ý $(B, T, T)$ trong `models.Attention` tăng đúng **$1{,}024\times$** ($8\text{ KB} \to 8{,}192\text{ KB}$), minh họa trực quan nút thắt cổ chai bậc hai $O(T^2)$ của Transformer.

---

## 8. Hướng dẫn chạy Kiểm thử & Tái lập Thực nghiệm

### 8.1. Chạy Autograder kiểm tra 4 Project
```powershell
# Project 1: Search (26/25)
cd d:\AIT2004_7\search; python autograder.py --no-graphics

# Project 2: Multi-Agent Search (25/25)
cd d:\AIT2004_7\multiagent; python autograder.py --no-graphics

# Project 3: Reinforcement Learning (26/25)
cd d:\AIT2004_7\reinforcement; python autograder.py --no-graphics

# Project 5: Machine Learning (25/25)
cd d:\AIT2004_7\machinelearning; python autograder.py --no-graphics
```

### 8.2. Chạy lại toàn bộ Kịch bản Ma trận lớn & Xuất báo cáo
```powershell
cd d:\AIT2004_7
python experiments/scenario.py
python experiments/exp_search.py
python experiments/exp_multiagent.py
python experiments/exp_rl.py
python experiments/exp_ml.py
python experiments/summarize.py
```

- Xem báo cáo phân tích số liệu đầy đủ tại: [`experiments/REPORT.md`](file:///d:/AIT2004_7/experiments/REPORT.md)
- Xem hướng dẫn giải thích chuyên sâu từng đoạn code tại: [`experiments/EXPLANATION_GUIDE.md`](file:///d:/AIT2004_7/experiments/EXPLANATION_GUIDE.md)
