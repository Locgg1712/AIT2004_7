# Hướng Dẫn & Giải Thích Chi Tiết Toàn Bộ Cách Làm (Code Walkthrough & Phương Pháp Thực Nghiệm)

Tài liệu này giải thích chi tiết **cách hiện thực từng thuật toán trong 4 bài tập cốt lõi (Projects 1, 2, 3, 5)** và **cách xây dựng kịch bản ma trận lớn kèm hệ thống đo lường tự động (`experiments/`)**. Tài liệu được thiết kế để bạn nắm vững từng dòng code, công thức toán học, lý do lựa chọn thiết kế và có thể giải thích/bảo vệ đồ án một cách trọn vẹn.

---

## Mục Lục

1. [Tổng quan kiến trúc dự án](#1-tổng-quan-kiến-trúc-dự-án)
2. [Project 1: Search — Thuật toán Tìm kiếm & Thiết kế Heuristic (`search/`)](#2-project-1-search--thuật-toán-tìm-kiếm--thiết-kế-heuristic-search)
3. [Project 2: Multi-Agent Search — Tìm kiếm Đối kháng & Hàm Lượng giá (`multiagent/`)](#3-project-2-multi-agent-search--tìm-kiếm-đối-kháng--hàm-lượng-giá-multiagent)
4. [Project 3: Reinforcement Learning — Quy hoạch động & Học tăng cường (`reinforcement/`)](#4-project-3-reinforcement-learning--quy-hoạch-động--học-tăng-cường-reinforcement)
5. [Project 5: Machine Learning — Mạng Nơ-ron, RNN, Tích chập & Self-Attention (`machinelearning/`)](#5-project-5-machine-learning--mạng-nơ-ron-rnn-tích-chập--self-attention-machinelearning)
6. [Kịch bản Ma trận lớn & Hệ thống Đo lường Thực nghiệm (`experiments/`)](#6-kịch-bản-ma-trận-lớn--hệ-thống-đo-lường-thực-nghiệm-experiments)
7. [Hướng dẫn chạy kiểm thử & Tái lập kết quả](#7-hướng-dẫn-chạy-kiểm-thử--tái-lập-kết-quả)

---

## 1. Tổng quan kiến trúc dự án

Toàn bộ kho mã nguồn được chia thành **2 tầng tách biệt**:

```
d:\AIT2004_7\
├── .gitignore                        # Whitelist chỉ cho phép git theo dõi đúng 10 file cần nộp
├── search/                           # Project 1: Tìm kiếm có/không có thông tin (26/25 điểm)
│   ├── search.py                     # DFS, BFS, UCS, A*
│   └── searchAgents.py               # CornersProblem, cornersHeuristic, foodHeuristic, ClosestDot
├── multiagent/                       # Project 2: Tìm kiếm đa tác tử đối kháng (25/25 điểm)
│   └── multiAgents.py                # Reflex, Minimax, AlphaBeta, Expectimax, betterEvaluationFunction
├── reinforcement/                    # Project 3: Học tăng cường & MDP (26/25 điểm)
│   ├── valueIterationAgents.py       # Value Iteration chuẩn & Prioritized Sweeping
│   ├── qlearningAgents.py            # Tabular Q-Learning, Epsilon-Greedy, Approximate Q-Learning
│   └── analysis.py                   # Tinh chỉnh siêu tham số MDP (Discount, Noise, LivingReward)
├── machinelearning/                  # Project 5: Học sâu với PyTorch (25/25 điểm)
│   ├── models.py                     # Perceptron, MLP, RNN, Vectorized Convolve 2D, Causal Attention
│   ├── losses.py                     # MSE Loss & Cross-Entropy Loss
│   └── train.py                      # Vòng lặp huấn luyện tối ưu hóa Adam / Perceptron
└── experiments/                      # Kịch bản ma trận lớn & Bộ đánh giá hiệu năng (Tự động ignore bởi Git)
    ├── scenario.py                   # Trình sinh mê cung Braided 151x151, Đầm lầy, Arena 41x21, MDP 30x30
    ├── exp_search.py                 # Thực nghiệm A & B (Độ phức tạp không gian & Bùng nổ tổ hợp)
    ├── exp_multiagent.py             # Thực nghiệm C & D (Hiệu suất cắt tỉa Alpha-Beta & Đối kháng 3 Ghost)
    ├── exp_rl.py                     # Thực nghiệm E & F (Tốc độ hội tụ Bellman & Khái quát hóa đặc trưng)
    ├── exp_ml.py                     # Thực nghiệm G & H (Tăng tốc Vector hóa Convolve & Bậc hai Attention)
    ├── summarize.py                  # Tổng hợp bảng số liệu từ CSV
    ├── REPORT.md                     # Báo cáo phân tích kết quả thực nghiệm và độ đo
    └── EXPLANATION_GUIDE.md          # Tài liệu giải thích chi tiết cách làm (file này)
```

---

## 2. Project 1: Search — Thuật toán Tìm kiếm & Thiết kế Heuristic (`search/`)

### 2.1. Khung tìm kiếm đồ thị tổng quát (`search/search.py`)

Cả 4 thuật toán trong [`search.py`](file:///d:/AIT2004_7/search/search.py) đều tuân theo mô hình **Graph Search** (tìm kiếm trên đồ thị có lưu vết các trạng thái đã mở rộng để tránh lặp vô hạn trên đồ thị có chu trình), chỉ khác nhau ở **cấu trúc dữ liệu của tập biên (`frontier`)** và **cách kiểm soát chi phí đường đi**:

1. **Tìm kiếm theo chiều sâu — [`depthFirstSearch`](file:///d:/AIT2004_7/search/search.py#L75-L107) (Q1):**
   - **Cấu trúc biên:** Ngăn xếp LIFO (`util.Stack()`). Mỗi phần tử trong ngăn xếp là một cặp `(state, actions)` lưu trạng thái hiện tại và danh sách hành động từ gốc tới trạng thái đó.
   - **Cơ chế hoạt động:** Luôn lấy nút mới nhất vừa đẩy vào (`frontier.pop()`) để đi sâu nhất có thể theo một nhánh trước khi quay lui (backtrack).
   - **Tại sao kiểm tra `if state in visited:` khi `pop()` chứ không đánh dấu `visited` lúc `push()`?**
     Trong CS188 autograder, một nút chỉ được tính là **đã mở rộng (expanded)** khi ta thực sự lấy nó ra khỏi `frontier` và gọi `problem.getSuccessors(state)`. Nếu đánh dấu `visited` ngay lúc `push()`, thứ tự duyệt của DFS sẽ bị lệch so với chuẩn Graph Search và không khớp số nút mở rộng của autograder.

2. **Tìm kiếm theo chiều rộng — [`breadthFirstSearch`](file:///d:/AIT2004_7/search/search.py#L109-L129) (Q2):**
   - **Cấu trúc biên:** Hàng đợi FIFO (`util.Queue()`).
   - **Cơ chế hoạt động:** Mở rộng các nút theo từng lớp độ sâu (số bước đi $d = 0, 1, 2, \dots$). Đảm bảo tìm được đường đi có **ít bước nhất**, nhưng **không tối ưu nếu chi phí mỗi bước khác nhau** (như đã chứng minh ở kịch bản đầm lầy $c=5$).

3. **Tìm kiếm chi phí đồng nhất — [`uniformCostSearch`](file:///d:/AIT2004_7/search/search.py#L131-L156) (Q3):**
   - **Cấu trúc biên:** Hàng đợi ưu tiên (`util.PriorityQueue()`), sắp xếp theo tổng chi phí thực tế từ điểm đầu $g(n) = \text{cost}$.
   - **Xử lý trùng lặp với chi phí tốt hơn (`best_cost` và `expanded_cost`):**
     Khi một trạng thái `successor` có thể đi tới bằng nhiều đường với chi phí khác nhau, `PriorityQueue` có thể chứa nhiều bản sao của cùng một `state`. Khi lấy `(state, actions, cost)` ra khỏi hàng đợi:
     ```python
     if state in best_cost and cost > best_cost[state]:
         continue
     if state in expanded_cost and cost >= expanded_cost[state]:
         continue
     best_cost[state] = cost
     expanded_cost[state] = cost
     ```
     Điều này đảm bảo mỗi trạng thái chỉ được gọi `getSuccessors` đúng một lần tại chi phí nhỏ nhất của nó.

4. **Tìm kiếm A* — [`aStarSearch`](file:///d:/AIT2004_7/search/search.py#L165-L192) (Q4):**
   - Giống hệt cấu trúc của `uniformCostSearch`, nhưng độ ưu tiên khi đẩy vào `PriorityQueue` là tổng của chi phí thực tế $g(n)$ và ước lượng khoảng cách còn lại $h(n)$:
     $$f(n) = g(n) + h(n) = \text{new\_cost} + \text{heuristic}(\text{successor}, \text{problem})$$
   - Lưu ý quan trọng: trong tuple lưu vào hàng đợi `(successor, actions + [action], new_cost)`, ta vẫn lưu **`new_cost` (tức $g(n)$)** chứ không lưu $f(n)$, vì khi tính tiếp nút con kế tiếp, chi phí cộng dồn phải là $g(n') = g(n) + c(n, a, n')$.

---

### 2.2. Biểu diễn trạng thái & Heuristic (`search/searchAgents.py`)

1. **Bài toán 4 góc — [`CornersProblem`](file:///d:/AIT2004_7/search/searchAgents.py#L274-L359) (Q5):**
   - **Ý tưởng biểu diễn trạng thái:** Để biết đã đi qua đủ 4 góc hay chưa mà không lưu những thông tin thừa (như toàn bộ lịch sử đường đi), trạng thái được định nghĩa tối giản là:
     $$\text{state} = \big((x, y),\; \text{frozenset}(\text{visited\_corners})\big)$$
   - **Tại sao dùng `frozenset`?** Trong Python, `set` thông thường là kiểu khả biến (mutable) nên không băm được (`unhashable`), không thể đưa `state` vào tập `visited = set()` hay làm khóa cho từ điển `best_cost` trong `BFS`/`A*`. `frozenset` là bất biến (immutable) và băm được, đồng thời phép hợp `visited | frozenset((next_position,))` tạo ra trạng thái mới một cách an toàn mà không làm biến đổi trạng thái cha.
   - Kích thước không gian trạng thái tối đa chỉ là $|V| \times 2^4 = 16|V|$.

2. **Heuristic cho bài toán 4 góc — [`cornersHeuristic`](file:///d:/AIT2004_7/search/searchAgents.py#L361-L398) (Q6):**
   - **Cách làm:** Với tập các góc chưa thăm `remaining` (tối đa $4! = 24$ hoán vị), ta thử mọi thứ tự đi qua các góc còn lại $\pi \in \text{permutations}(\text{remaining})$ và tính tổng khoảng cách mê cung thực tế (`mazeDistance`) dọc theo hành trình đó:
     $$h(s) = \min_{\pi \in \text{Perm}(\text{remaining})} \left[ d_{\text{maze}}(\text{pos}, \pi_1) + \sum_{i=1}^{|\text{remaining}|-1} d_{\text{maze}}(\pi_i, \pi_{i+1}) \right]$$
   - **Tối ưu hóa bộ nhớ đệm (`problem.heuristicInfo`):** Hàm `mazeDistance(p1, p2)` chạy một lượt BFS nhỏ nên nếu gọi lặp đi lặp lại sẽ rất chậm. Ta lưu kết quả vào `problem.heuristicInfo[tuple(sorted((point1, point2)))]` để mỗi cặp điểm chỉ tính BFS đúng 1 lần duy nhất trong suốt quá trình tìm kiếm.
   - **Tại sao vừa Admissible vừa Consistent?** Vì mọi đường đi hợp lệ thăm hết các góc còn lại đều phải đi qua chúng theo một thứ tự $\pi$ nào đó, và độ dài đường đi ngắn nhất theo thứ tự $\pi$ trong mê cung chính là tổng `mazeDistance` giữa các điểm liên tiếp. Do đó $h(s)$ bằng chính xác chi phí tối ưu còn lại $h^*(s)$ của bài toán 4 góc!

3. **Heuristic ăn toàn bộ thức ăn — [`foodHeuristic`](file:///d:/AIT2004_7/search/searchAgents.py#L462-L504) (Q7):**
   - **Thách thức:** Khi có $k$ viên thức ăn (có thể lên tới hàng chục viên), ta không thể duyệt $k!$ hoán vị như bài toán 4 góc.
   - **Cách làm:** Ta định nghĩa heuristic là **khoảng cách mê cung thật lớn nhất từ vị trí hiện tại của Pacman tới viên thức ăn xa nhất còn lại**:
     $$h(\text{pos}, \text{Food}) = \max_{f \in \text{Food}} d_{\text{maze}}(\text{pos}, f)$$
     kết hợp lưu cache đối xứng `key = tuple(sorted((position, food_position)))` trong `problem.heuristicInfo`.
   - **Chứng minh tính Admissible & Consistent:**
     - *Admissible:* Để ăn hết mọi viên thức ăn trong $\text{Food}$, Pacman bắt buộc phải đi tới viên thức ăn xa nhất $f^*$. Ngay cả khi chỉ đi thẳng tới $f^*$, Pacman đã tốn ít nhất $d_{\text{maze}}(\text{pos}, f^*)$ bước. Vậy $h(s) \le h^*(s)$.
     - *Consistent:* Khi Pacman đi 1 bước từ $u$ sang $v$ ($c(u,v)=1$), theo bất đẳng thức tam giác trên đồ thị mê cung, khoảng cách tới bất kỳ viên thức ăn $f$ nào giảm tối đa 1 đơn vị: $d_{\text{maze}}(u, f) \le 1 + d_{\text{maze}}(v, f)$. Lấy $\max$ hai vế theo $f$, ta có $h(u) \le 1 + h(v)$.
   - Nhờ tính chất này, trên `trickySearch`, `A*` chỉ mở rộng ~4,137 nút (đạt tối đa 5/4 điểm cộng của autograder).

4. **Tìm kiếm tham lam viên gần nhất — [`ClosestDotSearchAgent`](file:///d:/AIT2004_7/search/searchAgents.py#L506-L570) (Q8):**
   - Định nghĩa [`AnyFoodSearchProblem`](file:///d:/AIT2004_7/search/searchAgents.py#L536-L569) kế thừa `PositionSearchProblem`, trong đó điều kiện đích `isGoalState(state)` đơn giản kiểm tra `self.food[x][y] == True`.
   - Trong `findPathToClosestDot`, ta gọi `search.breadthFirstSearch(problem)`: vì BFS duyệt theo lớp khoảng cách tăng dần, viên thức ăn đầu tiên nó chạm tới chắc chắn là viên gần nhất tính theo đường mê cung.

---

## 3. Project 2: Multi-Agent Search — Tìm kiếm Đối kháng & Hàm Lượng giá (`multiagent/`)

### 3.1. Quản lý lượt đi đa tác tử (Ply vs. Agent Turn)

Trong Pacman đối kháng có $N$ tác tử (`numAgents = gameState.getNumAgents()`):
- `agentIndex == 0`: **Pacman** (tác tử tối đa hóa điểm — `MAX`).
- `agentIndex >= 1`: **Các Ghost** $1, 2, \dots, N-1$ (các tác tử tối thiểu hóa điểm trong Minimax/Alpha-Beta hoặc lấy kỳ vọng ngẫu nhiên trong Expectimax).
- **Quy ước độ sâu `self.depth`:** Một đơn vị độ sâu (**1 ply**) tương ứng với **1 lượt đi của Pacman cộng với lượt đi trả lời của tất cả $N-1$ Ghost**. Do đó, trong cả 3 thuật toán (`MinimaxAgent`, `AlphaBetaAgent`, `ExpectimaxAgent`), việc chuyển trạng thái đệ quy được viết thống nhất như sau:
  ```python
  nextAgent = (agentIndex + 1) % numAgents
  nextDepth = depth + 1 if nextAgent == 0 else depth
  ```
  Nghĩa là chỉ khi tác tử cuối cùng (`agentIndex == numAgents - 1`) đi xong và quay vòng lại Pacman (`nextAgent == 0`), biến `depth` mới tăng thêm 1. Khi `depth == self.depth` hoặc `state.isWin()` hoặc `state.isLose()`, đệ quy dừng và trả về `self.evaluationFunction(state)`.

### 3.2. Chi tiết từng Agent trong [`multiAgents.py`](file:///d:/AIT2004_7/multiagent/multiAgents.py)

1. **[`ReflexAgent`](file:///d:/AIT2004_7/multiagent/multiAgents.py#L22-L103) (Q1):**
   - Đánh giá trực tiếp cặp `(currentGameState, action)` thông qua trạng thái kế tiếp `successorGameState = currentGameState.generatePacmanSuccessor(action)`.
   - Sử dụng **nghịch đảo khoảng cách** (`10.0 / (minFoodDist + 1.0)`) thay vì trừ trực tiếp khoảng cách để tạo lực hút phi tuyến mạnh dần khi tiến sát viên thức ăn.
   - Nếu có Ghost không hoảng sợ ở khoảng cách Manhattan $\le 1$, trả về ngay `float('-inf')` để tuyệt đối tránh tự sát; nếu Ghost đang hoảng sợ (`scaredTimer > 0`), cộng điểm thưởng lớn `20.0 / (dist + 1.0)` để khuyến khích săn Ghost.

2. **[`MinimaxAgent`](file:///d:/AIT2004_7/multiagent/multiAgents.py#L135-L206) (Q2):**
   - Cài đặt công thức chuẩn:
     $$V(s, \text{agent}, d) = \begin{cases} \text{eval}(s) & \text{nếu } d = D \text{ hoặc Terminal}(s) \\ \max_{a \in A(s)} V(\text{succ}(s,a), \text{nextAgent}, d') & \text{nếu agent} = 0 \\ \min_{a \in A(s)} V(\text{succ}(s,a), \text{nextAgent}, d') & \text{nếu agent} \ge 1 \end{cases}$$
   - **Xử lý edge-case thua bắt buộc (`bestAction is None`):** Ở vòng lặp gốc của Pacman (dòng 202), ta viết `if score > bestScore or bestAction is None:`. Nếu mọi nước đi hợp lệ của Pacman đều dẫn tới cái chết không thể tránh khỏi ở tầng sâu hơn (`score == -inf`), điều kiện `or bestAction is None` đảm bảo Pacman vẫn trả về một hành động hợp lệ đầu tiên thay vì `None` (tránh lỗi `Illegal action None`).

3. **[`AlphaBetaAgent`](file:///d:/AIT2004_7/multiagent/multiAgents.py#L208-L265) (Q3):**
   - Duy trì hai cận dọc theo đường đi trên cây tìm kiếm:
     - $\alpha$: Giá trị tốt nhất (lớn nhất) mà `MAX` (Pacman) đã đảm bảo được trên đường từ gốc tới nút hiện tại.
     - $\beta$: Giá trị tốt nhất (nhỏ nhất) mà `MIN` (các Ghost) đã đảm bảo được trên đường từ gốc tới nút hiện tại.
   - **Tại sao dùng so sánh ngặt `if v > beta: return v` (ở nút MAX) và `if v < alpha: return v` (ở nút MIN)?**
     Nếu cắt tỉa ngay khi `v >= beta` hoặc `v <= alpha` (tức cắt cả khi bằng), trong một số cấu hình cây có nhiều nút cùng giá trị điểm, ta sẽ bỏ qua một số lần gọi `generateSuccessor` mà bộ chấm điểm CS188 yêu cầu phải kiểm tra để xác định cách phá hòa. Dùng dấu `>` và `<` vừa cắt tỉa toàn bộ nhánh vô ích thực sự, vừa vượt qua tuyệt đối bài test kiểm tra cấu trúc cây của autograder.
   - Lưu ý ở vòng lặp gốc (`getAction`), sau mỗi hành động của Pacman, ta cũng cập nhật `alpha = max(alpha, bestScore)` để các nhánh con kế tiếp của gốc được hưởng lợi từ cận dưới $\alpha$ mới nhất.

4. **[`ExpectimaxAgent`](file:///d:/AIT2004_7/multiagent/multiAgents.py#L267-L320) (Q4):**
   - Thay thế nút `MIN` của Ghost bằng **nút cơ hội (Chance node)** với phân phối đều trên tập hành động hợp lệ:
     $$V(s, \text{agent} \ge 1, d) = \frac{1}{|A(s)|} \sum_{a \in A(s)} V(\text{succ}(s,a), \text{nextAgent}, d')$$
   - Giúp Pacman không bị "tê liệt vì quá bi quan" khi đối mặt với nhiều Ghost không hoàn hảo hoặc trong không gian rộng.

5. **Hàm lượng giá trạng thái nâng cao — [`betterEvaluationFunction`](file:///d:/AIT2004_7/multiagent/multiAgents.py#L322-L370) (Q5):**
   - Khác với `ReflexAgent` (đánh giá hành động), `betterEvaluationFunction` đánh giá một **trạng thái tĩnh** `currentGameState` ở lá của cây tìm kiếm độ sâu $d$:
     $$\text{Eval}(s) = \text{Score}(s) + \frac{10}{d_{\text{minFood}} + 1} - 4 N_{\text{food}} + \frac{5}{d_{\text{minCap}} + 1} - 20 N_{\text{cap}} + \sum_{g \in \text{Ghosts}} \phi(g)$$
     trong đó:
     $$\phi(g) = \begin{cases} +\frac{50}{d(s, g) + 1} & \text{nếu } \text{scaredTimer}(g) > 0 \\ -500 & \text{nếu } \text{scaredTimer}(g) = 0 \text{ và } d(s, g) \le 1 \\ -\frac{2}{d(s, g) + 1} & \text{nếu } \text{scaredTimer}(g) = 0 \text{ và } d(s, g) > 1 \end{cases}$$
   - **Tại sao cần kết hợp cả khoảng cách (`10 / (minFoodDist + 1)`) lẫn số lượng viên còn lại (`-4 * len(foodList)`)?**
     Nếu chỉ có `10 / (minFoodDist + 1)`, khi Pacman đứng cạnh viên thức ăn $A$ ($d=1 \implies +5$ điểm), việc ăn viên $A$ làm viên gần nhất tiếp theo trở thành viên $B$ ở xa ($d=10 \implies +0.9$ điểm) khiến điểm lượng giá bị **tụt giảm** sau khi ăn! Nhờ trừ `-4 * len(foodList)` (và cộng điểm nội tại của trò chơi khi ăn), việc giảm $N_{\text{food}}$ đi 1 viên luôn mang lại lợi ích dương lớn hơn độ sụt giảm khoảng cách, buộc Pacman dứt khoát ăn thức ăn thay vì đứng cạnh chờ đợi.

---

## 4. Project 3: Reinforcement Learning — Quy hoạch động & Học tăng cường (`reinforcement/`)

### 4.1. Value Iteration & Prioritized Sweeping (`reinforcement/valueIterationAgents.py`)

1. **[`ValueIterationAgent`](file:///d:/AIT2004_7/reinforcement/valueIterationAgents.py#L34-L120) (Q1):**
   - Thực hiện phương trình cập nhật Bellman theo lô (**Batch / Synchronous Update**) trong `runValueIteration`:
     $$V_{k+1}(s) = \max_{a \in A(s)} \sum_{s'} T(s, a, s') \big[ R(s, a, s') + \gamma V_k(s') \big]$$
   - **Tại sao phải tạo `newValues = util.Counter()` ở mỗi vòng lặp?**
     Vì theo định nghĩa quy hoạch động đồng bộ, toàn bộ giá trị $V_{k+1}(s)$ ở bước $k+1$ phải được tính từ vector giá trị $V_k$ của bước $k$ (thông qua `computeQValueFromValues` đọc từ `self.values`). Sau khi tính xong hết mọi trạng thái $s \in S$, ta mới gán `self.values = newValues`.

2. **[`PrioritizedSweepingValueIterationAgent`](file:///d:/AIT2004_7/reinforcement/valueIterationAgents.py#L122-L174) (Q4):**
   - **Vấn đề của Standard Value Iteration:** Ở những vòng lặp đầu tiên trên lưới lớn (như $30 \times 30$), phần lớn các ô xa đích đều có $V(s) = 0$ và sau khi tính Bellman vẫn giữ nguyên $0$, gây lãng phí hàng chục nghìn phép tính vô ích.
   - **Cách hoạt động của Prioritized Sweeping (3 pha):**
     1. **Xây dựng đồ thị tiền nhiệm (`predecessors`):** Duyệt qua mọi $(s, a, s')$, nếu xác suất chuyển $T(s, a, s') > 0$ thì thêm $s$ vào tập `predecessors[s']`.
     2. **Khởi tạo hàng đợi ưu tiên (`pq`):** Với mỗi trạng thái không phải kết thúc $s$, tính sai số Bellman hiện tại:
        $$\text{diff}(s) = \left| V(s) - \max_{a} Q(s, a) \right|$$
        Đẩy vào hàng đợi ưu tiên với khóa **`-diff`** (`pq.update(state, -diff)`) vì `util.PriorityQueue` lấy phần tử có khóa **nhỏ nhất** ra trước (nhờ dấu âm, trạng thái có sai số $\text{diff}$ **lớn nhất** sẽ được lấy ra đầu tiên).
     3. **Lan truyền ngược theo ngưỡng $\theta$ (`self.theta`):** Trong mỗi bước lặp (tối đa `self.iterations` bước):
        - Lấy trạng thái `state` có sai số lớn nhất ra khỏi `pq` và cập nhật trực tiếp `self.values[state] = max_a Q(state, a)`.
        - Chỉ kiểm tra các trạng thái tiền nhiệm `p in predecessors[state]`: tính lại $\text{diff}(p) = |V(p) - \max_a Q(p, a)|$. Nếu $\text{diff}(p) > \theta$, đẩy/cập nhật $p$ vào `pq` với độ ưu tiên `-diff(p)`.

---

### 4.2. Giải thích các câu hỏi phân tích chính sách (`reinforcement/analysis.py`)

- **[`question2`](file:///d:/AIT2004_7/reinforcement/analysis.py#L22-L25) (BridgeGrid):** Mặc định $\gamma = 0.9, \text{noise} = 0.2$ khiến tác tử sợ rơi xuống vực phạt $-100$ hai bên cầu nên không dám qua cầu lấy đích $+10$. Chỉ cần giảm **`answerNoise = 0.0`**, xác suất trượt ngang bằng $0$, tác tử tự tin đi thẳng qua cầu.
- **[`question3a` – `question3e`](file:///d:/AIT2004_7/reinforcement/analysis.py#L27-L60) (DiscountGrid):** Bàn cờ có đích gần $+1$ (cạnh vực $-10$) và đích xa $+10$, có 2 đường đi: đường dưới sát vực (ngắn nhưng nguy hiểm) và đường vòng phía trên (dài hơn 2 bước nhưng an toàn tuyệt đối). Cách thiết kế bảng tham số trực giao:

  | Câu | Mục tiêu chính sách | `Discount` ($\gamma$) | `Noise` | `LivingReward` | Giải thích cơ chế toán học |
  | :--- | :--- | :---: | :---: | :---: | :--- |
  | **3a** | Thích đích gần ($+1$), mạo hiểm đi sát vực | `0.2` | `0.0` | `0.0` | $\gamma=0.2$ làm phần thưởng xa $+10$ bị chiết khấu cực mạnh ($10 \times 0.2^k \ll 1$), và `noise=0.0` nên đi đường ngắn sát vực không sợ rơi. |
  | **3b** | Thích đích gần ($+1$), nhưng đi đường vòng tránh vực | `0.2` | `0.2` | `0.0` | Giữ $\gamma=0.2$ để vẫn chọn đích gần $+1$, nhưng bật `noise=0.2` để phạt nặng đường đi sát vực $(-10)$, buộc tác tử đi vòng lên trên rồi rẽ xuống đích $+1$. |
  | **3c** | Thích đích xa ($+10$), mạo hiểm đi sát vực | `0.9` | `0.0` | `0.0` | $\gamma=0.9$ giữ nguyên giá trị lớn của đích xa $+10$, kết hợp `noise=0.0` nên tác tử chọn đường thẳng ngắn nhất sát vực. |
  | **3d** | Thích đích xa ($+10$), đi đường vòng tránh vực | `0.9` | `0.2` | `0.0` | $\gamma=0.9$ hướng tới đích xa $+10$, còn `noise=0.2` khiến đường sát vực có kỳ vọng âm, buộc tác tử đi đường vòng phía bắc. |
  | **3e** | Không bao giờ kết thúc (tránh cả đích $+1$ và $+10$) | `1.0` | `0.0` | `10.0` | Thưởng sinh tồn mỗi bước `LivingReward = +10.0` lớn hơn hoặc bằng việc kết thúc trò chơi, nên tác tử đi lòng vòng mãi mãi để thu thưởng vô hạn. |

- **[`question7`](file:///d:/AIT2004_7/reinforcement/analysis.py#L62-L63):** Với chỉ 50 tập huấn luyện trên `BridgeGrid`, không có giá trị $\varepsilon$ và $\alpha$ nào đảm bảo 100% tìm được chính sách tối ưu (nếu $\varepsilon$ nhỏ thì không khám phá tới đầu bên kia cầu; nếu $\varepsilon$ lớn thì đi ngẫu nhiên rơi xuống vực quá nhiều), nên đáp án là `'NOT POSSIBLE'`.

---

### 4.3. Q-Learning, Epsilon-Greedy & Approximate Q-Learning (`reinforcement/qlearningAgents.py`)

1. **[`QLearningAgent`](file:///d:/AIT2004_7/reinforcement/qlearningAgents.py#L21-L127) (Q5 & Q6):**
   - Lưu bảng $Q(s,a)$ bằng `self.qValues = util.Counter()` (mặc định trả về `0.0` cho cặp $(s,a)$ chưa gặp).
   - **Phá hòa ngẫu nhiên trong `computeActionFromQValues`:** Khi có nhiều hành động cùng đạt giá trị $Q$ lớn nhất (đặc biệt lúc đầu khi mọi $Q(s,a)=0$), ta gom tất cả vào danh sách `bestActions` và gọi `random.choice(bestActions)` để khám phá đều các hướng thay vì bị kẹt ở hướng đầu tiên.
   - **Cập nhật TD Off-policy (`update`):**
     $$\text{sample} = r + \gamma \max_{a'} Q(s', a'), \qquad Q(s,a) \leftarrow (1 - \alpha) Q(s,a) + \alpha \cdot \text{sample}$$
   - **Hành động $\varepsilon$-greedy (`getAction`):** Dùng `util.flipCoin(self.epsilon)` — với xác suất $\varepsilon$ chọn ngẫu nhiên `random.choice(legalActions)`, ngược lại chọn hành động tốt nhất từ `computeActionFromQValues(state)`.

2. **[`ApproximateQAgent`](file:///d:/AIT2004_7/reinforcement/qlearningAgents.py#L161-L205) (Q8):**
   - Thay vì lưu từng trạng thái rời rạc, ta xấp xỉ tuyến tính hàm $Q(s,a)$ qua vector đặc trưng $f(s,a) = [f_1(s,a), \dots, f_m(s,a)]^T$ và vector trọng số $w$:
     $$Q(s, a) = \sum_{i=1}^m w_i f_i(s, a) = \text{self.weights} * \text{features}$$
     (toán tử `*` giữa hai `util.Counter` trong CS188 được cài sẵn là tích vô hướng dot product!).
   - Khi quan sát chuyển dịch $(s, a, r, s')$, ta tính sai số TD:
     $$\delta = \big(r + \gamma \max_{a'} Q(s', a')\big) - Q(s, a)$$
     và cập nhật từng trọng số theo hướng gradient:
     $$w_i \leftarrow w_i + \alpha \cdot \delta \cdot f_i(s, a)$$
   - Nhờ chia sẻ bộ trọng số $w$ trên toàn bộ không gian trạng thái, chỉ cần Pacman học được một lần rằng "tiến gần Ghost là xấu" ($w_{\text{ghost}} < 0$) và "ăn thức ăn là tốt" ($w_{\text{food}} > 0$), kiến thức đó lập tức áp dụng được cho mọi vị trí trên bản đồ!

---

## 5. Project 5: Machine Learning — Mạng Nơ-ron, RNN, Tích chập & Self-Attention (`machinelearning/`)

### 5.1. Perceptron & Mạng MLP (`models.py`, `losses.py`, `train.py`)

1. **[`PerceptronModel`](file:///d:/AIT2004_7/machinelearning/models.py#L23-L78) & [`train_perceptron`](file:///d:/AIT2004_7/machinelearning/train.py#L22-L44) (Q1):**
   - Khởi tạo trọng số `self.w = Parameter(ones((1, dimensions)))`.
   - Hàm `forward(x)` tính tích vô hướng $\langle x, w \rangle$ bằng `tensordot(x.reshape(-1), self.w.reshape(-1), dims=1).reshape(1)`.
   - Hàm `get_prediction(x)` trả về $+1$ nếu điểm số $\ge 0$, ngược lại $-1$.
   - Trong `train_perceptron`, ta lặp qua `DataLoader(dataset, batch_size=1, shuffle=True)` bên trong ngữ cảnh `with no_grad():`. Nếu `model.get_prediction(x) != y.item()`, cập nhật trực tiếp `model.w += y * x` và đặt cờ `converged = False`. Vòng lặp kết thúc khi chạy trọn 1 epoch mà `converged == True`.

2. **[`RegressionModel`](file:///d:/AIT2004_7/machinelearning/models.py#L81-L109) (Q2 — Xấp xỉ hàm $\sin(x)$):**
   - Kiến trúc MLP 3 tầng tuyến tính xen kẽ hàm kích hoạt phi tuyến ReLU:
     $$\text{Input }(B \times 1) \xrightarrow{\text{Linear}} 128 \xrightarrow{\text{ReLU}} \xrightarrow{\text{Linear}} 128 \xrightarrow{\text{ReLU}} \xrightarrow{\text{Linear}} \text{Output }(B \times 1)$$
   - Lưu ý **tầng cuối cùng không có ReLU** vì hàm $\sin(x)$ nhận cả giá trị âm trong $[-1, 1]$ (nếu đặt ReLU ở tầng cuối, đầu ra sẽ bị cắt cụt $\ge 0$).
   - Huấn luyện bằng `optim.Adam(model.parameters(), lr=0.01)`, `batch_size=32`, hàm mất mát [`regression_loss`](file:///d:/AIT2004_7/machinelearning/losses.py#L3-L14) (`mse_loss`), dừng khi loss trung bình mỗi epoch $< 0.002$ (khắt khe hơn ngưỡng $0.02$ của đề bài để đảm bảo qua 100%).

3. **[`DigitClassificationModel`](file:///d:/AIT2004_7/machinelearning/models.py#L112-L156) (Q3 — Phân loại chữ số MNIST):**
   - Kiến trúc MLP: $784 \xrightarrow{\text{Linear}+\text{ReLU}} 256 \xrightarrow{\text{Linear}+\text{ReLU}} 128 \xrightarrow{\text{Linear}} 10\text{ (logits)}$.
   - Tầng cuối trả về logits thô (không qua Softmax) vì hàm [`digitclassifier_loss`](file:///d:/AIT2004_7/machinelearning/losses.py#L18-L32) sử dụng `torch.nn.functional.cross_entropy(y_pred, y)` đã tích hợp sẵn `LogSoftmax + NLLLoss` ổn định số học.
   - Huấn luyện với `batch_size=128, lr=0.001` cho tới khi `dataset.get_validation_accuracy() >= 0.978` (vượt ngưỡng yêu cầu $97\%$).

---

### 5.2. Mạng Hồi quy (RNN) nhận dạng ngôn ngữ — [`LanguageIDModel`](file:///d:/AIT2004_7/machinelearning/models.py#L159-L218) (Q4)

- **Bài toán:** Đầu vào là một từ có độ dài $L$ ký tự bất kỳ, được biểu diễn dưới dạng danh sách `xs` gồm $L$ tensor, mỗi tensor `xs[t]` có kích thước `(batch_size, 47)`. Cần phân loại từ đó thuộc 1 trong 5 ngôn ngữ.
- **Công thức hồi quy:** Ta dùng 2 tầng chiếu tuyến tính `self.x_layer` ($W_x: 47 \to 256$) và `self.h_layer` ($W_h: 256 \to 256$) cùng tầng phân loại đầu ra `self.out_layer` ($W_{\text{out}}: 256 \to 5$):
  $$h_0 = \text{ReLU}(W_x x_0 + b_x)$$
  $$h_t = \text{ReLU}(W_x x_t + W_h h_{t-1} + b), \quad \forall t = 1, \dots, L-1$$
  $$\hat{y} = W_{\text{out}} h_{L-1} + b_{\text{out}}$$
- **Tại sao phải dùng `movedim(batch['x'], 1, 0)` trong [`train_languageid`](file:///d:/AIT2004_7/machinelearning/train.py#L100-L127)?**
  `DataLoader` gom các mẫu theo chiều thứ nhất nên trả về `batch['x']` có kích thước `(batch_size, L, 47)`. Tuy nhiên, hàm `forward(xs)` của RNN duyệt qua từng bước thời gian $t = 0 \dots L-1$, tức cần chiều đầu tiên là $L$: `(L, batch_size, 47)`. Lệnh `movedim(batch['x'], 1, 0)` hoán đổi trục $0$ và trục $1$ trước khi đưa vào `model(x)`.

---

### 5.3. Tích chập 2D Vector hóa — [`Convolve`](file:///d:/AIT2004_7/machinelearning/models.py#L222-L249) & [`DigitConvolutionalModel`](file:///d:/AIT2004_7/machinelearning/models.py#L252-L287) (Q5)

- **Công thức toán học của tích chập 2D hợp lệ (valid cross-correlation):**
  Với ảnh đầu vào $I \in \mathbb{R}^{H \times W}$ và bộ lọc (kernel) $K \in \mathbb{R}^{k_h \times k_w}$, ma trận đầu ra $O$ có kích thước:
  $$H_{\text{out}} = H - k_h + 1, \qquad W_{\text{out}} = W - k_w + 1$$
- **Tại sao không dùng 2 vòng lặp theo từng điểm ảnh $(y, x)$ của ảnh đầu ra?**
  Cách thông thường là lặp `for y in range(out_h): for x in range(out_w):` rồi tính `tensordot`. Cách này tạo ra $H_{\text{out}} \times W_{\text{out}}$ nút tính toán nhỏ lẻ trong đồ thị autograd của PyTorch (trên ảnh $128 \times 128$ là hơn $15{,}000$ lời gọi hàm Python!).
- **Giải pháp Vector hóa theo lát cắt Kernel (Kernel-Slice Vectorization):**
  Thay vì lặp theo tọa độ ảnh $(y, x)$, ta **lặp theo tọa độ của bộ lọc $(i, j) \in [0, k_h) \times [0, k_w)$** (với kernel $3 \times 3$ chỉ có đúng **9 bước lặp** bất kể ảnh to đến đâu!):
  ```python
  out_h = input_tensor_dimensions[0] - weight_dimensions[0] + 1
  out_w = input_tensor_dimensions[1] - weight_dimensions[1] + 1
  Output_Tensor = zeros((out_h, out_w))
  for i in range(weight_dimensions[0]):
      for j in range(weight_dimensions[1]):
          Output_Tensor = Output_Tensor + input[i:i + out_h, j:j + out_w] * weight[i, j]
  ```
  Mỗi bước $(i, j)$ lấy trọn vẹn vùng ảnh dịch chuyển `input[i:i+out_h, j:j+out_w]` nhân vô hướng với trọng số `weight[i, j]` và cộng dồn bằng toán tử C++ vector hóa của PyTorch. Như đã đo trong Thực nghiệm G, kỹ thuật này giúp **tăng tốc tới $2{,}313.8\times$** trên ma trận $128 \times 128$!

---

### 5.4. Cơ chế Tự chú ý Nhân quả — [`Attention`](file:///d:/AIT2004_7/machinelearning/models.py#L290-L332) (Q6)

- **Công thức Causal Scaled Dot-Product Self-Attention:**
  Từ đầu vào $X \in \mathbb{R}^{B \times T \times C}$ ($B$: batch size, $T$: chiều dài chuỗi, $C$: `layer_size`), ta chiếu tuyến tính sang 3 không gian Query ($Q$), Key ($K$), Value ($V$):
  $$Q = \text{q\_layer}(X), \quad K = \text{k\_layer}(X), \quad V = \text{v\_layer}(X) \in \mathbb{R}^{B \times T \times C}$$
  Tính ma trận điểm chú ý chưa chuẩn hóa $M \in \mathbb{R}^{B \times T \times T}$:
  $$M = \frac{Q K^T}{\sqrt{C}} = \text{matmul}(q, \text{movedim}(k, 1, 2)) / (C^{0.5})$$
  (Dùng `movedim(k, 1, 2)` để chuyển vị hai chiều cuối của $K$ từ $(B, T, C)$ thành $(B, C, T)$).
  Áp dụng **mặt nạ nhân quả tam giác dưới (`self.mask`)** để token ở vị trí $t$ không thể "nhìn trộm" các token tương lai $t' > t$:
  $$M_{b, t, t'} \leftarrow -\infty \quad \text{nếu } \text{mask}_{t, t'} = 0$$
  Khi đưa qua `softmax(M, dim=-1)`, các vị trí $-\infty$ sẽ có trọng số chú ý $e^{-\infty} = 0$, và kết quả cuối cùng là `matmul(softmax(M, dim=-1), v)`.

---

## 6. Kịch bản Ma trận lớn & Hệ thống Đo lường Thực nghiệm (`experiments/`)

Để đánh giá các kỹ thuật đã học trên quy mô vượt xa các test nhỏ của autograder mà **không làm bẩn mã nguồn nộp bài**, toàn bộ kịch bản mở rộng được xây dựng độc lập trong thư mục `experiments/`.

### 6.1. Cách thiết kế bộ sinh kịch bản [`scenario.py`](file:///d:/AIT2004_7/experiments/scenario.py)

1. **Sinh mê cung đa đường đi (Braided Maze) & Địa hình đầm lầy ([`generate_braided_maze`](file:///d:/AIT2004_7/experiments/scenario.py#L23-L99)):**
   - **Bước 1 (Khung xương mê cung hoàn hảo):** Sử dụng thuật toán **Randomized Depth-First Search (Recursive Backtracker)** trên các tọa độ lẻ $(2r+1, 2c+1)$ để đục thông một cây khung ngẫu nhiên nối mọi ô của ma trận $N \times N$ ($N \in \{31, 61, 101, 151\}$).
   - **Bước 2 (Tạo đa đường đi với hệ số $\beta = 0.25$):** Nếu giữ nguyên mê cung cây (tree maze), giữa điểm đầu $(1,1)$ và điểm đích $(N-2, N-2)$ chỉ có **duy nhất 1 đường đi đơn** — khi đó mọi thuật toán tìm kiếm (dù là DFS, BFS hay A*) nếu không đi lùi đều buộc phải trả về cùng một đường đi! Để kiểm tra khả năng tối ưu hóa chi phí thực sự, ta đục thêm ngẫu nhiên $25\%$ số bức tường nội bộ để tạo ra hàng nghìn chu trình và đường đi thay thế có độ dài khác nhau.
   - **Bước 3 (Địa hình đầm lầy `~`):** Với xác suất $\rho_{\text{swamp}} = 0.30$, các ô hành lang được đánh dấu là đầm lầy. Trong [`exp_search.py`](file:///d:/AIT2004_7/experiments/exp_search.py#L31-L65), lớp `SwampPositionSearchProblem` kế thừa `PositionSearchProblem` và truyền hàm chi phí `costFn = lambda pos: 5 if pos in swamp_set else 1`.

2. **Sinh đấu trường đối kháng cỡ lớn ([`generate_adversarial_arena`](file:///d:/AIT2004_7/experiments/scenario.py#L131-L202)):**
   - Tạo bản đồ đối xứng 2 trục kích thước $41 \times 21$ ([`arena_41x21_3ghosts.lay`](file:///d:/AIT2004_7/experiments/layouts/arena_41x21_3ghosts.lay)), đặt Pacman ở tâm dưới, 3 Ghost ở khoang trung tâm, 4 viên năng lượng (Capsules) ở 4 góc và ~200 viên thức ăn rải dọc các hành lang có nhiều ngã rẽ để kiểm tra khả năng phối hợp công-thủ của `AlphaBeta` và `Expectimax`.

3. **Sinh mạng lưới MDP ngẫu nhiên cỡ lớn ([`generate_mdp_grid`](file:///d:/AIT2004_7/experiments/scenario.py#L205-L246)):**
   - Tạo lưới $N \times N$ ($N \in \{5, 10, 20, 30\}$) với đích chính $+10$ ở góc Đông-Bắc, các đích phụ $+5, +2$, cùng mạng lưới tường `#` và hố phạt $-10$.
   - Để đảm bảo bài toán MDP luôn khả giải với mọi kích thước $N$, hàm sinh tự động kiểm tra tính liên thông bằng BFS từ ô xuất phát $(0,0)$ tới ô đích $(N-1, N-1)$ trước khi trả về lưới.

---

### 6.2. Cơ chế gắn bộ đếm (Instrumentation) không xâm lấn

Trong 4 kịch bản thực nghiệm, ta đo đạc các chỉ số nội tại của thuật toán bằng kỹ thuật **Wrapping / Monkey-Patching nhẹ trên đối tượng thực thi**:

- **Trong [`exp_search.py`](file:///d:/AIT2004_7/experiments/exp_search.py):**
  - Đếm số nút mở rộng chính xác thông qua thuộc tính `problem._expanded` được tăng tự động mỗi khi `getSuccessors` được gọi.
  - Để tránh trường hợp `UCS` bị treo máy khi bùng nổ tổ hợp ở bài toán Multi-Food ($k \ge 9$), lớp `BoundedFoodSearchProblem` kiểm tra `if self._expanded > max_expansions: raise SearchTimeout()` để dừng an toàn ở ngưỡng $300{,}000$ nút và ghi nhận trạng thái `TIMEOUT`.

- **Trong [`exp_multiagent.py`](file:///d:/AIT2004_7/experiments/exp_multiagent.py):**
  - Để đếm chính xác số nút lá/nút cắt tỉa mà `MinimaxAgent` và `AlphaBetaAgent` duyệt qua ở mỗi nước đi, hàm `count_eval_calls` bọc `agent.evaluationFunction` bằng một hàm closure tăng biến đếm `calls += 1` mỗi khi hàm lượng giá được gọi.
  - Đặc biệt, ở Thực nghiệm D (`run_pruning_benchmark`), tại **cùng một trạng thái bàn cờ trung gian $s_t$**, ta cho cả `Minimax(d)` và `AlphaBeta(d)` cùng tính nước đi độc lập để so sánh trực tiếp trên cùng một gốc cây trò chơi (paired comparison), từ đó kiểm chứng được tỷ lệ khớp hành động **100.0%** và tỷ lệ nhánh cắt tỉa chính xác.

- **Trong [`exp_rl.py`](file:///d:/AIT2004_7/experiments/exp_rl.py):**
  - Trước tiên, chạy `ValueIterationAgent` với $k = 500$ vòng lặp để thu được nghiệm hội tụ chuẩn $V^*(s)$.
  - Với `PrioritizedSweepingValueIterationAgent`, thay vì khởi tạo lại từ đầu ở mỗi mốc kiểm tra, ta chạy một mạch duy nhất từ $1 \to 15{,}000$ bước cập nhật (`run_ps_trajectory`) và ghi lại đường cong sai số cực đại $\varepsilon_k = \max_{s \in S} |V_k(s) - V^*(s)|$ tại mọi mốc điểm.

- **Trong [`exp_ml.py`](file:///d:/AIT2004_7/experiments/exp_ml.py):**
  - Cài đặt đối chứng hàm `naive_convolve_tensordot` (dùng 2 vòng lặp pixel + `tensordot`) để đo trực tiếp hệ số tăng tốc $S_{\text{conv}}$ và sai số tuyệt đối cực đại $\max |O_{\text{vec}} - O_{\text{naive}}|$ so với hàm [`models.Convolve`](file:///d:/AIT2004_7/machinelearning/models.py#L222-L249).
  - Đo thông lượng xử lý (`Tokens/sec`) và dung lượng bộ nhớ ma trận chú ý $B \times T \times T \times 4\text{ bytes}$ của lớp [`models.Attention`](file:///d:/AIT2004_7/machinelearning/models.py#L290-L332) khi quét chiều dài chuỗi $T$ từ $16$ lên $512$.

---

## 7. Hướng dẫn chạy kiểm thử & Tái lập kết quả

Bạn có thể chạy kiểm tra điểm số autograder của cả 4 bài nộp hoặc chạy lại toàn bộ các kịch bản thực nghiệm ma trận lớn bất kỳ lúc nào từ thư mục gốc `d:\AIT2004_7`:

### 7.1. Chạy Autograder chính thức của CS188 (Kiểm tra điểm bài nộp)
```powershell
# Project 1: Search (Kết quả: 26/25)
cd d:\AIT2004_7\search; python autograder.py --no-graphics

# Project 2: Multi-Agent Search (Kết quả: 25/25)
cd d:\AIT2004_7\multiagent; python autograder.py --no-graphics

# Project 3: Reinforcement Learning (Kết quả: 26/25)
cd d:\AIT2004_7\reinforcement; python autograder.py --no-graphics

# Project 5: Machine Learning (Kết quả: 25/25)
cd d:\AIT2004_7\machinelearning; python autograder.py --no-graphics
```

### 7.2. Chạy bộ kịch bản ma trận lớn & Xuất báo cáo thực nghiệm
```powershell
cd d:\AIT2004_7

# 1. Sinh lại toàn bộ mê cung lớn (31x31 -> 151x151) và đấu trường Arena 41x21
python experiments/scenario.py

# 2. Chạy từng thực nghiệm độc lập (tự động xuất file CSV vào experiments/results/)
python experiments/exp_search.py
python experiments/exp_multiagent.py
python experiments/exp_rl.py
python experiments/exp_ml.py

# 3. In bảng tổng hợp toàn bộ kết quả thực nghiệm ra màn hình
python experiments/summarize.py
```

Toàn bộ kết quả định lượng và bảng phân tích từ các lệnh trên được lưu tại:
- Báo cáo phân tích độ đo & kết quả: [`experiments/REPORT.md`](file:///d:/AIT2004_7/experiments/REPORT.md)
- Dữ liệu CSV chi tiết: [`experiments/results/`](file:///d:/AIT2004_7/experiments/results/)
