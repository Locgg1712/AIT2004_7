# Hướng Dẫn Kiến Trúc Mã Nguồn & Giải Thích Kỹ Thuật Chi Tiết (`EXPLANATION_GUIDE.md`)

Tài liệu này giải thích chi tiết cấu trúc mã nguồn, các hàm và lớp thực tế được cài đặt trong 4 Project gốc (`search`, `multiagent`, `reinforcement`, `machinelearning`) cùng toàn bộ khung thực nghiệm mở rộng trong [`experiments/`](./). Để xem báo cáo kết quả định lượng đầy đủ (bảng số liệu `Mean ± std`, khoảng tin cậy `Wilson 95% CI`, hệ số lũy thừa $\alpha$ và 11 biểu đồ phân tích), vui lòng xem **[`REPORT.md`](./REPORT.md)**.

---

## 1. Sơ đồ Kiến trúc Tệp tin & Ánh xạ Hàm Thực tế

Để đảm bảo tính nguyên vẹn 100% với bộ chấm điểm tự động `autograder.py` của UC Berkeley CS188, **10 tệp bài nộp chuẩn** được giữ nguyên giao diện gốc, và toàn bộ các kịch bản mở rộng quy mô lớn được đóng gói độc lập trong thư mục [`experiments/`](./) thông qua cơ chế kế thừa và monkey-patching không xâm lấn:

```text
AIT2004_7/
├── search/                          # Project 1 (26/25 điểm)
│   ├── search.py                    # _graphSearch, depthFirstSearch, breadthFirstSearch, uniformCostSearch, aStarSearch
│   └── searchAgents.py              # CornersProblem, cornersHeuristic, foodHeuristic (5/4đ), ClosestDotSearchAgent
├── multiagent/                      # Project 2 (25/25 điểm)
│   └── multiAgents.py               # ReflexAgent, MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent, betterEvaluationFunction
├── reinforcement/                   # Project 3 (26/25 điểm)
│   ├── valueIterationAgents.py      # ValueIterationAgent, PrioritizedSweepingValueIterationAgent (5/4đ)
│   ├── qlearningAgents.py           # QLearningAgent, PacmanQAgent, ApproximateQAgent
│   └── analysis.py                  # Tham số tối ưu MDP cho Q2 (a..e), Q3 (a..e) và Q7 ('NOT POSSIBLE')
├── machinelearning/                 # Project 5 (25/25 điểm)
│   ├── models.py                    # PerceptronModel, RegressionModel, DigitClassificationModel, LanguageIDModel, Convolve, Attention
│   └── train.py                     # Vòng lặp huấn luyện train_perceptron, train_regression, train_digitclassifier, ...
└── experiments/                     # Khung Thực nghiệm Mở rộng Quy mô Lớn & Kịch bản Đô thị Hợp nhất Mega-City
    ├── scenario.py                  # Bộ sinh bản đồ: make_search_maze, make_classic_arena, make_gridworld, make_megacity
    ├── exp_search.py                # Thí nghiệm A (SwampProblem, WA*) & Thí nghiệm B (Precomputed BFS + Prim's MST)
    ├── exp_multiagent.py            # Thí nghiệm C & D (_counting_gen, get_wall_dist_map, mazeBetterEvaluationFunction, OrderedAlphaBetaAgent)
    ├── exp_rl.py                    # Thí nghiệm E (Backups + Q-evals) & Thí nghiệm F (MegaCityExtractor + Zero-Shot Transfer)
    ├── exp_ml.py                    # Thí nghiệm G (Convolve vs F.conv2d), H (Attention Memory), H2 (CharAttentionLM), I (ImitationMLP/CNN)
    ├── summarize.py                 # Trích xuất thống kê tự động (Mean ± std, Wilson 95% CI, hồi quy Log-Log) từ CSV
    ├── plot_all_figures.py          # Vẽ tự động 11 biểu đồ PNG từ dữ liệu CSV
    ├── layouts/                     # Các tệp bản đồ .lay sinh tự động (megacity_31x21.lay, maze_151x151.lay, arena_41x21_3ghosts.lay, ...)
    ├── results/                     # 9 tệp dữ liệu thực nghiệm CSV
    └── figures/                     # 11 biểu đồ trực quan PNG
```

---

## 2. Giải thích Chi tiết Bộ Sinh Kịch bản ([`scenario.py`](./scenario.py))

Tệp [`scenario.py`](./scenario.py) cung cấp 4 hàm sinh môi trường có kiểm soát hạt giống ngẫu nhiên (`random.Random(seed)`):

### 2.1. `make_megacity(mode="arena", seed=1)` — Đô thị Hợp nhất $31\times 21$
- Sinh lưới đô thị kích thước $31\times 21$ ($651$ ô, gồm $327$ ô đường đi tự do), xuất phát từ mê cung đệ quy $15\times 10$ ô logic được đục thông $35\%$ ngõ cụt (`braid = 0.35`) và mở rộng đại lộ trung tâm để tạo cấu trúc các tòa nhà phố (city blocks).
- Hỗ trợ 3 chế độ (`mode`) trên cùng một cấu trúc tường đô thị:
  - `mode="route"`: Đặt Pacman `P` tại góc Đông-Bắc `(29, 19)`, đích `.` tại góc Tây-Nam `(1, 1)` và rải $25\%$ ô đầm lầy (`~`, chi phí bước $= 5$).
  - `mode="food"`: Rải ngẫu nhiên $k$ viên thức ăn trên đô thị để kiểm thử bài toán `FoodSearchProblem`.
  - `mode="arena"` (Lưu tại [`layouts/megacity_31x21.lay`](./layouts/megacity_31x21.lay)): Đặt Pacman tại trung tâm `(15, 10)`, **3 Ghosts (`G`)**, **4 viên năng lượng (`o`)** ở 4 góc đô thị, và rải thức ăn trên toàn bộ các tuyến phố.

### 2.2. `make_search_maze(cells_w, cells_h, seed=1, braid=0.25, swamp_prob=0.25)`
- Sinh mê cung kích thước thực $(2 \cdot \text{cells\_w} + 1) \times (2 \cdot \text{cells\_h} + 1)$ (từ $31\times 31$ đến $151\times 151$) bằng thuật toán DFS lặp có ngăn xếp.
- Tham số `braid=0.25` đục ngẫu nhiên $25\%$ tường của các ô ngõ cụt (có 3 mặt tường) để tạo **đa lộ trình (multiple cycles)** giữa điểm xuất phát `(W-2, H-2)` và đích `(1, 1)`. Nếu không đục thông (`braid=0`), mê cung là cây khung duy nhất nên mọi thuật toán đều buộc phải đi cùng một đường; nhờ `braid=0.25`, ta đo được chính xác độ lệch chi phí của `DFS` và `BFS` (trên đầm lầy) so với đường đi tối ưu của `UCS`/`A*`.
- Tham số `swamp_prob=0.25` gán ký tự `'~'` cho $25\%$ các ô hành lang (trừ ô đầu và ô đích).

### 2.3. `make_classic_arena(half_w=20, half_h=10, num_ghosts=3, seed=1)`
- Sinh bản đồ đấu trường kích thước $(2 \cdot \text{half\_w} + 1) \times (2 \cdot \text{half\_h} + 1) = 41\times 21$ ($861$ ô, gấp $3.91\times$ diện tích `mediumClassic` $20\times 11 = 220$ ô).
- Đặt các khối tường chữ nhật ngẫu nhiên bên trong (không đối xứng gương), sau đó chạy BFS từ vị trí Pacman ở tâm bản đồ và tự động mở thông mọi ô bị cô lập để đảm bảo đồ thị liên thông $100\%$. Đặt $4$ viên năng lượng `o` ở 4 góc và `num_ghosts = 3` ma `G` quanh tâm.

### 2.4. `make_gridworld(w, h, seed=1, wall_prob=0.25, pit_prob=0.08)`
- Sinh lưới MDP kích thước $w \times h$ ($10\times 10, 20\times 20, 30\times 30, 40\times 40$) tương thích trực tiếp với `gridworld.Gridworld` của Project 3:
  - Ô đích chính thưởng **`+10`** tại góc trên phải `(w-1, h-1)` và lối thoát phụ thưởng **`+1`** tại góc trên trái `(0, h-1)`.
  - Ô xuất phát `'S'` tại `(0, 0)` cùng đường hành lang bảo đảm kết nối từ `'S'` tới đích `+10`.
  - Các ô còn lại có xác suất $8\%$ là hố bẫy phạt **`-10`** và $25\%$ là tường chắn `'#'`.

---

## 3. Giải thích Kỹ thuật Các Module Thực nghiệm (`exp_*.py`)

### 3.1. [`exp_search.py`](./exp_search.py) — Tìm kiếm Đầm lầy, Weighted A\* và Precomputed MST
1. **Lớp `SwampProblem(searchAgents.PositionSearchProblem)`:**
   - Đọc trực tiếp các ký tự `'~'` từ `layoutText` đưa vào tập hợp `self.swamp`.
   - Ghi đè phương thức `costFn(pos)`: trả về `5` nếu `pos in self.swamp`, ngược lại trả về `1`. Vì $\min \text{cost} = 1$, khoảng cách Manhattan và Euclid vẫn hoàn toàn **admissible** ($h(n) \le h^*(n)$) trên địa hình đầm lầy.
2. **Hàm `weighted_astar(problem, heuristic, w=2.0)`:**
   - Sử dụng `util.PriorityQueue` với độ ưu tiên $f_w(n) = g(n) + w \cdot h(n)$, cho phép đánh đổi có kiểm soát (bounded suboptimality $\le w \cdot C^*$) để giảm số nút mở rộng từ bậc hai $O(N^2)$ xuống gần tuyến tính $O(N^{1.01})$.
3. **Hàm `precompute_food_dists(problem)` & `make_mst_food_heuristic(dist_from_food)`:**
   - Thay vì gọi `mazeDistance` lặp lại hàng nghìn lần bên trong `foodHeuristic`, hàm `precompute_food_dists` chạy BFS một lần duy nhất từ từng viên thức ăn ban đầu ($k \le 11$ lần BFS, chỉ mất $4\text{–}35\text{ ms}$) để lưu bảng tra cứu khoảng cách $O(1)$ từ mọi viên thức ăn tới mọi ô trống trên bản đồ.
   - Hàm `make_mst_food_heuristic` tính tổng trọng số **Cây khung nhỏ nhất (thuật toán Prim)** trên tập các viên thức ăn còn lại $F$ cộng với khoảng cách mê cung từ Pacman tới viên thức ăn gần nhất:
     $$h_{\text{MST}}(\text{pos}, F) = \min_{u \in F} d_{\text{maze}}(\text{pos}, u) + \text{MST\_Weight}(F, d_{\text{maze}})$$
     Nhờ cận dưới chặt của MST, A\* giải được $100\%$ các mẫu ở $k=11$ chỉ với trung bình $694.7$ nút mở rộng.

### 3.2. [`exp_multiagent.py`](./exp_multiagent.py) — Khoảng cách Mê cung $O(1)$, Giá trị Hữu hạn & Move Ordering
1. **Đếm nút không xâm lấn (`_counting_gen`):**
   - Bọc phương thức `pacman.GameState.generateSuccessor` để tăng bộ đếm `NODES[0] += 1` mỗi khi một trạng thái con được sinh ra trong cây tìm kiếm.
2. **Bộ nhớ đệm khoảng cách mê cung $O(1)$ (`get_wall_dist_map(walls)`):**
   - Tính trước khoảng cách BFS giữa mọi cặp ô trống trên lưới tường `walls` và gắn trực tiếp từ điển kết quả vào thuộc tính `walls._dist_map` (đồng thời lưu trong `_MAZE_DIST_CACHE[(walls.width, walls.height)]`). Mọi truy vấn khoảng cách trong hàm lượng giá chỉ tốn $O(1)$ tra cứu từ điển.
3. **Hàm `mazeBetterEvaluationFunction(currentGameState)`:**
   - Trả về giá trị kết thúc **hữu hạn** (`+10000.0 + score` khi thắng, `-10000.0 + score` khi thua) thay vì `±inf`, giúp `ExpectimaxAgent` tính đúng kỳ vọng toán học khi có nhánh tử vong xác suất nhỏ.
   - Sử dụng khoảng cách mê cung thực từ `get_wall_dist_map` tới thức ăn, viên năng lượng và các Ghost, triệt tiêu hoàn toàn hiện tượng Pacman bị kẹt dao động sau tường trên `Arena 41x21` và `MegaCity 31x21`.
4. **Lớp `OrderedAlphaBetaAgent`:**
   - Tại nút gốc và các nút `MAX` ở tầng đầu, sắp xếp danh sách trạng thái kế tiếp theo thứ tự giảm dần của `self.evaluationFunction(state)` trước khi gọi đệ quy Alpha-Beta, giúp đẩy ngưỡng cận dưới $\alpha$ lên cao ngay từ nhánh đầu tiên và tăng tỷ lệ cắt tỉa lên $56.8\%$ ở $d=3$.

### 3.3. [`exp_rl.py`](./exp_rl.py) — Đo Kép MDP (Backups & Q-Evals) và Chuyển giao Zero-Shot
1. **Hàm `run_mdp_convergence(size, seed, tol=1e-3)`:**
   - Đo độc lập 2 đại lượng công việc cho cả `ValueIteration` và `PrioritizedSweeping`:
     - `backups_to_tol`: Số lần giá trị trạng thái $V(s)$ được cập nhật.
     - `q_evals_to_tol`: Tổng số lần phương thức `computeQValueFromValues(s, a)` được gọi (bao gồm cả vòng lặp kiểm tra tiền nhiệm $p \in \text{Pred}(s)$ của Prioritized Sweeping).
   - Bấm giờ `pure_ms` tách biệt hoàn toàn khỏi hàm kiểm tra hội tụ `max_err_vs_ref` và `metrics_vs_ref`.
2. **Lớp `MegaCityExtractor(featureExtractors.FeatureExtractor)` & Chuyển giao `MegaCity-Transfer`:**
   - Trích xuất 7 đặc trưng chuẩn hóa cho bài toán đa tác tử có Capsule.
   - Trong `run_pacman_rl`, tác tử sau khi huấn luyện trên `mediumClassic` được đóng băng trọng số (`epsilon = 0.0, alpha = 0.0`) và kiểm tra trên cả `mediumClassic` (in-domain) lẫn `MegaCity 31x21` (zero-shot transfer).

### 3.4. [`exp_ml.py`](./exp_ml.py) — Tích chập `F.conv2d`, Bộ nhớ Autograd, Char-LM và Imitation Learning
1. **Thí nghiệm G (`experiment_g`):** So sánh `naive_convolve`, `models.Convolve` và `F.conv2d` trên 1 luồng CPU duy nhất với 3 lần warm-up và lấy trung vị 15 lần chạy.
2. **Thí nghiệm H (`measure_attention_memory_kb` & `experiment_h`):** Tính chính xác tổng dung lượng byte của các tensor kích hoạt trung gian ($Q, K, V, M_{\text{raw}}, M_{\text{masked}}, W_{\text{attn}}, \text{Out}$) lưu trong đồ thị Autograd cho phép lan truyền ngược.
3. **Thí nghiệm H2 (`CharAttentionLM` vs `CharBigramMLP`):** Huấn luyện mô hình ngôn ngữ ký tự sử dụng trực tiếp lớp `models.Attention` (Project 5 Q6) để kiểm chứng khả năng giảm Perplexity nhờ cơ chế chú ý nhân quả.
4. **Thí nghiệm I (`ImitationMLP` & `ImitationCNN`):** Thu thập quỹ đạo chuyên gia từ `AlphaBeta-d2 (mazeEval)` trên `MegaCity`, huấn luyện mạng MLP và mạng CNN (dùng trực tiếp `models.Convolve` từ Project 5 Q5 trên cửa sổ $7\times 7\times 2$ quanh Pacman) để đánh giá tốc độ suy luận và hiện tượng lệch phân phối của Behavioral Cloning.

---

## 4. Hướng dẫn Tự Chạy Trực quan Hóa (GUI) cho Từng Project

> **Lưu ý quan trọng về thư mục làm việc (`Working Directory`):**  
> Hàm `loadAgent` trong `pacman.py` của CS188 chỉ quét các tệp `*Agents.py` tại thư mục hiện hành (`.` / `os.getcwd()`). Do đó, trước khi chạy `pacman.py` hoặc `gridworld.py`, bạn **cần `cd` vào đúng thư mục của từng Project** như các lệnh dưới đây:

### 4.1. Xem Trực quan Project 1 (`search`)
```powershell
cd D:\AIT2004_7\search

# 1. Xem A* tìm đường trên mê cung mở rộng 61x61
python pacman.py -l ../experiments/layouts/maze_61x61.lay -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic --frameTime 0.01 --zoom 0.7

# 2. Xem Pacman tìm đường trên đô thị hợp nhất Mega-City 31x21
python pacman.py -l ../experiments/layouts/megacity_31x21.lay -p ClosestDotSearchAgent --frameTime 0.03

# 3. Xem A* + foodHeuristic giải bài toán trickySearch (6.690 nút mở rộng - đạt 5/4 điểm)
python pacman.py -l trickySearch -p AStarFoodSearchAgent --frameTime 0.02
```

### 4.2. Xem Trực quan Project 2 (`multiagent`)
```powershell
cd D:\AIT2004_7\multiagent

# 1. Xem Expectimax (depth=2) đối đầu 3 Ghosts trên Đấu trường Arena 41x21
python pacman.py -p ExpectimaxAgent -a evalFn=better,depth=2 -l ../experiments/layouts/arena_41x21_3ghosts.lay -g DirectionalGhost -k 3 --frameTime 0.03

# 2. Xem AlphaBeta (depth=2) thi đấu trên đô thị Mega-City 31x21 (3 Ghosts)
python pacman.py -p AlphaBetaAgent -a evalFn=better,depth=2 -l ../experiments/layouts/megacity_31x21.lay -g RandomGhost -k 3 --frameTime 0.03
```

### 4.3. Xem Trực quan Project 3 (`reinforcement`)
```powershell
cd D:\AIT2004_7\reinforcement

# 1. Xem Prioritized Sweeping Value Iteration trên DiscountGrid
python gridworld.py -a priosweepvalue -i 100 -g DiscountGrid

# 2. Xem Approximate Q-Learning (SimpleExtractor) huấn luyện 50 ván rồi tự chơi trên mediumClassic
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 55 -l mediumClassic

# 3. Xem Approximate Q-Learning huấn luyện nhanh và chơi trên đô thị Mega-City 31x21
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 53 -l ../experiments/layouts/megacity_31x21.lay -k 3
```

### 4.4. Xem Trực quan Project 5 (`machinelearning`)
```powershell
cd D:\AIT2004_7\machinelearning

# Chạy chấm điểm kèm đồ thị huấn luyện trực tiếp (Perceptron, Sin Regression, Digit Classification, LanguageID, CNN, Attention)
python autograder.py
```
