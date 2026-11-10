# Báo cáo Đánh giá Thực nghiệm: Kịch bản "Thành phố Mê cung" (Pacman Mega-City) với Kích thước Ma trận Lớn

Tài liệu này mô tả kịch bản mở rộng quy mô lớn đã được dựng trong thư mục `experiments/`, hệ thống độ đo (metrics) định lượng và kết quả thực nghiệm đánh giá các kỹ thuật từ **Project 1 (Search)**, **Project 2 (Multi-Agent Search)**, **Project 3 (Reinforcement Learning)** và **Project 5 (Machine Learning)**.

---

## 1. Thiết kế Kịch bản Mở rộng (Scenario Design)

Thay vì các bản đồ nhỏ mặc định của đồ án (`tinyMaze` $7\times 7$, `mediumClassic` $20\times 11$, `DiscountGrid` $5\times 5$, ảnh $28\times 28$), kịch bản **Pacman Mega-City** ([scenario.py](file:///d:/AIT2004_7/experiments/scenario.py)) mở rộng không gian trạng thái và kích thước ma trận trên cả 4 bài toán:

1. **Kịch bản Tìm đường trên Mê cung Đa địa hình ([exp_search.py](file:///d:/AIT2004_7/experiments/exp_search.py))**:
   - **Thí nghiệm A (Mở rộng kích thước lưới & chi phí không đồng nhất)**: Mê cung có vòng lặp (braided maze) tăng kích thước từ $31\times 31 \to 61\times 61 \to 101\times 101 \to 151\times 151$ (tới $22.801$ ô). Bổ sung địa hình **đầm lầy (`~`)** chiếm $25\%$ diện tích với chi phí di chuyển $c = 5$ (so với đường thường $c = 1$).
   - **Thí nghiệm B (Bài toán thu thập $k$ điểm thức ăn - NP-hard)**: Trên mê cung $31\times 31$, tăng số điểm thức ăn $k \in \{3, 5, 7, 9, 11\}$ (không gian trạng thái $|V| \cdot 2^k$), giới hạn ngân sách $300.000$ nút mở rộng hoặc $60$ giây.

2. **Kịch bản Đối kháng Đa tác tử trên Đấu trường Lớn ([exp_multiagent.py](file:///d:/AIT2004_7/experiments/exp_multiagent.py))**:
   - **Thí nghiệm C (Thi đấu đối kháng)**: So sánh bản đồ gốc `mediumClassic` ($20\times 11$, 2 ma) với đấu trường mới **`Arena 41x21`** ($41\times 21$, 3 ma, 4 viên năng lượng) trước 2 loại ma: `RandomGhost` (ngẫu nhiên) và `DirectionalGhost` (săn đuổi chủ động).
   - **Thí nghiệm D (Hiệu quả cắt tỉa Alpha-Beta)**: Đo trực tiếp trên các trạng thái thực tế lấy mẫu từ ván đấu với độ sâu $d = 2$ và $d = 3$.

3. **Kịch bản Quy hoạch MDP & Học tăng cường Không gian Lớn ([exp_rl.py](file:///d:/AIT2004_7/experiments/exp_rl.py))**:
   - **Thí nghiệm E (MDP Gridworld kích thước lớn)**: Mở rộng từ `DiscountGrid` $5\times 5$ (22 trạng thái) lên Gridworld ngẫu nhiên liên thông $10\times 10$, $20\times 20$ và $30\times 30$ (tới ~747 trạng thái, có tường `#`, hố bẫy `-10`, lối thoát gần `+1` và lối thoát xa `+10`, nhiễu chuyển trạng thái `noise = 0.2`).
   - **Thí nghiệm F (Khái quát hóa trên Pacman)**: So sánh `Tabular Q-Learning` và `Approximate Q-Learning` (`SimpleExtractor`) từ `smallGrid` ($7\times 7$) lên `mediumClassic` ($20\times 11$, không gian trạng thái $\sim 10^{30}$).

4. **Kịch bản Mở rộng Ma trận Học sâu ([exp_ml.py](file:///d:/AIT2004_7/experiments/exp_ml.py))**:
   - **Thí nghiệm G (Tích chập 2D - `Convolve`)**: Tăng kích thước ma trận đầu vào từ $28\times 28 \to 64\times 64 \to 128\times 128 \to 256\times 256$ với các kích thước nhân (kernel) $3\times 3, 5\times 5, 7\times 7$.
   - **Thí nghiệm H (Causal Self-Attention)**: Tăng chiều dài chuỗi $T \in \{16, 64, 128, 256, 512\}$ (ma trận chú ý $T\times T$) và chiều nhúng $C \in \{64, 128, 256\}$.

---

## 2. Hệ thống Độ đo Đánh giá (Evaluation Metrics)

| Nhóm kỹ thuật | Độ đo (Metric) | Công thức / Ý nghĩa |
| :--- | :--- | :--- |
| **1. Tìm kiếm (Search)** | **Tỷ lệ tối ưu ($\text{OptRatio}$)** | $\frac{C(\text{alg})}{C^*}$ — bằng $1.00$ nếu thuật toán tìm được đường đi có tổng chi phí nhỏ nhất. |
| | **Độ phủ không gian ($\text{Coverage}$)** | $\frac{\text{Expanded}}{|V_{\text{open}}|}$ — tỷ lệ ô trống trong mê cung bị thuật toán mở rộng. |
| | **Độ xuyên thấu ($\text{Penetrance}$)** | $\frac{L}{N_{\text{expanded}}}$ — mức độ tập trung hướng đích của tìm kiếm. |
| | **Đỉnh bộ nhớ ($\text{PeakFrontier}$)** | Số phần tử cực đại lưu trong `Stack`/`Queue`/`PriorityQueue` suốt quá trình chạy. |
| **2. Đối kháng (Multi-Agent)** | **Tỷ lệ thắng & Tiến độ ăn hạt** | $\text{WinRate}$ (%) và $\text{FoodCleared} = 1 - \frac{\text{Food}_{\text{remain}}}{\text{Food}_{\text{total}}}$. |
| | **Tỷ lệ cắt tỉa ($\text{PruningRatio}$)** | $1 - \frac{\text{Nodes}(\text{AlphaBeta})}{\text{Nodes}(\text{Minimax})}$ và **Tăng tốc ($\text{Speedup}$)** $= \frac{t_{\text{MM}}}{t_{\text{AB}}}$. |
| | **Độ bảo toàn quyết định** | $\text{ActionMatch}$ (%) — tỷ lệ Alpha-Beta trả về đúng hành động của Minimax. |
| **3. Học tăng cường (RL)** | **Số phép cập nhật hội tụ** | Số phép cập nhật Bellman (backups) cần thiết để $\max_s |V(s) - V^*(s)| < 10^{-3}$. |
| | **Hiệu quả mẫu & Nén tham số** | $\text{WinRate}$, $\text{AvgScore}$ theo số tập huấn luyện $N_{\text{train}}$ và số tham số mô hình $|\theta|$. |
| **4. Học máy (ML)** | **Hệ số tăng tốc ($\text{Speedup}$)** | $\frac{t_{\text{naive}}}{t_{\text{vectorized}}}$, thông lượng ($\text{tokens/s}$) và dung lượng bộ nhớ ma trận $O(T^2)$. |

---

## 3. Kết quả Thực nghiệm & Phân tích

### 3.1. Project 1 — Tìm kiếm trên Mê cung Lớn & Nhiều điểm thức ăn

#### A. Mở rộng kích thước mê cung từ $31\times 31$ đến $151\times 151$ (trung bình qua 5 hạt giống ngẫu nhiên)

| Kích thước | Địa hình | Thuật toán | $\text{OptRatio}$ | Số nút mở rộng (`Expanded`) | Độ phủ (`Coverage`) | Đỉnh Frontier | Thời gian (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$61\times 61$** | Đồng nhất (`unit`) | `DFS` | $3.15\times$ | $1.458,8$ | $77,3\%$ | $86,6$ | $9,45$ |
| | | `BFS` / `UCS` | **$1.00\times$** | $1.786,8$ | $94,6\%$ | $27,2$ | $11,23$ |
| | | `A*-Manhattan` | **$1.00\times$** | **$1.135,4$** | **$60,1\%$** | $34,0$ | **$8,89$** |
| **$61\times 61$** | Đầm lầy (`swamp`) | `DFS` | $4.06\times$ | $1.458,8$ | $77,3\%$ | $86,6$ | $8,12$ |
| | | `BFS` | $1.17\times$ | $1.786,8$ | $94,6\%$ | $27,2$ | $9,54$ |
| | | `UCS` | **$1.00\times$** | $1.750,8$ | $92,7\%$ | $32,8$ | $13,97$ |
| | | `A*-Manhattan` | **$1.00\times$** | **$1.504,4$** | **$79,7\%$** | $40,6$ | $15,76$ |
| **$151\times 151$** | Đồng nhất (`unit`) | `DFS` | $6.08\times$ | $7.408,6$ | $62,5\%$ | $428,2$ | $52,58$ |
| | | `BFS` / `UCS` | **$1.00\times$** | $11.815,2$ | $99,6\%$ | $78,2$ | $51,02$ |
| | | `A*-Euclid` | **$1.00\times$** | $10.653,2$ | $89,8\%$ | $104,2$ | $58,88$ |
| | | `A*-Manhattan` | **$1.00\times$** | **$9.597,6$** | **$81,0\%$** | $164,4$ | $53,41$ |
| **$151\times 151$** | Đầm lầy (`swamp`) | `DFS` | $8.58\times$ | $7.408,6$ | $62,5\%$ | $428,2$ | $52,50$ |
| | | `BFS` | $1.23\times$ | $11.815,2$ | $99,6\%$ | $78,2$ | $51,08$ |
| | | `UCS` | **$1.00\times$** | $11.629,0$ | $98,1\%$ | $98,2$ | $62,42$ |
| | | `A*-Manhattan` | **$1.00\times$** | **$10.205,6$** | **$86,1\%$** | $142,4$ | $57,03$ |

#### B. Bài toán thu thập $k$ điểm thức ăn trên mê cung $31\times 31$

| Số thức ăn $k$ | `UCS` (Tối ưu, không heuristic) | `A* + foodHeuristic` (Tối ưu, có heuristic) | `ClosestDot` (Tham lam) |
| :---: | :--- | :--- | :--- |
| **$k = 3$** | Giải $3/3$, TB $1.769$ nút ($0,9$s) | Giải $3/3$, TB **$573$ nút** (giảm $3,1\times$) | Giải $3/3$, $\text{OptRatio}=1,00$, $0,05$s |
| **$k = 5$** | Giải $3/3$, TB $5.675$ nút ($3,1$s) | Giải $3/3$, TB **$1.986$ nút** (giảm $2,9\times$) | Giải $3/3$, $\text{OptRatio}=1,00$, $0,06$s |
| **$k = 7$** | Giải $3/3$, TB $29.223$ nút ($17,5$s) | Giải $3/3$, TB **$7.085$ nút** (giảm **$4,1\times$**) | Giải $3/3$, $\text{OptRatio}=1,10$, $0,10$s |
| **$k = 9$** | **Quá ngân sách $0/3$** ($>300.000$ nút) | **Giải $2/3$**, TB $15.668$ nút ($23,5$s) | Giải $3/3$, $\text{OptRatio}=1,01$, $0,13$s |
| **$k = 11$** | **Quá ngân sách $0/3$** ($>300.000$ nút) | **Giải $1/3$**, $29.478$ nút ($19,8$s) | Giải $3/3$, $\text{OptRatio}=1,05$, $0,15$s |

**Nhận xét kỹ thuật (Project 1)**:
- Khi kích thước lưới tăng lên $151\times 151$ và có nhiều vòng lặp, `DFS` trả về đường đi dài gấp **$6,08\times$** (đồng nhất) đến **$8,58\times$** (đầm lầy) so với tối ưu.
- Khi có địa hình đầm lầy ($c=5$), `BFS` không còn tối ưu ($\text{OptRatio} = 1,23\times$ vì đi xuyên qua đầm lầy), trong khi `UCS` và `A*` vẫn đảm bảo $\text{OptRatio} = 1,00\times$ nhờ xét chi phí tích lũy $g(n)$.
- `A*-Manhattan` luôn mở rộng ít nút hơn `A*-Euclid` và `UCS` vì khoảng cách Manhattan chặt hơn (dominates) khoảng cách Euclid trên lưới 4 hướng.
- Với bài toán nhiều mục tiêu ($k$ thức ăn), `UCS` bùng nổ tổ hợp và thất bại hoàn toàn ở $k \ge 9$. `A* + foodHeuristic` giảm hơn $4\times$ số nút mở rộng và giải được tới $k=11$, còn `ClosestDot` chạy tức thời ($<0,2$s) với sai lệch chi phí chỉ $1\%\text{–}10\%$.

---

### 3.2. Project 2 — Tìm kiếm Đối kháng Đa tác tử

#### C. Kết quả thi đấu trên `mediumClassic` ($20\times 11$, 2 ma) vs. `Arena 41x21` (3 ma)

| Bản đồ | Loại Ma | Tác tử | Tỷ lệ thắng | Điểm TB | Thức ăn đã dọn | Nút / nước | ms / nước |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`mediumClassic`** | `Random` | `AlphaBeta-d2 (scoreEval)` | $0\%$ | $200,6$ | $63,9\%$ | $77,6$ | $21,4$ |
| ($20\times 11$, 2 ma) | | `Reflex` | **$100\%$** | $1578,6$ | $100,0\%$ | $3,2$ | $1,2$ |
| | | `AlphaBeta-d2 (betterEval)` | $80\%$ | **$1681,8$** | $99,8\%$ | $73,8$ | $21,7$ |
| | | `Expectimax-d2 (betterEval)` | $60\%$ | $1427,0$ | $95,3\%$ | $94,0$ | $25,2$ |
| **`mediumClassic`** | `Directional` | `AlphaBeta-d2 (scoreEval)` | $0\%$ | $263,0$ | $59,2\%$ | $76,3$ | $18,2$ |
| ($20\times 11$, 2 ma) | | `Reflex` | $20\%$ | $731,0$ | $75,5\%$ | $3,3$ | $1,3$ |
| | | `AlphaBeta-d2 (betterEval)` | **$40\%$** | **$1193,6$** | $85,6\%$ | $74,1$ | $21,2$ |
| | | `Expectimax-d2 (betterEval)` | **$40\%$** | $1161,2$ | **$86,4\%$** | $112,2$ | $31,0$ |
| **`Arena 41x21`** | `Directional` | `AlphaBeta-d2 (scoreEval)` | $0\%$ | $-85,4$ | $21,7\%$ | $106,4$ | $106,7$ |
| ($41\times 21$, 3 ma) | | `AlphaBeta-d2 (betterEval)` | $0\%$ | $-5,4$ | $22,3\%$ | $99,7$ | $105,7$ |
| | | `Reflex` | $0\%$ | $114,2$ | $35,0\%$ | $3,2$ | $4,5$ |
| | | `Expectimax-d2 (betterEval)` | $0\%$ | **$551,4$** | **$38,8\%$** | $149,5$ | $132,2$ |

#### D. Hiệu quả cắt tỉa Alpha-Beta so với Minimax

| Bản đồ & Cấu hình | Độ sâu $d$ | Nút Minimax | Nút Alpha-Beta | Tỷ lệ cắt tỉa | Tăng tốc | Khớp hành động |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `mediumClassic` (2 ma, 3 tầng/ply) | $d = 2$ | $89,0$ | $63,9$ | **$21,3\%$** | $1,36\times$ | **$100,0\%$** |
| `mediumClassic` (2 ma, 3 tầng/ply) | $d = 3$ | $642,1$ | $319,7$ | **$38,0\%$** | **$2,08\times$** | **$100,0\%$** |
| `Arena 41x21` (3 ma, 4 tầng/ply) | $d = 2$ | $164,4$ | $94,3$ | **$23,5\%$** | **$1,58\times$** | **$100,0\%$** |

**Nhận xét kỹ thuật (Project 2)**:
- **Vai trò của hàm lượng giá**: `AlphaBeta-d2` dùng `scoreEvaluationFunction` đạt $0\%$ thắng trên mọi bản đồ vì tầm nhìn $d=2$ không thấy được thức ăn ở xa. Khi thay bằng `betterEvaluationFunction`, điểm trung bình trên `mediumClassic` tăng vọt từ $200,6 \to 1681,8$ (Random) và $263,0 \to 1193,6$ (Directional).
- **Minimax/Alpha-Beta vs. Expectimax trên bản đồ lớn 3 ma**: Trên `Arena 41x21` với 3 ma `DirectionalGhost`, cây tìm kiếm độ sâu $d=2$ có $1 + 3 = 4$ tầng/ply. `AlphaBeta` giả định cả 3 ma phối hợp tối ưu tuyệt đối để vây bắt nên trở nên quá bi quan (bị kẹt hoặc đứng yên, điểm $-5,4$), trong khi `Expectimax` mô hình hóa xác suất hợp lý hơn và đạt điểm cao nhất (**$551,4$ điểm**, dọn **$38,8\%$** thức ăn).
- **Cắt tỉa Alpha-Beta**: Giữ nguyên $100\%$ quyết định của Minimax nhưng cắt bỏ $38,0\%$ số nút ở $d=3$ (tăng tốc $2,08\times$).

---

### 3.3. Project 3 — Quy hoạch Động MDP & Học Tăng cường

#### E. Value Iteration vs. Prioritized Sweeping khi mở rộng Gridworld ($5\times 5 \to 30\times 30$)

| Kích thước Gridworld | Số trạng thái $\|S\|$ | Phép cập nhật `ValueIteration` | Phép cập nhật `PrioritizedSweeping` | Hệ số giảm phép cập nhật Bellman |
| :--- | :---: | :---: | :---: | :---: |
| **$5\times 5$ (DiscountGrid)** | $22$ | $726$ ($33,0$ vòng) | **$200$** ($9,1$ vòng) | **Giảm $3,6\times$** |
| **$10\times 10$ (TB 3 seed)** | $78,3$ | $3.717$ ($48,0$ vòng) | **$1.075$** ($13,8$ vòng) | **Giảm $3,5\times$** |
| **$20\times 20$ (TB 3 seed)** | $331,3$ | $20.325$ ($61,3$ vòng) | **$4.383$** ($13,2$ vòng) | **Giảm $4,6\times$** |
| **$30\times 30$ (TB 3 seed)** | $745,0$ | $50.166$ ($67,3$ vòng) | **$11.900$** ($16,0$ vòng) | **Giảm $4,2\times$** |

#### F. Tabular Q-Learning vs. Approximate Q-Learning trên Pacman

| Bản đồ | Thuật toán | $N_{\text{train}}$ | Tỷ lệ thắng | Điểm TB | Thức ăn đã dọn | Số tham số đã học |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`smallGrid`** | `Tabular Q-Learning` | $50$ | $0,0\%$ | $-509,8$ | $6,2\%$ | $931$ |
| ($7\times 7$) | | $500$ | $35,0\%$ | $-158,7$ | $61,2\%$ | $4.261$ |
| | | $1000$ | **$65,0\%$** | $+141,9$ | $76,2\%$ | $5.730$ |
| | `Approximate Q-Learning` | **$50$** | **$72,5\%$** | **$+226,8$** | **$86,2\%$** | **$4$** |
| **`mediumClassic`** | `Tabular Q-Learning` | $50$ | $0,0\%$ | $-433,6$ | $13,4\%$ | $15.619$ |
| ($20\times 11$) | | $1000$ | **$0,0\%$** | **$-401,0$** | $14,7\%$ | **$182.412$** |
| | `Approximate Q-Learning` | **$50$** | **$92,5\%$** | **$+1241,2$** | **$97,1\%$** | **$4$** |
| | | **$200$** | **$95,0\%$** | **$+1281,6$** | **$98,6\%$** | **$4$** |

**Nhận xét kỹ thuật (Project 3)**:
- `PrioritizedSweeping` tập trung cập nhật dọc theo biên lan truyền phần thưởng từ các ô thoát nên giảm **$3,5\times \text{–} 4,6\times$** số phép cập nhật Bellman so với `ValueIteration` (chỉ cần tương đương $13\text{–}16$ vòng quét thay vì $61\text{–}67$ vòng trên lưới $20\times 20$ và $30\times 30$).
- Kết quả trên `mediumClassic` thể hiện rõ nhất **lời nguyền số chiều (curse of dimensionality)**: sau $1.000$ ván huấn luyện, `Tabular Q-Learning` lưu tới **$182.412$** cặp $(s, a)$ nhưng tỷ lệ thắng vẫn là **$0,0\%$** vì hầu như mọi trạng thái gặp khi kiểm tra đều chưa từng xuất hiện. Ngược lại, `Approximate Q-Learning` chỉ học đúng **$4$ trọng số đặc trưng** và đạt **$92,5\%$ thắng ngay sau 50 ván huấn luyện** (và **$95,0\%$** sau 200 ván).

---

### 3.4. Project 5 — Mở rộng Kích thước Ma trận trong Học máy

#### G. Tích chập 2D (`Convolve`) trên ma trận từ $28\times 28$ đến $256\times 256$

| Kích thước Ma trận | Nhân (Kernel) | Naive Loop (`tensordot` từng ô) | Vectorized Kernel-Slice (`models.Convolve`) | Hệ số Tăng tốc (`Speedup`) | Sai số cực đại |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$28\times 28$** | $3\times 3$ | $103,93$ ms | $3,61$ ms | **$28,8\times$** | $4,8\times 10^{-7}$ |
| **$64\times 64$** | $3\times 3$ | $465,64$ ms | $0,53$ ms | **$871,6\times$** | $9,5\times 10^{-7}$ |
| **$128\times 128$** | $3\times 3$ | $2.078,31$ ms | **$0,90$ ms** | **$2.313,8\times$** | $7,2\times 10^{-7}$ |
| **$128\times 128$** | $5\times 5$ | $2.128,50$ ms | **$1,74$ ms** | **$1.222,3\times$** | $2,4\times 10^{-6}$ |
| **$128\times 128$** | $7\times 7$ | $2.218,25$ ms | **$4,04$ ms** | **$549,2\times$** | $9,5\times 10^{-6}$ |

#### H. Mở rộng Ma trận Causal Self-Attention ($B = 8$)

| Độ dài chuỗi $T$ | Chiều nhúng $C$ | Kích thước ma trận Attention | Bộ nhớ ma trận Attention | Thời gian FW+BW (ms) | Thông lượng (tokens/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$16$** | $128$ | $8\times 16\times 16$ | $8,0$ KB | $294,3$ | $435$ |
| **$64$** | $128$ | $8\times 64\times 64$ | $128,0$ KB | $971,5$ | $527$ |
| **$128$** | $128$ | $8\times 128\times 128$ | $512,0$ KB | $1.035,0$ | $989$ |
| **$256$** | $128$ | $8\times 256\times 256$ | $2.048,0$ KB | $1.338,0$ | $1.531$ |
| **$512$** | $128$ | $8\times 512\times 512$ | **$8.192,0$ KB** | $2.255,8$ | $1.816$ |

**Nhận xét kỹ thuật (Project 5)**:
- Việc biểu diễn phép tích chập 2D thành tổng của $k_H \times k_W$ lát cắt ma trận (`models.Convolve`) giảm số lời gọi toán tử từ $(H-k+1)(W-k+1)$ xuống $k^2$, giúp tăng tốc tới **$2.313,8\times$** trên ma trận $128\times 128$ với sai số số học chỉ $\sim 10^{-7}$.
- Trên cơ chế `Attention`, khi chiều dài ngữ cảnh $T$ tăng gấp $32$ lần ($16 \to 512$), kích thước ma trận tương quan $Q K^\top / \sqrt{d_k}$ tăng gấp **$1.024$ lần** ($8\text{ KB} \to 8.192\text{ KB}$), minh họa trực quan quy luật bậc hai $O(T^2)$ của Transformer.
