# CS188 Artificial Intelligence — Projects 1, 2, 3, 5 & Nghiên Cứu Mở Rộng Quy Mô Lớn trên Đô Thị "Pacman Mega-City"

Kho lưu trữ này chứa lời giải hoàn chỉnh đạt điểm tuyệt đối kèm thưởng (**102 / 100 điểm**) cho 4 bài tập lớn cốt lõi của học phần Trí tuệ Nhân tạo (**UC Berkeley CS188 / AIT2004**), cùng một khung thực nghiệm mở rộng quy mô lớn ([`experiments/`](./experiments)) liên kết cả 4 dự án trên cùng một kịch bản đô thị hợp nhất **Pacman Mega-City ($31\times 21$)** và các ma trận kích thước lớn ($151\times 151$, $k=11$ thức ăn, MDP $40\times 40$, ảnh $256\times 256$, chuỗi $T=512$).

---

## 1. Tài Liệu Chính Thức trong Kho Mã Nguồn

Để tránh trùng lặp nội dung và đảm bảo tính nhất quán tuyệt đối với dữ liệu CSV, hệ thống tài liệu được phân tách rõ ràng theo 3 cấp độ:

1. **[`experiments/REPORT.md`](./experiments/REPORT.md) — Báo Cáo Khoa Học Toàn Diện (Tài liệu chính):**  
   Trình bày đầy đủ mục tiêu, phương pháp toán học của từng câu hỏi trong Projects 1, 2, 3, 5 (điểm autograder, tham số `analysis.py`, kiến trúc mạng Project 5) và toàn bộ kết quả định lượng của **9 bộ thí nghiệm mở rộng (A $\to$ I)** kèm độ lệch chuẩn (`± std`), khoảng tin cậy `Wilson 95% CI`, hệ số lũy thừa $\alpha$, và 11 biểu đồ phân tích.
2. **[`experiments/EXPLANATION_GUIDE.md`](./experiments/EXPLANATION_GUIDE.md) — Hướng Dẫn Kiến Trúc Mã Nguồn & Chạy Trực Quan GUI:**  
   Giải thích chi tiết từng hàm/lớp thực tế trong mã nguồn (`make_megacity`, `SwampProblem`, `mazeBetterEvaluationFunction`, `OrderedAlphaBetaAgent`, `MegaCityExtractor`, `CharAttentionLM`, `ImitationCNN`) và các lệnh dòng lệnh để mở cửa sổ đồ họa xem Pacman tự chạy.
3. **[`experiments/summarize.py`](./experiments/summarize.py) & [`experiments/results/`](./experiments/results) — Dữ Liệu Kiểm Chứng Tự Động:**  
   Chứa 9 tệp dữ liệu thực nghiệm `.csv` và kịch bản tổng hợp thống kê tự động đảm bảo **100% số liệu trong tài liệu khớp hoàn toàn với kết quả thực thi**.

---

## 2. Tổng Hợp Điểm Chấm Tự Động (CS188 Autograder)

| Project | Chủ đề Cốt lõi | Tệp Bài nộp Chuẩn | Điểm Đạt được | Chi tiết Từng Câu hỏi |
| :--- | :--- | :--- | :---: | :--- |
| **Project 1** | Tìm kiếm Không gian Trạng thái (`search/`) | [`search.py`](./search/search.py), [`searchAgents.py`](./search/searchAgents.py) | **26 / 25** | Q1: 3/3, Q2: 3/3, Q3: 3/3, Q4: 3/3, Q5: 3/3, Q6: 3/3, **Q7: 5/4** (6.690 nút `< 7.000`), Q8: 3/3 |
| **Project 2** | Tìm kiếm Đối kháng Đa tác tử (`multiagent/`) | [`multiAgents.py`](./multiagent/multiAgents.py) | **25 / 25** | Q1: 4/4, Q2: 5/5, Q3: 5/5, Q4: 5/5, Q5: 6/6 (`> 1.350` điểm trung bình) |
| **Project 3** | Quy hoạch Động MDP & Học Tăng cường (`reinforcement/`) | [`valueIterationAgents.py`](./reinforcement/valueIterationAgents.py), [`qlearningAgents.py`](./reinforcement/qlearningAgents.py), [`analysis.py`](./reinforcement/analysis.py) | **26 / 25** | Q1: 4/4, Q2: 5/5, Q3: 5/5, **Q4: 5/4** (`PrioritizedSweeping`), Q5: 2/2, Q6: 4/4, Q7: 1/1 |
| **Project 5** | Học Máy & Mạng Nơ-ron Sâu (`machinelearning/`) | [`models.py`](./machinelearning/models.py), [`train.py`](./machinelearning/train.py) | **25 / 25** | Q1: 6/6, Q2: 4/4, Q3: 6/6 (`>98%` MNIST), Q4: 5/5 (`>86.5%` LangID), Q5: 2/2, Q6: 2/2 |
| **Tổng cộng** | | **10 tệp bài nộp chuẩn** | **102 / 100** | Vượt qua 100% test cases của cả 4 bộ `autograder.py` |

---

## 3. Điểm Nhấn Kết Quả từ Các Thí Nghiệm Mở Rộng (A $\to$ I)

Toàn bộ số liệu dưới đây được trích xuất trực tiếp từ `python experiments/summarize.py` trên 9 tệp CSV tại [`experiments/results/`](./experiments/results):

| Thí nghiệm | Tệp Dữ liệu CSV | Quy mô Mở rộng | Kết quả Định lượng Chính xác (Khớp 100% CSV) |
| :--- | :--- | :--- | :--- |
| **Exp A: Search Scaling & Đầm lầy** | [`search_scaling.csv`](./experiments/results/search_scaling.csv) (282 dòng) | Mê cung $31^2 \to 151^2$, `MegaCity 31x21`, 25% đầm lầy (`~`, cost 5) | • Ở $151\times 151$ (đầm lầy): `DFS` đội chi phí lên **$5.89\times$** ($3{,}189.6$ vs $548.0$), `BFS` lệch **$1.24\times$** ($688.0$ vs $548.0$, và lệch **$1.56\times$** trên `MegaCity-Route`: $107.3$ vs $68.7$).<br>• `DFS` có đỉnh hàng đợi `peak_frontier` lớn hơn `BFS` trong mê cung ($258.2$ vs $70.4$ ở $151\times 151$, tỷ lệ $\sim N^{1.73}$ vs $\sim N^{1.01}$).<br>• `WA*-Manhattan (w=2.0)` giảm bậc mở rộng xuống gần tuyến tính ($\sim N^{1.01}$), chỉ mở rộng **$946.2 \pm 264.8$ nút ($8.0\%$ mê cung)** ở $151\times 151$ trong $6.69\text{ ms}$. |
| **Exp B: Multi-Food & MST** | [`search_food.csv`](./experiments/results/search_food.csv) (90 dòng) | $31\times 31$, $k \in \{3, 5, 7, 9, 11\}$ thức ăn ($|S| = |\mathcal{V}| \cdot 2^k$) | • Ở $k=7$: `A* + foodHeuristic` mở rộng **$7{,}084.7 \pm 5{,}908.6$ nút** so với **$29{,}223.3 \pm 5{,}653.3$ nút** của `UCS`. Tiền tính toán BFS (`precomputed`, mất $4\text{–}10\text{ ms}$) giúp giảm thời gian ở $k=3$ từ $2{,}922.6\text{ ms}$ xuống **$679.4\text{ ms}$** (nhanh hơn `UCS` $1{,}945.0\text{ ms}$).<br>• `A* + foodHeuristic (precomputed)` giải được **`1/3` mẫu ở $k=11$** ($29{,}478$ nút ở seed 1; trung bình 3 seeds là $209{,}826.0$ nút).<br>• **`A* + MSTHeuristic (precomputed)`** (Cây khung nhỏ nhất Prim) giải **100% (`3/3`) mẫu tới $k=11$** với chi phí tối ưu $1.00\times$ ($178.3$), chỉ mở rộng **$694.7 \pm 481.6$ nút** trong **$3{,}015.4\text{ ms}$**! `ClosestDot` mở rộng $1{,}122.0$ nút ($403.2\text{ ms}$, chi phí $1.16\times$). |
| **Exp C: Đấu trường 3 Ghosts & `mazeEval`** | [`multiagent_games.csv`](./experiments/results/multiagent_games.csv) (288 dòng) | `mediumClassic`, `MegaCity 31x21` (3 ma), `Arena 41x21` (3 ma, gấp $3.91\times$ `mediumClassic`), 8 seeds/ô | • Thay khoảng cách Manhattan + `±inf` (`betterEval`) bằng khoảng cách mê cung BFS có cache + giá trị hữu hạn $\pm 10{,}000$ (`mazeEval`) giúp xóa bỏ hoàn toàn lỗi kẹt tường 600 nước: trên `MegaCity 31x21` (vs `RandomGhost`), tỷ lệ thắng tăng từ **$0.0\%$ (Timeout 8/8) lên $87.5\%$ (`7/8`, CI `[52.9, 97.8]`)**, dọn **$99.7\%$ thức ăn**, điểm tăng từ $-47.5$ lên **$1{,}693.0 \pm 361.2$** (`AlphaBeta`) và **$1{,}757.0 \pm 309.6$** (`Expectimax`).<br>• Trước 3 `DirectionalGhost` trên `MegaCity`, `Expectimax-d2 (mazeEval)` đạt **$25.0\%$ thắng**, điểm **$1{,}045.8 \pm 858.5$** (so với $-84.0$ của `betterEval`). |
| **Exp D: Cắt tỉa Alpha-Beta & Move Ordering** | [`multiagent_pruning.csv`](./experiments/results/multiagent_pruning.csv) (70 dòng) | Đo tuần tự độc lập (Median 5 lần chạy), $d=2, 3$ | • Khớp quyết định nước đi **$100.0\%$** với `Minimax` trên toàn bộ 70 trạng thái.<br>• Ở $d=3$ (`mediumClassic`, 9 ply): `Minimax` duyệt $952.2$ nút $\to$ `AlphaBeta` duyệt **$375.0$ nút** (cắt **$44.8\%$**, tăng tốc **$2.98\times$**) $\to$ `OrderedAlphaBeta` duyệt **$323.4$ nút** (cắt **$56.8\%$**). |
| **Exp E: Hội tụ MDP (Backups vs Q-Evals)** | [`rl_mdp_summary.csv`](./experiments/results/rl_mdp_summary.csv) (26 dòng), [`rl_mdp_curves.csv`](./experiments/results/rl_mdp_curves.csv) | Gridworld $5\times 5 \to 40\times 40$ ($22 \to 1{,}319$ trạng thái) | • `PrioritizedSweeping` giảm **$3.46\times \to 4.64\times$** số lần cập nhật trạng thái (`backups_to_tol`): ở $30\times 30$ ($745.0$ trạng thái) giảm từ $50{,}166.0$ xuống **$11{,}900.0$ backups** ($16.0$ vòng quét vs $67.3$ vòng); ở $40\times 40$ ($1{,}318.7$ trạng thái) giảm từ $98{,}464.3$ xuống **$24{,}950.0$ backups**.<br>• Tuy nhiên, do mỗi lần `pop` phải tính lại $Q(p,a)$ cho các tiền nhiệm, tổng số lần tính $Q$ (`q_evals_to_tol`) của PS cao hơn VI khoảng $1.05\times\text{–}1.48\times$ ($224{,}357$ vs $195{,}008$ ở $30\times 30$). |
| **Exp F: Q-Learning & Zero-Shot Transfer** | [`rl_pacman.csv`](./experiments/results/rl_pacman.csv) (144 dòng) | `smallGrid`, `mediumClassic` $\to$ Chuyển giao Zero-Shot sang `MegaCity 31x21` (90 ván test/ô) | • Đối chứng `ApproxQ (IdentityExtractor)` trùng khít với `Tabular Q-Learning` (cùng **$0.0\%$ thắng** trên `mediumClassic` ở $N=1000$ với $54{,}762$ và $178{,}504$ tham số), chứng minh khả năng khái quát hóa đến từ đặc trưng.<br>• Trên `mediumClassic`, `MegaCityExtractor` (7 trọng số) đạt điểm cao nhất **$1{,}794.6 \pm 12.3$** ($91.1\%$ thắng ở $N=500$).<br>• Khi chuyển giao **Zero-Shot** sang `MegaCity 31x21` (3 Ghosts), `SimpleExtractor` (4 trọng số) thắng ngay **$83.3 \pm 5.8\%$ (`CI [74.3, 89.6]`, điểm $1{,}060.8$, dọn $94.2\%$ thức ăn)** mà không cần huấn luyện lại! |
| **Exp G: Tích chập 2D (`Convolve` vs `F.conv2d`)** | [`ml_convolve_scaling.csv`](./experiments/results/ml_convolve_scaling.csv) (12 dòng) | Ma trận $28^2 \to 256^2$, Kernel $3\times 3, 5\times 5, 7\times 7$ (Đơn luồng, Median 15 lần) | • Ở $128\times 128$: `Vectorized Q5` đạt $0.351\text{ ms}$ ($k=3$, tăng tốc **$2{,}656.1\times$**), $0.889\text{ ms}$ ($k=5$, **$985.4\times$**), $1.781\text{ ms}$ ($k=7$, **$435.4\times$**).<br>• Ở $256\times 256$ ($k=3$): `Vectorized Q5` đạt **$0.590 \pm 0.025\text{ ms}$** (tăng tốc **$6{,}106.4\times$** so với `Naive` $3{,}601.53\text{ ms}$ và **chỉ chậm hơn C++ `F.conv2d` $1.4\times$** ở $0.421\text{ ms}$). Sai số tuyệt đối $\le 9.5\times 10^{-6}$. |
| **Exp H & H2: Self-Attention & Char-LM** | [`ml_attention_scaling.csv`](./experiments/results/ml_attention_scaling.csv), [`ml_char_lm.csv`](./experiments/results/ml_char_lm.csv) | $T = 16 \to 512$, $C \in \{64, 128, 256\}$, $B=8$ + Huấn luyện Char-LM | • Khi $T$ tăng từ $64 \to 512$ ($C=64$), thông lượng giảm từ **$217{,}428 \to 70{,}381\text{ tok/s}$** do chi phí $O(T^2)$ ($\text{Time} \sim T^{1.58}$); tổng bộ nhớ kích hoạt Autograd thực tế tăng từ $152.0\text{ KB}$ ($T=16$) lên **$28{,}672.0\text{ KB}$** ($C=64$) và **$40{,}960.0\text{ KB}$** ($C=256$) ở $T=512$.<br>• Trong bài toán Character-LM (`vocab=90`), `Attention-CharLM (Q6)` giảm Validation Perplexity xuống **$8.89$** (`acc = 44.7%`), vượt qua `Bigram-MLP` (**`PPL = 9.72`**, `acc = 40.6%`). |
| **Exp I: Học Bắt chước trên Mega-City** | [`ml_imitation.csv`](./experiments/results/ml_imitation.csv) (3 dòng) | Học bắt chước `AlphaBeta-d2 (mazeEval)` bằng `Imitation-MLP` & `Imitation-CNN` (`Convolve`) | • Khớp hành động chuyên gia đạt **$86.4\%$** (`MLP`) và **$85.5\%$** (`CNN`), tăng tốc suy luận gấp **$20.7\times$** ($0.751\text{ ms/nước}$) và **$11.8\times$** ($1.317\text{ ms/nước}$) so với Giáo viên Alpha-Beta ($15.545\text{ ms/nước}$); `Imitation-MLP` dọn $27.6\%$ thức ăn trước 3 `DirectionalGhosts` (vượt `Reflex` $22.1\%$ và `AlphaBeta-scoreEval` $23.0\%$). |

---

## 4. Hướng Dẫn Chạy Nhanh (Quickstart)

### 4.1. Chạy Chấm điểm Tự động 4 Project Gốc
```powershell
cd D:\AIT2004_7\search;          python autograder.py --no-graphics
cd D:\AIT2004_7\multiagent;      python autograder.py --no-graphics
cd D:\AIT2004_7\reinforcement;   python autograder.py --no-graphics
cd D:\AIT2004_7\machinelearning; python autograder.py
```

### 4.2. Chạy Xem Đồ họa Trực quan (GUI Demo)
> **Lưu ý:** Cần `cd` vào đúng thư mục của từng Project trước khi chạy `pacman.py` để trình nạp tác tử `loadAgent` tìm thấy tệp `*Agents.py`.

```powershell
# Project 1: Xem A* tìm đường trên mê cung lớn 61x61
cd D:\AIT2004_7\search
python pacman.py -l ../experiments/layouts/maze_61x61.lay -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic --frameTime 0.01 --zoom 0.7

# Project 2: Xem AlphaBeta / Expectimax thi đấu với 3 Ghosts trên Đô thị Mega-City 31x21
cd D:\AIT2004_7\multiagent
python pacman.py -p ExpectimaxAgent -a evalFn=better,depth=2 -l ../experiments/layouts/megacity_31x21.lay -g RandomGhost -k 3 --frameTime 0.03

# Project 3: Xem Approximate Q-Learning học 50 ván và thi đấu trên Đô thị Mega-City 31x21
cd D:\AIT2004_7\reinforcement
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 53 -l ../experiments/layouts/megacity_31x21.lay -k 3
```

### 4.3. Tái lập Toàn bộ Dữ liệu CSV & Biểu đồ Mở rộng
```powershell
cd D:\AIT2004_7
python experiments/scenario.py
python experiments/exp_search.py
python experiments/exp_multiagent.py
python experiments/exp_rl.py
python experiments/exp_ml.py
python experiments/summarize.py
python experiments/plot_all_figures.py
```
