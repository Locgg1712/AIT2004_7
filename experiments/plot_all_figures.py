"""
experiments/plot_all_figures.py
-------------------------------
Tạo toàn bộ biểu đồ trực quan chất lượng cao từ dữ liệu CSV thực nghiệm:
1. Project 1:
   - p1_search_scaling.png: Mở rộng kích thước mê cung (Unit & Swamp), Suboptimality và Peak Frontier (DFS vs BFS)
   - p1_food_combinatorics.png: Bùng nổ tổ hợp Multi-Food (Nodes & Runtime) và Pareto Frontier (UCS, A* food, A* MST, WA*, ClosestDot)
2. Project 2:
   - p2_alphabeta_pruning.png: Hiệu suất cắt tỉa nút và Speedup (Minimax vs AlphaBeta vs OrderedAlphaBeta)
   - p2_arena_performance.png: Đối chiếu 6 tác tử (betterEval vs mazeEval) trên Arena 41x21 và MegaCity 31x21
3. Project 3:
   - p3_bellman_convergence.png: Hội tụ Bellman trung bình qua các seed (theo Backups và theo Q-evaluations)
   - p3_qlearning_generalization.png: Tách biệt smallGrid, mediumClassic và Zero-Shot Transfer sang MegaCity (có đối chứng IdentityExtractor)
   - p3_mdp_value_heatmap.png: Bản đồ nhiệt V*(s) trên lưới MDP 10x10
4. Project 5 & Liên kết Đa Dự án:
   - p5_sin_regression.png: Xấp xỉ hàm sin(x) của RegressionModel
   - p5_attention_matrix_heatmap.png: Heatmap ma trận Causal Masked Self-Attention
   - p5_scaling_performance.png: Tăng tốc Convolve 2D (Naive vs Vectorized Q5 vs F.conv2d) & Bộ nhớ thực tế vs lý thuyết của Attention
   - p5_megacity_imitation_charlm.png: Huấn luyện Char-LM bằng Attention (Q6) & Học bắt chước Imitation-CNN (Q5) trên MegaCity
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, 'experiments')
RESULTS_DIR = os.path.join(EXP_DIR, 'results')
FIGURES_DIR = os.path.join(EXP_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# -------------------------------------------------------------
# 1. Project 1 Plots: Search Scaling & Food Scaling
# -------------------------------------------------------------
def plot_search():
    print("[1/4] Vẽ biểu đồ Project 1 (Search)...")
    csv_scaling = os.path.join(RESULTS_DIR, 'search_scaling.csv')
    if os.path.exists(csv_scaling):
        df = pd.read_csv(csv_scaling)
        # Chỉ lấy các kích thước mê cung N x N tiêu chuẩn (map == 'maze') để vẽ đường scaling mượt theo N
        df_sq = df[df['map'] == 'maze'].copy()
        df_sq['n'] = df_sq['size'].apply(lambda s: int(str(s).split('x')[0]))
        df_unit = df_sq[df_sq['terrain'] == 'unit']
        agg = df_unit.groupby(['n', 'algo']).agg(
            expanded_mean=('expanded', 'mean'),
            expanded_std=('expanded', 'std'),
            cost_mean=('cost', 'mean'),
            frontier_mean=('peak_frontier', 'mean'),
        ).reset_index()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))
        algos_colors = [
            ('DFS', '#e74c3c', 'o'),
            ('BFS', '#3498db', 's'),
            ('UCS', '#9b59b6', '^'),
            ('A*-Manhattan', '#2ecc71', 'D'),
            ('WA*-Manhattan (w=2.0)', '#f39c12', 'v'),
        ]

        for alg, color, marker in algos_colors:
            sub = agg[agg['algo'] == alg].sort_values('n')
            if not sub.empty:
                ax1.errorbar(sub['n'], sub['expanded_mean'], yerr=sub['expanded_std'].fillna(0),
                             marker=marker, color=color, linewidth=2, capsize=3, label=alg)
                ax2.plot(sub['n'], sub['cost_mean'], marker=marker, color=color, linewidth=2, label=alg)
                ax3.plot(sub['n'], sub['frontier_mean'], marker=marker, color=color, linewidth=2, label=alg)

        ax1.set_title("A1: Số nút mở rộng (Unit Cost ± std)", fontsize=10.5, fontweight='bold')
        ax1.set_xlabel("Kích thước mê cung (N x N)")
        ax1.set_ylabel("Số nút mở rộng (Nodes Expanded)")
        ax1.legend(fontsize=8.5)
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2.set_title("A2: Chi phí đường đi (Suboptimality của DFS/WA*)", fontsize=10.5, fontweight='bold')
        ax2.set_xlabel("Kích thước mê cung (N x N)")
        ax2.set_ylabel("Chi phí đường đi (Path Cost)")
        ax2.legend(fontsize=8.5)
        ax2.grid(True, linestyle='--', alpha=0.6)

        ax3.set_title("A3: Đỉnh hàng đợi (Peak Frontier: DFS > BFS)", fontsize=10.5, fontweight='bold')
        ax3.set_xlabel("Kích thước mê cung (N x N)")
        ax3.set_ylabel("Kích thước Frontier cực đại")
        ax3.legend(fontsize=8.5)
        ax3.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'p1_search_scaling.png'), dpi=200)
        plt.close()

    csv_food = os.path.join(RESULTS_DIR, 'search_food.csv')
    if os.path.exists(csv_food):
        df_f = pd.read_csv(csv_food)
        agg_f = df_f[df_f['solved'] == True].groupby(['k', 'algo']).agg(
            expanded_mean=('expanded', 'mean'),
            time_mean=('time_ms', 'mean'),
        ).reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        food_algos = [
            ('UCS', '#e74c3c', 'o'),
            ('A* + foodHeuristic', '#e67e22', 's'),
            ('A* + foodHeuristic (precomputed)', '#f1c40f', 'D'),
            ('A* + MSTHeuristic (precomputed)', '#2ecc71', '^'),
            ('WA* + MST (w=2.0)', '#3498db', 'v'),
            ('ClosestDot (greedy)', '#95a5a6', 'x'),
        ]
        for alg, color, marker in food_algos:
            sub = agg_f[agg_f['algo'] == alg].sort_values('k')
            if not sub.empty:
                ax1.plot(sub['k'], sub['expanded_mean'], marker=marker, color=color, linewidth=2.2, label=alg)
                ax2.plot(sub['k'], sub['time_mean'], marker=marker, color=color, linewidth=2.2, label=alg)

        ax1.set_yscale('log')
        ax1.set_title("B1: Số nút mở rộng theo số viên thức ăn k (Log scale)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Số lượng viên thức ăn (k)")
        ax1.set_ylabel("Số nút mở rộng (Log Scale)")
        ax1.legend(fontsize=8.5)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)

        ax2.set_yscale('log')
        ax2.set_title("B2: Thời gian thực thi (ms) — Hiệu quả Precomputed Distance", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Số lượng viên thức ăn (k)")
        ax2.set_ylabel("Thời gian chạy (ms, Log Scale)")
        ax2.legend(fontsize=8.5)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'p1_food_combinatorics.png'), dpi=200)
        plt.close()


# -------------------------------------------------------------
# 2. Project 2 Plots: AlphaBeta Pruning & Arena/MegaCity Games
# -------------------------------------------------------------
def plot_multiagent():
    print("[2/4] Vẽ biểu đồ Project 2 (Multi-Agent)...")
    csv_prune = os.path.join(RESULTS_DIR, 'multiagent_pruning.csv')
    if os.path.exists(csv_prune):
        df_p = pd.read_csv(csv_prune)
        agg_p = df_p.groupby(['map', 'depth']).agg(
            mm_nodes=('minimax_nodes', 'mean'),
            ab_nodes=('alphabeta_nodes', 'mean'),
            ord_nodes=('ordered_ab_nodes', 'mean'),
            ab_speedup=('speedup', 'mean'),
            ord_speedup=('ordered_speedup', 'mean'),
        ).reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(agg_p))
        width = 0.26
        labels = [f"{row['map']}\nd={row['depth']}" for _, row in agg_p.iterrows()]

        ax1.bar(x - width, agg_p['mm_nodes'], width, label='Minimax', color='#e74c3c', alpha=0.85)
        ax1.bar(x, agg_p['ab_nodes'], width, label='AlphaBeta', color='#3498db', alpha=0.85)
        if 'ord_nodes' in agg_p and not agg_p['ord_nodes'].isna().all():
            ax1.bar(x + width, agg_p['ord_nodes'], width, label='OrderedAlphaBeta', color='#2ecc71', alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=8.5)
        ax1.set_yscale('log')
        ax1.set_ylabel("Số nút mở rộng mỗi lượt (Log Scale)")
        ax1.set_title("D1: Số nút duyệt của Minimax vs AlphaBeta vs OrderedAB", fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)

        ax2.bar(x - width/2, agg_p['ab_speedup'], width, label='AlphaBeta Speedup', color='#3498db', alpha=0.85)
        if 'ord_speedup' in agg_p and not agg_p['ord_speedup'].isna().all():
            ax2.bar(x + width/2, agg_p['ord_speedup'], width, label='OrderedAlphaBeta Speedup', color='#2ecc71', alpha=0.85)
        ax2.axhline(1.0, color='red', linestyle='--', label='Minimax Baseline (1.0x)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=8.5)
        ax2.set_ylabel("Hệ số tăng tốc thời gian (Speedup)")
        ax2.set_title("D2: Tăng tốc thời gian thực (Đo tuần tự độc lập, Median)", fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'p2_alphabeta_pruning.png'), dpi=200)
        plt.close()

    csv_games = os.path.join(RESULTS_DIR, 'multiagent_games.csv')
    if os.path.exists(csv_games):
        df_g = pd.read_csv(csv_games)
        large_df = df_g[df_g['map'].str.contains('Arena|MegaCity', case=False, na=False)]
        if not large_df.empty:
            agg_g = large_df.groupby(['map', 'agent']).agg(
                win_rate=('win', 'mean'),
                score_mean=('score', 'mean'),
                score_std=('score', 'std'),
                timeout_rate=('timeout', 'mean'),
            ).reset_index()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
            maps = agg_g['map'].unique()
            agents = agg_g['agent'].unique()
            x = np.arange(len(agents))
            width = 0.36

            for idx, m in enumerate(maps):
                sub = agg_g[agg_g['map'] == m].set_index('agent').reindex(agents).reset_index()
                offset = (idx - (len(maps) - 1) / 2) * width
                ax1.bar(x + offset, sub['score_mean'], width, yerr=sub['score_std'].fillna(0),
                        capsize=3, label=m, alpha=0.85)
                ax2.bar(x + offset, sub['win_rate'] * 100.0, width, label=f"{m} (Win%)", alpha=0.85)

            ax1.set_xticks(x)
            ax1.set_xticklabels(agents, rotation=20, ha='right', fontsize=8.5)
            ax1.set_title("C1: Điểm số trung bình (± std) trên Arena & MegaCity", fontsize=11, fontweight='bold')
            ax1.set_ylabel("Điểm trung bình (Average Score)")
            ax1.legend(fontsize=8.5)
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2.set_xticks(x)
            ax2.set_xticklabels(agents, rotation=20, ha='right', fontsize=8.5)
            ax2.set_title("C2: Tỷ lệ thắng (%) — Hiệu quả của mazeBetterEvaluationFunction", fontsize=11, fontweight='bold')
            ax2.set_ylabel("Tỷ lệ thắng (Win Rate %)")
            ax2.set_ylim(0, 105)
            ax2.legend(fontsize=8.5)
            ax2.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'p2_arena_performance.png'), dpi=200)
            plt.close()


# -------------------------------------------------------------
# 3. Project 3 Plots: RL MDP Convergence & Pacman Generalization
# -------------------------------------------------------------
def plot_reinforcement():
    print("[3/4] Vẽ biểu đồ Project 3 (Reinforcement Learning)...")
    csv_curves = os.path.join(RESULTS_DIR, 'rl_mdp_curves.csv')
    if os.path.exists(csv_curves):
        df_c = pd.read_csv(csv_curves)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for size, ax in zip([20, 30], [ax1, ax2]):
            sub = df_c[df_c['size'] == size]
            for algo, color, label in [
                ('ValueIteration', '#e74c3c', 'Standard Value Iteration'),
                ('PrioritizedSweeping', '#2ecc71', 'Prioritized Sweeping'),
            ]:
                s_alg = sub[sub['algo'] == algo]
                if not s_alg.empty:
                    # Nội suy trên lưới backups chung để lấy trung bình và độ lệch chuẩn qua các seed
                    max_b = s_alg.groupby('seed')['backups'].max().min()
                    grid_b = np.linspace(s_alg['backups'].min(), max_b, 80)
                    interp_vals = []
                    for seed, grp in s_alg.groupby('seed'):
                        grp_sorted = grp.sort_values('backups')
                        y_interp = np.interp(grid_b, grp_sorted['backups'], np.maximum(grp_sorted['max_err'], 1e-8))
                        interp_vals.append(y_interp)
                    if interp_vals:
                        arr = np.vstack(interp_vals)
                        mean_y = arr.mean(axis=0)
                        std_y = arr.std(axis=0)
                        ax.plot(grid_b, mean_y, label=label, color=color, linewidth=2.2)
                        ax.fill_between(grid_b, np.maximum(mean_y - std_y, 1e-8), mean_y + std_y, color=color, alpha=0.2)

            ax.set_yscale('log')
            ax.set_title(f"E: Hội tụ Bellman MDP {size}x{size} (Trung bình ± std qua 3 seeds)", fontsize=11, fontweight='bold')
            ax.set_xlabel("Số phép cập nhật trạng thái (State Backups)")
            ax.set_ylabel("Max Error ||V - V*||_inf (Log scale)")
            ax.legend()
            ax.grid(True, which='both', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'p3_bellman_convergence.png'), dpi=200)
        plt.close()

    csv_pacman = os.path.join(RESULTS_DIR, 'rl_pacman.csv')
    if os.path.exists(csv_pacman):
        df_p = pd.read_csv(csv_pacman)
        agg_p = df_p.groupby(['layout', 'agent', 'n_train']).agg(
            win_mean=('win_rate', 'mean'),
            win_std=('win_rate', 'std'),
            score_mean=('avg_score', 'mean'),
            score_std=('avg_score', 'std'),
        ).reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        agent_styles = [
            ('Tabular Q-Learning', '#e74c3c', 'o'),
            ('ApproxQ (IdentityExtractor)', '#e67e22', '^'),
            ('ApproxQ (SimpleExtractor)', '#2ecc71', 's'),
            ('ApproxQ (MegaCityExtractor)', '#2980b9', 'D'),
        ]

        for layout, ax in zip(['smallGrid', 'mediumClassic'], [ax1, ax2]):
            sub_l = agg_p[agg_p['layout'] == layout]
            for agent, color, marker in agent_styles:
                sub_a = sub_l[sub_l['agent'] == agent].sort_values('n_train')
                if not sub_a.empty:
                    ax.errorbar(sub_a['n_train'], sub_a['win_mean'] * 100.0,
                                yerr=sub_a['win_std'].fillna(0) * 100.0,
                                marker=marker, color=color, linewidth=2.2, capsize=3, label=agent)
            ax.set_title(f"F: Tỷ lệ thắng (%) trên {layout} (Trung bình ± std)", fontsize=11, fontweight='bold')
            ax.set_xlabel("Số ván huấn luyện (N_train)")
            ax.set_ylabel("Tỷ lệ thắng (Win Rate %)")
            ax.set_ylim(-5, 105)
            ax.legend(fontsize=8.5)
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'p3_qlearning_generalization.png'), dpi=200)
        plt.close()

    # 3. Heatmap giá trị trạng thái V*(s) trên lưới MDP 10x10
    sys.path.insert(0, os.path.join(BASE_DIR, 'reinforcement'))
    sys.path.insert(0, os.path.join(BASE_DIR, 'experiments'))
    import gridworld
    from valueIterationAgents import ValueIterationAgent
    import scenario

    grid_obj = scenario.make_gridworld(10, 10, seed=42)
    grid_mdp = gridworld.Gridworld(grid_obj)
    grid_mdp.setNoise(0.2)
    vi_agent = ValueIterationAgent(grid_mdp, discount=0.9, iterations=100)

    val_matrix = np.full((10, 10), np.nan)
    for x in range(10):
        for y in range(10):
            val = vi_agent.getValue((x, y))
            val_matrix[9 - y, x] = val

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.imshow(val_matrix, cmap='viridis', interpolation='nearest')
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Hàm giá trị V*(s)")
    ax.set_title("Project 3: Bản đồ nhiệt (Heatmap) Hàm giá trị V*(s) trên MDP 10x10", fontsize=11, fontweight='bold')
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate (Đảo chiều hiển thị)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'p3_mdp_value_heatmap.png'), dpi=200)
    plt.close()


# -------------------------------------------------------------
# 4. Project 5 Plots: Regression, Attention, Scaling & Imitation
# -------------------------------------------------------------
def plot_machinelearning():
    print("[4/4] Vẽ biểu đồ Project 5 (Machine Learning & Deep Learning)...")
    sys.path.insert(0, os.path.join(BASE_DIR, 'machinelearning'))
    import torch
    from torch import tensor
    from models import RegressionModel, Attention
    from train import train_regression
    import backend

    # 1. Vẽ đồ thị khớp hàm sin(x) của RegressionModel
    model = RegressionModel()
    dataset = backend.RegressionDataset(model=model)
    train_regression(model, dataset)

    xs = np.linspace(-2 * np.pi, 2 * np.pi, 300)
    ys_true = np.sin(xs)
    with torch.no_grad():
        x_tensor = tensor(xs.reshape(-1, 1), dtype=torch.float32)
        ys_pred = model(x_tensor).numpy().flatten()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(xs, ys_true, 'r--', label='Đường chuẩn thực tế y = sin(x)', linewidth=2.5)
    ax.plot(xs, ys_pred, 'b-', label='Mạng MLP RegressionModel dự đoán', linewidth=2.0)
    ax.set_title("Project 5: Kết quả Huấn luyện Xấp xỉ hàm sin(x) của RegressionModel", fontsize=11, fontweight='bold')
    ax.set_xlabel("x (radians)")
    ax.set_ylabel("y = f(x)")
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'p5_sin_regression.png'), dpi=200)
    plt.close()

    # 2. Heatmap ma trận Causal Attention Weights
    torch.manual_seed(42)
    attn_layer = Attention(layer_size=16, block_size=32)
    sample_input = torch.randn(1, 16, 16)
    with torch.no_grad():
        q = attn_layer.q_layer(sample_input)
        k = attn_layer.k_layer(sample_input)
        M = torch.matmul(q, torch.movedim(k, 1, 2)) / (16 ** 0.5)
        M = M.masked_fill(attn_layer.mask[:, :, :16, :16] == 0, float('-inf'))[0]
        weights = torch.softmax(M, dim=-1).squeeze().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(weights, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax, ax=ax, label="Attention Weight (Xác suất chú ý)")
    ax.set_title("Project 5: Ma trận Trọng số Chú ý (Causal Masked Self-Attention)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Key Position (Token vị trí nguồn)")
    ax.set_ylabel("Query Position (Token vị trí hiện tại)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'p5_attention_matrix_heatmap.png'), dpi=200)
    plt.close()

    # 3. Biểu đồ tăng tốc Vector hóa Convolve & Bậc hai Attention
    csv_conv = os.path.join(RESULTS_DIR, 'ml_convolve_scaling.csv')
    csv_attn = os.path.join(RESULTS_DIR, 'ml_attention_scaling.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if os.path.exists(csv_conv):
        df_conv = pd.read_csv(csv_conv)
        for k_str, color, marker in [('3x3', '#2980b9', 'o'), ('5x5', '#e67e22', 's'), ('7x7', '#27ae60', '^')]:
            sub_k = df_conv[df_conv['kernel_size'] == k_str].sort_values('size')
            if not sub_k.empty:
                ax1.plot(sub_k['matrix_size'], sub_k['speedup'], marker=marker,
                         color=color, linewidth=2.2, label=f'Vectorized Q5 vs Naive (Kernel {k_str})')
        ax1.set_title("G: Hệ số tăng tốc của Vectorized Convolve (Single-Thread Median)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Kích thước ma trận ảnh (N x N)")
        ax1.set_ylabel("Hệ số tăng tốc (Speedup vs Naive 2-Loop)")
        ax1.legend(fontsize=9)
        ax1.grid(True, linestyle='--', alpha=0.5)

    if os.path.exists(csv_attn):
        df_attn = pd.read_csv(csv_attn)
        sub_c64 = df_attn[df_attn['emb_dim_C'] == 64].sort_values('seq_len_T')
        if not sub_c64.empty:
            ax2.plot(sub_c64['seq_len_T'], sub_c64['attn_matrix_kb'], marker='o', linestyle='--',
                     color='#8e44ad', linewidth=2.0, label='Lý thuyết 1x Ma trận B*T^2*4B (KB)')
            if 'measured_attn_tensors_kb' in sub_c64:
                ax2.plot(sub_c64['seq_len_T'], sub_c64['measured_attn_tensors_kb'], marker='s',
                         color='#e67e22', linewidth=2.2, label='Đo thực tế: Ma trận Điểm + Trọng số (KB)')
            if 'measured_total_activations_kb' in sub_c64:
                ax2.plot(sub_c64['seq_len_T'], sub_c64['measured_total_activations_kb'], marker='^',
                         color='#c0392b', linewidth=2.2, label='Đo thực tế: Toàn bộ Tensor Autograd (KB)')
        ax2.set_title("H: Chi phí bộ nhớ O(T^2) của Self-Attention (C=64, B=8)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Độ dài chuỗi Sequence Length (T)")
        ax2.set_ylabel("Dung lượng bộ nhớ (KB)")
        ax2.legend(fontsize=8.5)
        ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'p5_scaling_performance.png'), dpi=200)
    plt.close()

    # 4. Biểu đồ Char-LM (H2) và Imitation Learning trên MegaCity (I)
    csv_lm = os.path.join(RESULTS_DIR, 'ml_char_lm.csv')
    csv_im = os.path.join(RESULTS_DIR, 'ml_imitation.csv')
    if os.path.exists(csv_lm) or os.path.exists(csv_im):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        if os.path.exists(csv_lm):
            df_lm = pd.read_csv(csv_lm)
            for m_name, color, marker in [
                ('Attention-CharLM (Q6)', '#2ecc71', 'o'),
                ('Bigram-MLP (No Attention)', '#e74c3c', 's'),
            ]:
                sub = df_lm[df_lm['model'] == m_name].sort_values('epoch')
                if not sub.empty:
                    ax1.plot(sub['epoch'], sub['val_perplexity'], marker=marker, color=color, linewidth=2.2, label=m_name)
            ax1.set_title("H2: Perplexity tập Validation của Character-LM (models.Attention)", fontsize=11, fontweight='bold')
            ax1.set_xlabel("Epoch huấn luyện")
            ax1.set_ylabel("Validation Perplexity (Thấp hơn là tốt hơn)")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.5)

        if os.path.exists(csv_im):
            df_im = pd.read_csv(csv_im)
            x = np.arange(len(df_im))
            colors = ['#27ae60', '#3498db', '#9b59b6'][:len(df_im)]
            bars = ax2.bar(x, df_im['megacity_score'], yerr=df_im['megacity_score_std'].fillna(0),
                           capsize=4, color=colors, width=0.45, alpha=0.88)
            ax2.set_xticks(x)
            labels = [
                f"{r['agent']}\n(Acc={r['val_action_acc']*100:.1f}%, {r['speedup_vs_teacher']:.0f}x faster)"
                for _, r in df_im.iterrows()
            ]
            ax2.set_xticklabels(labels, fontsize=8.5)
            ax2.set_title("I: Điểm số thi đấu trên MegaCity 31x21 (Teacher vs Neural Students)", fontsize=11, fontweight='bold')
            ax2.set_ylabel("Điểm số trung bình (± std) trên MegaCity")
            ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'p5_megacity_imitation_charlm.png'), dpi=200)
        plt.close()


def main():
    print("=" * 60)
    print("BẮT ĐẦU TẠO HÌNH ẢNH TRỰC QUAN CHO TOÀN BỘ THỰC NGHIỆM")
    print("=" * 60)
    plot_search()
    plot_multiagent()
    plot_reinforcement()
    plot_machinelearning()
    print("=" * 60)
    print(f"HOÀN THÀNH! Toàn bộ hình ảnh đã được lưu tại: {FIGURES_DIR}")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith('.png'):
            print(f" -> {f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
