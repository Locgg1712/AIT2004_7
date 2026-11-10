"""
experiments/plot_all_figures.py
-------------------------------
Tạo toàn bộ biểu đồ, hình ảnh trực quan chất lượng cao cho:
1. Project 1: Mở rộng kích thước mê cung (Search scaling) & Bùng nổ tổ hợp Multi-Food
2. Project 2: Hiệu suất cắt tỉa Alpha-Beta & Đấu trường đối kháng 3 Ghosts
3. Project 3:
   - Đường cong hội tụ Bellman: Standard Value Iteration vs Prioritized Sweeping trên Gridworld 30x30
   - Khả năng sống sót & Điểm số: Tabular Q-Learning vs Approximate Q-Learning
   - Bản đồ nhiệt (Heatmap) giá trị trạng thái V*(s) trên lưới MDP 10x10
4. Project 5:
   - Đồ thị xấp xỉ hàm sin(x) của RegressionModel (Ground truth vs Prediction)
   - Heatmap ma trận chú ý (Causal Masked Self-Attention Weight Matrix)
   - Biểu đồ tăng tốc Vector hóa Convolve 2D & Độ phức tạp bậc hai bộ nhớ của Attention
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # Render không cần GUI window
import matplotlib.pyplot as plt
import pandas as pd

# Thiết lập style thẩm mỹ hiện đại
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['axes.edgecolor'] = '#cccccc'
matplotlib.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, 'experiments')
RESULTS_DIR = os.path.join(EXP_DIR, 'results')
FIGURES_DIR = os.path.join(EXP_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Project 1 Plots: Search Scaling & Food Scaling
# -------------------------------------------------------------
def plot_search():
    print("[1/5] Vẽ biểu đồ Project 1 (Search)...")
    csv_scaling = os.path.join(RESULTS_DIR, 'search_scaling.csv')
    if os.path.exists(csv_scaling):
        df = pd.read_csv(csv_scaling)
        # Lọc bản đồ unit cost
        df_unit = df[df['terrain'] == 'unit']
        agg = df_unit.groupby(['size', 'algo'])[['expanded', 'cost', 'time_ms']].mean().reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        # Biểu đồ số nút mở rộng
        for alg in ['DFS', 'BFS', 'UCS', 'A*-Manhattan']:
            sub = agg[agg['algo'] == alg].sort_values('size')
            if not sub.empty:
                ax1.plot(sub['size'], sub['expanded'], marker='o', linewidth=2, label=alg)
        ax1.set_title("Search: Số nút mở rộng theo kích thước ma trận (Unit Cost)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Kích thước ma trận (N x N)")
        ax1.set_ylabel("Số nút mở rộng (Nodes Expanded)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Biểu đồ chi phí đường đi
        for alg in ['DFS', 'BFS', 'UCS', 'A*-Manhattan']:
            sub = agg[agg['algo'] == alg].sort_values('size')
            if not sub.empty:
                ax2.plot(sub['size'], sub['cost'], marker='s', linewidth=2, label=alg)
        ax2.set_title("Search: Chi phí đường đi tìm được (Suboptimality của DFS)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Kích thước ma trận (N x N)")
        ax2.set_ylabel("Chi phí đường đi (Cost)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'p1_search_scaling.png'), dpi=200)
        plt.close()

    csv_food = os.path.join(RESULTS_DIR, 'search_food.csv')
    if os.path.exists(csv_food):
        df_f = pd.read_csv(csv_food)
        agg_f = df_f[df_f['solved'] == True].groupby(['k', 'algo'])[['expanded', 'time_ms']].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        for alg in ['UCS', 'A*-foodHeuristic', 'ClosestDot']:
            sub = agg_f[agg_f['algo'] == alg].sort_values('k')
            if not sub.empty:
                ax.plot(sub['k'], sub['expanded'], marker='^', linewidth=2.2, label=alg)
        ax.set_yscale('log')
        ax.set_title("Project 1: Bùng nổ tổ hợp Multi-Food Search (Log scale)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Số lượng viên thức ăn (k)")
        ax.set_ylabel("Số nút mở rộng (Log Scale)")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'p1_food_combinatorics.png'), dpi=200)
        plt.close()

# -------------------------------------------------------------
# 2. Project 2 Plots: AlphaBeta Pruning & Arena Games
# -------------------------------------------------------------
def plot_multiagent():
    print("[2/5] Vẽ biểu đồ Project 2 (Multi-Agent)...")
    csv_prune = os.path.join(RESULTS_DIR, 'multiagent_pruning.csv')
    if os.path.exists(csv_prune):
        df_p = pd.read_csv(csv_prune)
        agg_p = df_p.groupby(['map', 'depth'])[['minimax_nodes', 'alphabeta_nodes', 'speedup']].mean().reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        x = np.arange(len(agg_p))
        width = 0.35
        labels = [f"{row['map']}\nd={row['depth']}" for _, row in agg_p.iterrows()]

        ax1.bar(x - width/2, agg_p['minimax_nodes'], width, label='Minimax Nodes', color='#e74c3c', alpha=0.85)
        ax1.bar(x + width/2, agg_p['alphabeta_nodes'], width, label='AlphaBeta Nodes', color='#2ecc71', alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylabel("Số lượng nút mở rộng mỗi lượt (Nodes/Move)")
        ax1.set_title("Project 2: Hiệu quả cắt tỉa số nút của Alpha-Beta", fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Hệ số tăng tốc Speedup
        ax2.bar(labels, agg_p['speedup'], color='#3498db', width=0.45, alpha=0.85)
        ax2.axhline(1.0, color='red', linestyle='--', label='Baseline (1.0x)')
        ax2.set_ylabel("Hệ số tăng tốc (Speedup Factor)")
        ax2.set_title("Project 2: Tốc độ tăng tốc của Alpha-Beta vs Minimax", fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'p2_alphabeta_pruning.png'), dpi=200)
        plt.close()

    csv_games = os.path.join(RESULTS_DIR, 'multiagent_games.csv')
    if os.path.exists(csv_games):
        df_g = pd.read_csv(csv_games)
        arena_df = df_g[df_g['map'].str.contains('arena', case=False, na=False)].groupby('agent')[['win', 'score', 'food_cleared']].mean().reset_index()

        if not arena_df.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
            ax1.bar(arena_df['agent'], arena_df['score'], color=['#95a5a6', '#e67e22', '#27ae60'], width=0.45)
            ax1.set_title("Đấu trường Arena 41x21 (3 Ghosts): Điểm số trung bình", fontsize=11, fontweight='bold')
            ax1.set_ylabel("Điểm số trung bình (Average Score)")
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2.bar(arena_df['agent'], arena_df['food_cleared'], color=['#95a5a6', '#e67e22', '#27ae60'], width=0.45)
            ax2.set_title("Đấu trường Arena 41x21: Số viên thức ăn ăn được", fontsize=11, fontweight='bold')
            ax2.set_ylabel("Food Cleared")
            ax2.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'p2_arena_performance.png'), dpi=200)
            plt.close()

# -------------------------------------------------------------
# 3. Project 3 Plots: RL MDP Convergence & Pacman Generalization
# -------------------------------------------------------------
def plot_reinforcement():
    print("[3/5] Vẽ biểu đồ Project 3 (Reinforcement Learning)...")
    # 1. Đường cong hội tụ Bellman
    csv_curves = os.path.join(RESULTS_DIR, 'rl_mdp_curves.csv')
    if os.path.exists(csv_curves):
        df_c = pd.read_csv(csv_curves)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for size, ax in zip([10, 30], [ax1, ax2]):
            sub = df_c[df_c['size'] == size]
            std = sub[sub['algo'] == 'ValueIteration']
            ps = sub[sub['algo'] == 'PrioritizedSweeping']

            if not std.empty:
                ax.plot(std['backups'], std['max_err'], label='Standard Value Iteration', color='#e74c3c', linewidth=2)
            if not ps.empty:
                ax.plot(ps['backups'], ps['max_err'], label='Prioritized Sweeping', color='#2ecc71', linewidth=2)
            ax.set_yscale('log')
            ax.set_title(f"MDP {size}x{size}: Sai số Bellman cực đại theo số Backups", fontsize=11, fontweight='bold')
            ax.set_xlabel("Số phép cập nhật Bellman (Backups)")
            ax.set_ylabel("Max Bellman Error (Log scale)")
            ax.legend()
            ax.grid(True, which='both', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'p3_bellman_convergence.png'), dpi=200)
        plt.close()

    # 2. Hiệu năng Q-Learning Tabular vs Approximate
    csv_pacman = os.path.join(RESULTS_DIR, 'rl_pacman.csv')
    if os.path.exists(csv_pacman):
        df_p = pd.read_csv(csv_pacman)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        for agent, color, marker, label in [
            ('Tabular Q-Learning', '#e74c3c', 'o', 'Tabular Q-Learning'),
            ('Approximate Q-Learning (SimpleExtractor)', '#27ae60', 's', 'Approximate Q-Learning')
        ]:
            sub = df_p[df_p['agent'] == agent].sort_values('n_train')
            if not sub.empty:
                ax1.plot(sub['n_train'], sub['win_rate'], marker=marker, color=color, linewidth=2.2, label=label)
                ax2.plot(sub['n_train'], sub['avg_score'], marker=marker, color=color, linewidth=2.2, label=label)

        ax1.set_title("Project 3: Tỷ lệ thắng (Win Rate) theo số ván Train", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Số ván huấn luyện (Num Training Episodes)")
        ax1.set_ylabel("Tỷ lệ thắng (Win Rate)")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2.set_title("Project 3: Điểm số trung bình (Score) theo số ván Train", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Số ván huấn luyện (Num Training Episodes)")
        ax2.set_ylabel("Điểm trung bình (Average Score)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'p3_qlearning_generalization.png'), dpi=200)
        plt.close()

    # 3. Heatmap giá trị trạng thái V*(s) trên lưới MDP 10x10 trực quan
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
            val_matrix[9 - y, x] = val # Đảo y để góc (0,0) ở góc dưới bên trái

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
# 4. Project 5 Plots: Regression sin(x), Convolve & Attention
# -------------------------------------------------------------
def plot_machinelearning():
    print("[4/5] Vẽ biểu đồ Project 5 (Machine Learning & Deep Learning)...")
    sys.path.insert(0, os.path.join(BASE_DIR, 'machinelearning'))
    import torch
    from torch import tensor
    from models import RegressionModel, Attention, Convolve
    from train import train_regression
    import autograder

    # 1. Vẽ đồ thị khớp hàm sin(x) của RegressionModel
    import backend
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
    attn_layer = Attention(layer_size=16, block_size=32)
    sample_input = torch.randn(1, 16, 16) # B=1, T=16, C=16
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    if os.path.exists(csv_conv):
        df_conv = pd.read_csv(csv_conv)
        sub_k3 = df_conv[df_conv['kernel_size'] == 3].sort_values('matrix_size')
        sub_k5 = df_conv[df_conv['kernel_size'] == 5].sort_values('matrix_size')

        ax1.plot(sub_k3['matrix_size'], sub_k3['speedup'], marker='o', color='#2980b9', linewidth=2.2, label='Kernel 3x3')
        ax1.plot(sub_k5['matrix_size'], sub_k5['speedup'], marker='s', color='#e67e22', linewidth=2.2, label='Kernel 5x5')
        ax1.set_title("Project 5: Hệ số tăng tốc của Vectorized Convolve 2D", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Kích thước ma trận ảnh (H x W)")
        ax1.set_ylabel("Hệ số tăng tốc (Speedup vs Naive Loop)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

    if os.path.exists(csv_attn):
        df_attn = pd.read_csv(csv_attn).sort_values('seq_len_T')
        ax2.plot(df_attn['seq_len_T'], df_attn['attn_matrix_kb'], marker='^', color='#8e44ad', linewidth=2.2, label='Attention Matrix Memory')
        ax2.set_title("Project 5: Chi phí bộ nhớ O(T^2) của Attention Matrix", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Độ dài chuỗi Sequence Length (T)")
        ax2.set_ylabel("Dung lượng bộ nhớ (KB)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'p5_scaling_performance.png'), dpi=200)
    plt.close()

# -------------------------------------------------------------
# 5. Tổng hợp hình ảnh trực quan vào thư viện
# -------------------------------------------------------------
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
