import json
import matplotlib.pyplot as plt
import seaborn as sns

def set_academic_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 20,
        "axes.labelsize": 12,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

def draw_single_plot(ax, json_path_1, json_path_2, item_index, title=""):
    """在指定的 ax 上绘制单组对比图"""
    
    def get_data(path, idx):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        item = data[idx]
        return item.get("entropies", []), item.get("entropy_stats", {})

    ent_1, stats_1 = get_data(json_path_1, item_index)
    ent_2, stats_2 = get_data(json_path_2, item_index)

    color_1, color_2 = '#4C72B0', '#C44E52'
    
    # 绘制熵值曲线
    ax.plot(range(len(ent_1)), ent_1, label='ReAct Agent (Baseline)', 
            color=color_1, alpha=0.85, linewidth=1.2, zorder=2)
    ax.plot(range(len(ent_2)), ent_2, label='ReAct Agent w. TS-Flow', 
            color=color_2, alpha=0.85, linewidth=1.2, zorder=3)

    # 内部辅助函数绘制垂直分割线
    def draw_vlines(stats, color, y_pos_ratio):
        current_pos = 0
        sorted_keys = sorted(stats.keys(), key=lambda x: int(x.split('_')[1]))
        for i in range(len(sorted_keys) - 1):
            current_pos += stats[sorted_keys[i]]['length']
            next_key = sorted_keys[i+1]
            ax.axvline(x=current_pos, color=color, linestyle=':', alpha=0.6, linewidth=1.5, zorder=1)
            
            text = "Guardrail" if "after_guard" in next_key else "Tool Call"
            ax.text(current_pos, ax.get_ylim()[1] * y_pos_ratio, text,
                    color=color, fontsize=12, fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor=color, linewidth=0.5, alpha=0.9, pad=1.2))

    # 动态调整 Y 轴范围
    max_val = max(max(ent_1) if ent_1 else 0, max(ent_2) if ent_2 else 0)
    ax.set_ylim(-0.1, max_val * 1.4) 

    draw_vlines(stats_1, color_1, 0.82)
    draw_vlines(stats_2, color_2, 0.92)

    ax.set_ylabel('Entropy')
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold')

def plot_combined_entropy(configs, save_path):
    set_academic_style()
    
    # 创建 2行 1列 的画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=False)

    # 绘制第一个子图 (AgentHarm)
    draw_single_plot(ax1, configs[0]['path_a'], configs[0]['path_b'], 
                     configs[0]['idx'], title="(a) AgentHarm - Malicious Agentic Task")
    
    # 绘制第二个子图 (ASB-OPI)
    draw_single_plot(ax2, configs[1]['path_a'], configs[1]['path_b'], 
                     configs[1]['idx'], title="(b) ASB - Prompt Injection")

    ax2.set_xlabel('Token Position')

    # 全局图例：放在整个画布的最上方
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=2, frameon=True, facecolor='white', edgecolor='#cccccc')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为顶部图例留出空间
    
    # 保存
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_path}.png", bbox_inches='tight', dpi=300)
    plt.show()

# --- 执行部分 ---

configs = [
    {
        "path_a": './results_logs_3.7/agentharm_harmful/Qwen2.5-14B-Instruct_react_REACT_SYSTEM_PROMPT/meta_data.json',
        "path_b": './results_logs_3.7/agentharm_harmful/Qwen2.5-14B-Instruct_sec_react_REACT_SYSTEM_PROMPT_ashell-guardian/meta_data.json',
        "idx": 89
    },
    {
        "path_a": './results_logs_3.7/asb_OPI/Qwen2.5-14B-Instruct_react_REACT_SYSTEM_PROMPT/meta_data.json',
        "path_b": './results_logs_3.7/asb_OPI/Qwen2.5-14B-Instruct_sec_react_REACT_SYSTEM_PROMPT_ashell-guardian/meta_data.json',
        "idx": 89
    }
]

plot_combined_entropy(configs, save_path="./figures/entropy_comparison_combined")