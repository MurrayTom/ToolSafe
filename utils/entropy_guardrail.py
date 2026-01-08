import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import zip_longest

def set_academic_style():
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"], # 如果系统有 Times New Roman 可换上
        "font.size": 16,
        "legend.fontsize": 16,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

def load_data_advanced(folder_path):
    """
    一次性读取文件夹数据，返回：
    1. 每个句子的平均熵列表 (for Distribution)
    2. 对齐后的平均序列和标准差序列 (for Trend)
    """
    all_entropies_lists = []
    sentence_means = []
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    for file_name in files:
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "entropies" in data and len(data["entropies"]) > 0:
                e_list = data["entropies"]
                all_entropies_lists.append(e_list)
                sentence_means.append(np.mean(e_list))
                
    # 对齐处理
    aligned = np.array(list(zip_longest(*all_entropies_lists, fillvalue=np.nan)))
    mean_curve = np.nanmean(aligned, axis=1)
    std_curve = np.nanstd(aligned, axis=1) # 计算标准差用于画阴影
    
    return sentence_means, mean_curve, std_curve

def plot_combined_entropy_analysis(folders_dict, save_path=None):
    set_academic_style()
    # 宽度设定为单栏略宽或双栏，2x1 布局通常使用 9x3.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    
    colors = ["#4C72B0", "#C44E52", "#55A868"] # 经典学术红蓝绿
    
    for i, (model_name, folder) in enumerate(folders_dict.items()):
        color = colors[i % len(colors)]
        s_means, m_curve, s_curve = load_data_advanced(folder)
        
        # --- 左子图: Distribution (KDE + Histogram) ---
        sns.histplot(s_means, bins=25, kde=True, ax=ax1, color=color, 
                     label=model_name, element="step", stat="density", alpha=0.2)
        
        # --- 右子图: Alignment Trend (Line + Shadow) ---
        x = np.arange(len(m_curve))
        ax2.plot(x, m_curve, color=color, label=model_name, linewidth=1.5)
        # 画置信区间阴影（使用标准差）
        ax2.fill_between(x, m_curve - 0.5*s_curve, m_curve + 0.5*s_curve, 
                         color=color, alpha=0.1, edgecolor='none')

    # 修饰左图
    ax1.set_xlabel("Mean Token Entropy per Sentence")
    ax1.set_ylabel("Density")
    ax1.set_title("(a) Entropy Distribution", fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.2, linestyle='--')
    ax1.legend(frameon=False)

    # 修饰右图
    ax2.set_xlabel("Token Position (Step)")
    ax2.set_ylabel("Mean Entropy")
    ax2.set_title("(b) Token-level Entropy Trend", fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.2, linestyle='--')
    ax2.legend(frameon=False)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight', dpi=300)
        plt.savefig(f"{save_path}.png", bbox_inches='tight', dpi=300)
    plt.show()

# 调用示例
# folders = {"Model A": "./path1", "Model B": "./path2", "Model C": "./path3"}
folders = {
    "Qwen2.5-7B-IT": "./entropy_analysis_v4/qwen2.5-7b-instruct/",
    "ShieldAgent-THU": "./entropy_analysis_v4/shieldagent/",
    "TS-Guard": "./entropy_analysis_v4/ToolGuard/"
}
plot_combined_entropy_analysis(folders, save_path="./figures/entropy_guard")