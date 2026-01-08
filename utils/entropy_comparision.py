import numpy as np
import os
import json
from itertools import zip_longest

import matplotlib.pyplot as plt
import seaborn as sns

def set_academic_style():
    # 使用基础样式并进行微调
    plt.style.use('seaborn-v0_8-paper') 
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"], # 兼容性好且接近 Times
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42, # 确保 PDF 字体可嵌入，避开审稿系统的字体报错
        "ps.fonttype": 42
    })

set_academic_style()


def load_and_align_entropies(folder_path):
    """
    读取文件夹下所有 json，按位置对齐所有 entropy list，
    返回一个代表该 folder 平均走势的 list。
    """
    all_lists = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    for file_name in files:
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "entropies" in data and len(data["entropies"]) > 0:
                all_lists.append(data["entropies"])
    
    if not all_lists:
        return []

    # 使用 zip_longest 进行对齐，缺失位置填充 np.nan
    # 这样可以得到一个 (max_len, num_files) 的结构
    aligned_data = np.array(list(zip_longest(*all_lists, fillvalue=np.nan)))
    
    # 在行方向上计算均值，忽略 nan
    # 得到的 mean_curve 长度等于最长的那个句子
    mean_curve = np.nanmean(aligned_data, axis=1)
    
    return mean_curve.tolist()


def plot_alignment_comparison(folders_dict, save_path=None):
    plt.figure(figsize=(4.5, 3.5))
    set_academic_style() # 使用之前定义的样式
    
    colors = ["#4C72B0", "#C44E52", "#55A868"] # 经典红蓝绿
    
    for i, (model_name, folder) in enumerate(folders_dict.items()):
        # 这里你可以选择返回 mean 和 std 两个值
        # 为了简单，这里直接画平均走势
        mean_curve = load_and_align_entropies(folder)
        x = np.arange(len(mean_curve))
        
        plt.plot(x, mean_curve, label=model_name, color=colors[i], 
                 linewidth=1.5, alpha=0.9)
        
        # 可选：如果想画阴影，可以在 load 阶段同时返回 std
        # plt.fill_between(x, mean-std, mean+std, color=colors[i], alpha=0.1)

    plt.xlabel("Token Position (Step)")
    plt.ylabel("Mean Entropy")
    plt.legend(frameon=False)
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.show()


# --- 使用示例 ---
folders = {
    "Qwen2.5-7B-IT": "./entropy_analysis_v4/qwen2.5-7b-instruct/",
    "ShieldAgent-THU": "./entropy_analysis_v4/shieldagent/",
    "ToolGuard": "./entropy_analysis_v4/ToolGuard/"
}

plot_alignment_comparison(folders, save_path="./figures/entropy_curves_comparison")