import os
import json
import numpy as np
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

def load_and_average_entropy(folder_path):
    """
    读取文件夹下所有 json，计算每个 json 的平均熵，返回一个列表
    """
    mean_entropies = []
    # 遍历文件夹中的所有 json 文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    for file_name in files:
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 假设 json 格式为 {"entropies": [0.1, 0.5, ...]}
            if "entropies" in data and len(data["entropies"]) > 0:
                avg_e = np.mean(data["entropies"])
                mean_entropies.append(avg_e)
    
    return mean_entropies

def plot_models_comparison(folders_dict, save_path=None):
    """
    folders_dict: {"Model_Name": "folder_path"}
    """

    plt.figure(figsize=(4.5, 3.5))
    
    # 学术调色盘：深蓝、暗红、翠绿
    colors = ["#4C72B0", "#C44E52", "#55A868"] 
    
    for i, (model_name, folder) in enumerate(folders_dict.items()):
        vals = load_and_average_entropy(folder)

        # 使用 sns.kdeplot 或 histplot
        # kde=True 会画出平滑曲线，alpha 控制透明度，element="step" 让边缘更清晰
        sns.histplot(vals, bins=30, kde=True, 
                     label=model_name, 
                     color=colors[i % len(colors)],
                     alpha=0.3, 
                     element="step", 
                     stat="density", # 归一化，确保不同样本量的模型可比
                     common_norm=False)

    plt.xlabel("Mean Token Entropy per Output", labelpad=10)
    plt.ylabel("Density", labelpad=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 移除图例边框，学术味更浓
    plt.legend(frameon=False, fontsize=10)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + ".pdf", bbox_inches='tight', dpi=300)
        plt.savefig(save_path + ".png", bbox_inches='tight', dpi=300)
    plt.show()

# --- 使用示例 ---
folders = {
    "Qwen2.5-7B-IT": "./entropy_analysis_v4/qwen2.5-7b-instruct/",
    "ShieldAgent-THU": "./entropy_analysis_v4/shieldagent/",
    "ToolGuard": "./entropy_analysis_v4/ToolGuard/"
}

plot_models_comparison(folders, save_path="./figures/entropy_distribution_comparison")