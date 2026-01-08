# import matplotlib.pyplot as plt
# import numpy as np

# # Data: F1 scores
# methods = ["Qwen2.5-7B-IT", "+ SFT", "+ SFT + RL", "+ RL only"]
# agentharm_f1 = [80.17, 83.55, 87.26, 90.16]
# agentdojo_f1 = [50.47, 52.09, 69.94, 86.18]

# x = np.arange(len(methods))
# width = 0.35

# plt.figure(figsize=(5, 3))
# plt.bar(x - width / 2, agentharm_f1, width, label="AgentHarm-traj")
# plt.bar(x + width / 2, agentdojo_f1, width, label="AgentDojo-traj")

# plt.ylabel("F1 Score")
# plt.xticks(x, methods)
# plt.ylim(40, 95)
# plt.legend()
# plt.tight_layout()

# # 保存
# plt.savefig("./figures/ablation_training_strategy.pdf", bbox_inches="tight")
# plt.savefig("./figures/ablation_training_strategy.png", dpi=300, bbox_inches="tight")

# plt.show()

################################################################################
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import cm

# # Models & datasets
# models = ["Qwen2.5-7B-IT", "+ SFT", "+ SFT + RL", "+ RL only"]
# datasets = ["AgentHarm-traj", "AgentDojo-traj"]

# # F1 scores（示例，替换 RL only）
# agentharm_f1 = [80.17, 83.55, 87.26, 90.16]
# agentdojo_f1 = [50.47, 52.09, 69.94, 86.18]

# data = [agentharm_f1, agentdojo_f1]

# num_datasets = len(datasets)
# num_models = len(models)

# x = np.arange(num_datasets)
# width = 0.18

# # 色系：蓝 & 橙（由浅到深）
# blue_shades = cm.Blues(np.linspace(0.4, 0.85, num_models))
# orange_shades = cm.Oranges(np.linspace(0.4, 0.85, num_models))

# plt.figure(figsize=(5, 3))

# for i in range(num_models):
#     plt.bar(
#         x[0] + (i - (num_models - 1) / 2) * width,
#         agentharm_f1[i],
#         width,
#         color=blue_shades[i],
#         label=models[i] if x[0] == 0 else None
#     )
#     plt.bar(
#         x[1] + (i - (num_models - 1) / 2) * width,
#         agentdojo_f1[i],
#         width,
#         color=orange_shades[i]
#     )

# plt.ylabel("F1 Score")
# plt.xticks(x, datasets)
# plt.ylim(40, 105)

# # 只保留方法图例（颜色语义在 caption 里解释）
# plt.legend(title="Training Strategy", ncol=2, fontsize=8)

# plt.tight_layout()

# plt.savefig("./figures/ablation_training_strategy.pdf", bbox_inches="tight")
# plt.savefig("./figures/ablation_training_strategy.png", dpi=300, bbox_inches="tight")

# plt.show()

###################################################################

# import matplotlib.pyplot as plt
# import numpy as np

# # Models & datasets
# models = ["Qwen2.5-7B-IT", "+ SFT", "+ SFT + RL", "+ RL only"]
# datasets = ["AgentHarm-traj", "AgentDojo-traj"]

# # F1 scores
# agentharm_f1 = [80.17, 83.55, 87.26, 90.16]
# agentdojo_f1 = [50.47, 52.09, 69.94, 86.18]

# x = np.arange(len(datasets))
# width = 0.18

# # 解决方案：使用统一的一组颜色
# # 建议使用渐变色（代表方法迭代）或高对比色
# colors = ['#DEEBF7', '#9ECAE1', '#4292C6', '#084594'] # 统一使用由浅到深的蓝色

# plt.figure(figsize=(6, 4)) # 稍微调宽一点，避免标签拥挤

# for i in range(len(models)):
#     # 提取两个数据集在当前模型下的得分
#     model_scores = [agentharm_f1[i], agentdojo_f1[i]]
    
#     plt.bar(
#         x + (i - (len(models) - 1) / 2) * width,
#         model_scores,
#         width,
#         color=colors[i],
#         label=models[i],
#         edgecolor='white', # 增加白边让柱状图更清晰
#         linewidth=0.5
#     )

# plt.ylabel("F1 Score")
# plt.xticks(x, datasets)
# plt.ylim(40, 100) # F1通常上限是100

# # 优化图例：放在上方或侧方
# #plt.legend(title="Training Strategy", ncol=2, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 1.15))
# plt.legend(title="Training Strategy", ncol=2, fontsize=9)

# #plt.grid(axis='y', linestyle='--', alpha=0.3) # 增加参考虚线
# plt.tight_layout()

# plt.savefig("./figures/ablation_training_strategy.pdf", bbox_inches="tight")
# plt.savefig("./figures/ablation_training_strategy.png", dpi=300, bbox_inches="tight")

# plt.show()
# #---------------------------------------------------------------


# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import cm

# # Data: F1 scores
# methods = ["Qwen2.5-7B-IT", "RL w. Single-task", "RL w. Multi-task"]
# datasets = ["AgentHarm-traj", "AgentDojo-traj"]

# # 原始数据
# agentharm_f1 = [80.17, 81.49, 90.16]
# agentdojo_f1 = [50.47, 74.74, 86.18]

# x = np.arange(len(datasets))
# width = 0.22  # 方法比之前少一个，宽度可以稍微调大一点

# # 使用橙色渐变色系
# # 颜色从浅到深代表策略的演进
# colors = ['#FEE6CE', '#FDAE6B', '#E6550D'] 

# plt.figure(figsize=(5, 3.5))

# for i in range(len(methods)):
#     # 提取该方法在两个数据集上的得分
#     model_scores = [agentharm_f1[i], agentdojo_f1[i]]
    
#     plt.bar(
#         x + (i - (len(methods) - 1) / 2) * width,
#         model_scores,
#         width,
#         color=colors[i],
#         label=methods[i],
#         edgecolor='white',
#         linewidth=0.5
#     )

# plt.ylabel("F1 Score")
# plt.xticks(x, datasets)
# plt.ylim(40, 100)

# # 图例保持一致风格
# plt.legend(title="Reward Strategy", ncol=1, fontsize=9)

# # 如果需要网格线可以取消注释
# # plt.grid(axis='y', linestyle='--', alpha=0.3)

# plt.tight_layout()

# # 保存
# plt.savefig("./figures/ablation_reward.pdf", bbox_inches="tight")
# plt.savefig("./figures/ablation_reward.png", dpi=300, bbox_inches="tight")

# plt.show()


#-------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np

# --- 全局设置 ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"

# --- 数据准备 ---
datasets = ["AgentHarm-Traj", "AgentDojo-Traj"]
x = np.arange(len(datasets))

# 子图1数据 (Training Strategy)
models_1 = ["Qwen2.5-IT", "+ SFT", "+ SFT + RL", "+ RL only"]
ah_f1_1 = [80.17, 83.55, 87.26, 90.16]
ad_f1_1 = [50.47, 52.09, 69.94, 86.18]
colors_1 = ['#DEEBF7', '#9ECAE1', '#4292C6', '#084594']

# 子图2数据 (Reward Strategy)
methods_2 = ["Qwen2.5-IT", "RL w. Single-task", "RL w. Multi-task"]
ah_f1_2 = [80.17, 81.49, 90.16]
ad_f1_2 = [50.47, 74.74, 86.18]
colors_2 = ['#FEE6CE', '#FDAE6B', '#E6550D']

# --- 开始绘图 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2)) # 调整比例适合并排放置

# --- 左图: Training Strategy ---
width1 = 0.15
for i in range(len(models_1)):
    scores = [ah_f1_1[i], ad_f1_1[i]]
    ax1.bar(x + (i - (len(models_1)-1)/2) * width1, scores, width1, 
            label=models_1[i], color=colors_1[i], edgecolor='white', linewidth=0.5)

ax1.set_ylabel("F1 Score", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets)
ax1.set_ylim(40, 105)
ax1.legend(title="Training Strategy", fontsize=8, ncol=2, loc='upper left')
ax1.set_title("(a) Training Stages", fontsize=12, pad=10)

# --- 右图: Reward Strategy ---
width2 = 0.15
for i in range(len(methods_2)):
    scores = [ah_f1_2[i], ad_f1_2[i]]
    ax2.bar(x + (i - (len(methods_2)-1)/2) * width2, scores, width2, 
            label=methods_2[i], color=colors_2[i], edgecolor='white', linewidth=0.5)

ax2.set_xticks(x)
ax2.set_xticklabels(datasets)
ax2.set_ylim(40, 105)
ax2.legend(title="Reward Strategy", fontsize=8, loc='upper left')
ax2.set_title("(b) Reward Designs", fontsize=12, pad=10)

# --- 自动调整布局 ---
plt.tight_layout()

# 保存
plt.savefig("./figures/combined_ablation.pdf", bbox_inches="tight")
plt.savefig("./figures/combined_ablation.png", dpi=300, bbox_inches="tight")

plt.show()