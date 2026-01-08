# import json
# import matplotlib.pyplot as plt

# def plot_entropy_with_rounds(json_file, save_path, item_index):
#     # 1. 加载数据
#     with open(json_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # 获取指定的 item
#     if item_index >= len(data):
#         print(f"Error: Index {item_index} 超出范围。")
#         return
    
#     item = data[item_index]
#     entropies = item.get("entropies", [])
#     entropy_stats = item.get("entropy_stats", {})
    
#     # 2. 准备绘图数据
#     x = range(len(entropies))
#     y = entropies
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(x, y, label='Entropy', color="#b46e1f", linewidth=1.5)
    
#     # 3. 计算并绘制轮次结束的竖线
#     current_pos = 0
#     # 确保按 round_1, round_2... 的顺序读取
#     sorted_rounds = sorted(entropy_stats.keys(), key=lambda x: int(x.split('_')[1]))
    
#     for round_key in sorted_rounds:
#         length = entropy_stats[round_key]['length']
#         current_pos += length
        
#         # 绘制竖线，表示工具调用/轮次结束
#         if current_pos <= len(entropies):
#             plt.axvline(x=current_pos, color='red', linestyle='--', alpha=0.7, 
#                         label=f'End of {round_key}' if item_index == 0 else "")
#             # 在竖线旁边标注
#             plt.text(current_pos, max(y)*0.9, f'{round_key}', 
#                      rotation=90, verticalalignment='center', color='red', fontsize=9)

#     # 4. 图表装饰
#     plt.title(f'Entropy across Token Positions (Item Index: {item_index})')
#     plt.xlabel('Token Position')
#     plt.ylabel('Entropy')
#     plt.grid(True, which='both', linestyle=':', alpha=0.5)
#     plt.tight_layout()
    
#     plt.savefig(f"{save_path}.png", bbox_inches='tight')
#     plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
#     plt.show()

# # 使用示例
# plot_entropy_with_rounds('./results_logs_3.7/agentharm_harmful/Qwen2.5-14B-Instruct_react_REACT_SYSTEM_PROMPT/meta_data.json', save_path="./figures/entropy_safeflow", item_index=61)

# plot_entropy_with_rounds('./results_logs_3.7/agentharm_harmful/Qwen2.5-14B-Instruct_sec_react_REACT_SYSTEM_PROMPT_ashell-guardian/meta_data.json', save_path="./figures/entropy_safeflow_2", item_index=61)

#----------------------------------------------------------------------------------------------------

# import json
# import matplotlib.pyplot as plt

# def set_academic_style():
#     # 使用基础样式并进行微调
#     plt.style.use('seaborn-v0_8-paper') 
#     plt.rcParams.update({
#         "font.family": "serif",
#         "font.serif": ["DejaVu Serif"], # 兼容性好且接近 Times
#         "font.size": 10,
#         "axes.labelsize": 12,
#         "axes.titlesize": 12,
#         "legend.fontsize": 10,
#         "grid.alpha": 0.2,
#         "grid.linestyle": "--",
#         "axes.spines.top": False,
#         "axes.spines.right": False,
#         "pdf.fonttype": 42, # 确保 PDF 字体可嵌入，避开审稿系统的字体报错
#         "ps.fonttype": 42
#     })

# set_academic_style()

# def plot_comparison_entropy(json_path_1, json_path_2, save_path, item_index):
#     def get_data(path, idx):
#         with open(path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         item = data[idx]
#         return item.get("entropies", []), item.get("entropy_stats", {})

#     # 1. 获取两组数据
#     ent_1, stats_1 = get_data(json_path_1, item_index)
#     ent_2, stats_2 = get_data(json_path_2, item_index)

#     plt.figure(figsize=(14, 7))

#     # 2. 绘制两条熵值曲线
#     # 第一组：蓝色
#     plt.plot(range(len(ent_1)), ent_1, label='ReAct Agent w/o. Guardrail', color='#1f77b4', alpha=0.8, linewidth=1.2)
#     # 第二组：橙色
#     plt.plot(range(len(ent_2)), ent_2, label='ReAct Agent w. SafeFlow Guardrail', color='#ff7f0e', alpha=0.8, linewidth=1.2)

#     # 3. 绘制各自的轮次分割线 (Round Boundaries)
#     def draw_vlines(stats, max_val, color, prefix):
#         current_pos = 0
#         sorted_keys = sorted(stats.keys(), key=lambda x: int(x.split('_')[1]))
#         for i in range(len(sorted_keys)):
#             if i == len(sorted_keys)-1:
#                 break
#             r_key = sorted_keys[i+1]
#             print(r_key)
#             current_pos += stats[sorted_keys[i]]['length']
#             plt.axvline(x=current_pos, color=color, linestyle='--', alpha=0.4)
#             # 标注放在图顶部不同高度，避免文字重叠
#             offset = 0.95 if color == 'blue' else 0.85
#             if "after_guard" in r_key:
#                 text = "Guardrail Feedback"
#             else:
#                 text = "Tool Calling"
#             plt.text(current_pos, max_val * offset, text, 
#                      rotation=0, color=color, fontsize=8, verticalalignment='top')

#     # 计算全局最大熵值以确定文字标注高度
#     max_y = max(max(ent_1) if ent_1 else 0, max(ent_2) if ent_2 else 0)
    
#     draw_vlines(stats_1, max_y, 'blue', 'A')
#     draw_vlines(stats_2, max_y, 'red', 'B')

#     # 4. 图表装饰
#     #plt.title(f'Entropy Comparison at Item Index: {item_index}', fontsize=14)
#     plt.xlabel('Token Position', fontsize=12)
#     plt.ylabel('Entropy', fontsize=12)
#     plt.legend(loc='upper right')
#     plt.grid(True, which='both', linestyle=':', alpha=0.3)
    
#     # 自动调整布局，防止文字超出边界
#     plt.tight_layout()
    
#     # 5. 保存并展示
#     plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
#     plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
#     plt.show()






import json
import matplotlib.pyplot as plt
import seaborn as sns

def set_academic_style():
    # 使用学术论文常用配置
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"], # 优先使用 Times New Roman
        "font.size": 11,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

def plot_comparison_entropy(json_path_1, json_path_2, save_path, item_index):
    set_academic_style()
    
    def get_data(path, idx):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        item = data[idx]
        return item.get("entropies", []), item.get("entropy_stats", {})

    ent_1, stats_1 = get_data(json_path_1, item_index)
    ent_2, stats_2 = get_data(json_path_2, item_index)

    # 1. 颜色选择：使用更具学术感的对比色
    # 蓝色 (ReAct) vs 红色/深橙色 (SafeFlow)
    color_1 = '#4C72B0' # Muted Blue
    color_2 = '#C44E52' # Muted Red
    
    fig, ax = plt.subplots(figsize=(12, 5)) # 稍微收窄高度，更符合论文分栏图比例

    # 2. 绘制熵值曲线 - 稍微加粗，增加辨识度
    ax.plot(range(len(ent_1)), ent_1, label='ReAct Agent (Baseline)', 
            color=color_1, alpha=0.85, linewidth=1.5, zorder=2)
    ax.plot(range(len(ent_2)), ent_2, label='ReAct Agent w/ SafeFlow', 
            color=color_2, alpha=0.85, linewidth=1.5, zorder=3)

    # 3. 绘制分割线和横向文字
    def draw_vlines(stats, color, prefix_label, y_pos_ratio):
        current_pos = 0
        sorted_keys = sorted(stats.keys(), key=lambda x: int(x.split('_')[1]))
        
        # 仅遍历到倒数第二个元素计算位置
        for i in range(len(sorted_keys) - 1):
            current_pos += stats[sorted_keys[i]]['length']
            next_key = sorted_keys[i+1]
            
            # 画竖线：调低 alpha 避免干扰数据
            ax.axvline(x=current_pos, color=color, linestyle=':', alpha=0.75, linewidth=2, zorder=1)
            
            # 确定文本
            text = "Guardrail" if "after_guard" in next_key else "Tool Call"
            
            # 文本排布逻辑：
            # 使用 ax.annotate 可以更精准控制位置，避免覆盖
            # ax.text(current_pos, ax.get_ylim()[1] * y_pos_ratio, text,
            #         color=color, fontsize=9, fontweight='bold',
            #         ha='center', va='bottom',
            #         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

            ax.text(current_pos, 
                ax.get_ylim()[1] * y_pos_ratio,             # 直接指定在 Y 轴的具体高度
                text,
                color=color, 
                fontsize=9, 
                fontweight='bold',
                ha='center', 
                va='center',         # 垂直居中对齐坐标点
                rotation=0,          # 保持横向
                bbox=dict(facecolor='white', 
                          edgecolor=color,   # 给文字框加一个细边框增加学术感
                          linewidth=0.5,
                          alpha=0.9,         # 增加背景不透明度
                          pad=1.5))

    # 设置 Y 轴范围，留出顶部空间放文字
    max_val = max(max(ent_1), max(ent_2))
    ax.set_ylim(-0.1, max_val * 1.3) 

    # 绘制两组标注，y_pos_ratio 错开防止重叠
    draw_vlines(stats_1, color_1, "A", 0.85) # 基准线的标注在稍低位置
    draw_vlines(stats_2, color_2, "B", 1.05) # SafeFlow的标注在更高位置

    # 4. 细节修饰
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Entropy')
    
    # 图例修饰：去掉边框，水平排列或放在角落
    # ax.legend(loc='upper center', frameon=True, shadow=False, facecolor='white', edgecolor='#cccccc')
    
    # 将图例放在图表外部的上方，避免遮挡任何曲线或标签
    ax.legend(
        loc='upper right', 
        #bbox_to_anchor=(0.65, 0.9), # (x, y) 坐标，0.5代表水平居中，1.15代表在图表上方
        ncol=2,                      # 设置为2列，水平排列，节省垂直空间
        frameon=True, 
        shadow=False, 
        facecolor='white', 
        edgecolor='#cccccc'
    )

    plt.tight_layout()
    
    # 5. 保存
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_path}.png", bbox_inches='tight', dpi=300)
    plt.show()

# 调用部分保持不变...

# 调用
path_a = './results_logs_3.7/asb_OPI/Qwen2.5-14B-Instruct_react_REACT_SYSTEM_PROMPT/meta_data.json'
path_b = './results_logs_3.7/asb_OPI/Qwen2.5-14B-Instruct_sec_react_REACT_SYSTEM_PROMPT_ashell-guardian/meta_data.json'

plot_comparison_entropy(path_a, path_b, save_path="./figures/entropy_comparison_asb_opi_10", item_index=89)