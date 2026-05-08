import matplotlib.pyplot as plt
import numpy as np

# 1. 设置 IEEE 学术图表全局样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 2. 从 CJA Table 7 提取的数据
labels = ['1-vs-1', '2-vs-2', '5-vs-5']

# HRL (Hierarchical) 胜率与标准误 (Mean & SE)
hrl_means = [0.446, 0.653, 0.864]
hrl_errors = [0.062, 0.112, 0.131]

# E2E (End-to-End) 胜率与标准误 (Mean & SE)
e2e_means = [0.062, 0.000, 0.000]
e2e_errors = [0.080, 0.000, 0.000]

# 3. 设置柱状图的 X 轴位置和宽度
x = np.arange(len(labels))
width = 0.35  # 柱子宽度

# 4. 创建画布 (大小调整为适合复合图的左半边)
fig, ax = plt.subplots(figsize=(5, 4.5))

# 5. 绘制带有误差棒的柱状图
rects1 = ax.bar(x - width/2, hrl_means, width, yerr=hrl_errors, 
                label='Hierarchical (HRL)', color='#1f77b4', 
                edgecolor='black', capsize=5, zorder=3)
                
rects2 = ax.bar(x + width/2, e2e_means, width, yerr=e2e_errors, 
                label='End-to-End (E2E)', color='#d62728', 
                edgecolor='black', capsize=5, zorder=3)

# 6. 图表细节美化
ax.set_ylabel('Success Rate')
ax.set_ylim(0, 1.1) # 稍微再调高一点顶部空间，给浮在误差棒上方的文字留足位置
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# 7. 强制标注数值的核心函数（已修复误差棒重叠问题）
def autolabel(rects, means, errors):
    """
    现在额外接收 errors 数组，确保文字锚点在误差棒之上。
    """
    for rect, mean, err in zip(rects, means, errors):
        height = rect.get_height()
        
        if height > 0:
            # 核心修改：如果是非 0 值，基准点设在误差棒的最顶端 (height + err)
            y_pos = height + err
            offset = 5  # 距离误差棒横线向上偏移 5 像素
        else:
            # 如果是 0 值，基准点设在 0
            y_pos = 0
            offset = 3  # 距离 X 轴向上偏移 3 像素
            
        ax.annotate(f'{mean:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                    xytext=(0, offset),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

# 这里调用时，把对应的 errors 数组也传进去
autolabel(rects1, hrl_means, hrl_errors)
autolabel(rects2, e2e_means, e2e_errors)

# 8. 输出保存为高精度 PDF
fig.tight_layout()
output_filename = 'hrl_vs_e2e_bar.pdf'
plt.savefig(output_filename, format='pdf', bbox_inches='tight')
print(f"图表已成功生成并保存为: {output_filename}")

plt.show()