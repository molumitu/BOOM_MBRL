import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("figures/distribution", exist_ok=True)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 生成高斯分布数据
x = np.linspace(-4, 4, 200)
y = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

# 定义四种优雅的颜色方案
colors = [
    ('blue', '#1f77b4'),      # 经典蓝
    ('green', '#2ca02c'),     # 深绿
    ('orange', '#ff7f0e'),    # 橙色
    ('purple', '#9467bd'),    # 紫色
]

# 为每种颜色创建并保存图形
for color_name, color_hex in colors:
    fig, ax = plt.subplots(figsize=(8, 4))

    # 填充曲线下方的区域
    ax.fill_between(x, y, alpha=0.3, color=color_hex)

    # 绘制曲线
    ax.plot(x, y, color=color_hex, linewidth=2)

    # 只保留x轴
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 设置x轴位置在底部
    ax.spines['bottom'].set_position('zero')

    # 去除刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])

    # 去除刻度线
    ax.tick_params(left=False, bottom=False)

    # 设置坐标范围
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.45)

    # 去除边框外的边距
    plt.tight_layout()

    # 保存文件
    filename = f'figures/distribution/gaussian_distribution_{color_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                pad_inches=0.05, transparent=True)
    plt.close()

    print(f'已保存: {filename}')

print('\n全部完成！生成了四个颜色的高斯分布图。')
