import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("figures/distribution", exist_ok=True)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 生成x轴数据
x = np.linspace(-6, 6, 400)

# 定义颜色
# 可选颜色方案（参考plot_gaussian.py）：
# color = '#1f77b4'      # 蓝色 - 经典配色，适合通用场景
# color = '#2ca02c'      # 绿色 - 清新自然，适合表示正面指标
# color = '#ff7f0e'      # 橙色 - 温暖醒目，适合强调重点
# color = '#9467bd'      # 紫色 - 优雅现代，适合科技感展示
color = '#4a4a4a'  # 深灰色 - 专业稳重，适合学术图表

# 定义多峰分布函数
def multi_peak_distribution(x, peaks):
    """生成多峰分布，峰高、位置、宽度都不规则，更自然
    peaks: 峰值数量
    """
    y = np.zeros_like(x)

    # 为不同数量的峰定义不同的参数（位置、高度、宽度）
    if peaks == 2:
        peak_params = [
            {'pos': -1.5, 'height': 0.4, 'width': 1.2},
            {'pos': 2.0, 'height': 0.5, 'width': 0.8},
        ]
    elif peaks == 3:
        peak_params = [
            {'pos': -3.0, 'height': 0.45, 'width': 0.8},
            {'pos': 0.5, 'height': 0.3, 'width': 0.9},
            {'pos': 3.0, 'height': 0.5, 'width': 0.7},
        ]
    else:  # peaks == 4
        peak_params = [
            {'pos': -3.5, 'height': 0.4, 'width': 0.9},
            {'pos': -1.2, 'height': 0.7, 'width': 0.7},
            {'pos': 1.5, 'height': 0.5, 'width': 0.8},
            {'pos': 3.8, 'height': 0.3, 'width': 0.7},
        ]

    # 叠加各个峰
    for param in peak_params:
        peak = param['height'] * np.exp(-(x - param['pos'])**2 / (2 * param['width']**2))
        y += peak

    return y

# 创建三种多峰分布
peak_configs = [
    (2, 'two_peak'),
    (3, 'three_peak'),
    (4, 'four_peak')
]

for n_peaks, filename in peak_configs:
    fig, ax = plt.subplots(figsize=(8, 4))

    # 计算多峰分布
    y = multi_peak_distribution(x, n_peaks)

    # 填充曲线下方的区域
    ax.fill_between(x, y, alpha=0.3, color=color)

    # 绘制曲线
    ax.plot(x, y, color=color, linewidth=2)

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
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.8)

    # 去除边框外的边距
    plt.tight_layout()

    # 保存文件
    output_filename = f'figures/distribution/{filename}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight',
                pad_inches=0.05, transparent=True)
    plt.close()

    print(f'已保存: {output_filename}')

print('\n全部完成！生成了三种多峰分布图。')
