import plotly.express as px
import numpy as np
import pandas as pd

def get_consistent_colors(labels_dict):
    """
    为每个 Cluster ID 生成固定的颜色映射。
    Outliers (-1) 永远是灰色。
    """
    # 获取 Plotly 默认的高对比度色盘
    palette = px.colors.qualitative.G10 + px.colors.qualitative.Plotly

    color_map = {}
    ids = sorted([k for k in labels_dict.keys() if k != -1])

    # 映射正常类
    for i, cluster_id in enumerate(ids):
        # 循环使用色盘
        color_map[labels_dict[cluster_id]] = palette[i % len(palette)]

    # 映射噪音类
    if -1 in labels_dict:
        color_map[labels_dict[-1]] = "#d3d3d3"  # 浅灰色

    return color_map


def plot_paper_map(df, cluster_labels):
    """
    绘制交互式散点图，使用固定颜色映射
    """
    # 1. 映射 Label
    if isinstance(cluster_labels, dict):
        df['topic_label'] = df['cluster'].map(cluster_labels)
    else:
        df['topic_label'] = df['cluster'].astype(str)

    # 2. 大小处理
    df['citations'] = df['citations'].fillna(0)
    df['size_log'] = np.log1p(df['citations']) * 3 + 3

    # 3. 按照 Cluster ID 排序，保证图例顺序一致
    df = df.sort_values(by='cluster')

    # 4. 生成固定颜色映射 (Map label string -> hex color)
    color_map = get_consistent_colors(cluster_labels)

    # 5. 绘图
    fig = px.scatter(
        df,
        x='x', y='y',
        color='topic_label',
        size='size_log',
        hover_data={
            'x': False, 'y': False, 'size_log': False,
            'topic_label': True, 'title': True, 'venue': True,
            'year': True, 'citations': True
        },
        title="<b>Semantic Research Landscape</b>",
        color_discrete_map=color_map,  # <--- 关键：强制指定颜色
        # 确保图例顺序按照 Cluster ID 排列 (Plotly 默认按出现顺序)
        category_orders={"topic_label": [cluster_labels[k] for k in sorted(cluster_labels.keys())]}
    )

    # 6. 美化
    fig.update_layout(
        plot_bgcolor='rgba(240,242,246, 0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        legend=dict(
            title="Research Topics",
            orientation="v",
            yanchor="top", y=1,
            xanchor="left", x=1.02
        ),
        font=dict(family="Inter, sans-serif"),
        margin=dict(t=50, l=0, r=0, b=0)
    )

    fig.update_traces(
        marker=dict(line=dict(width=0.5, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )

    return fig