import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from sklearn import preprocessing
from pyecharts.charts import Line
from pyecharts import options as opts


def plot_heatmap(pro_seq, pro_attn, fig_path, width=50, norm=False, vmin=-0.1, vmax=0.2):
    pro_attn = pro_attn.reshape(-1)
    smooth_attn = gaussian_filter(pro_attn, sigma=1)

    # 填充空白色块数值
    blank_val = (vmin+vmax)/2 if norm else (np.max(smooth_attn)+np.min(smooth_attn))/2
    if smooth_attn.shape[0] % width != 0:
        smooth_attn = np.pad(smooth_attn, (0, width - (len(pro_seq) % width)), 'constant', constant_values=blank_val)
    smooth_attn = smooth_attn.reshape(-1, width)

    fig, ax = plt.subplots(figsize=(smooth_attn.shape[1], smooth_attn.shape[0]))

    # Plot the heatmap
    im = ax.imshow(smooth_attn)
    im.cmap = plt.get_cmap('bwr')

    # 颜色标准化
    if norm:
        im.norm = matplotlib.colors.Normalize(vmin, vmax)
    
    # 削弱颜色
    if norm==False:
        im.norm = matplotlib.colors.Normalize(np.min(smooth_attn)-np.abs(np.mean(smooth_attn)*0.5),np.max(smooth_attn)+np.abs(np.mean(smooth_attn)*0.5))

    # Create colorbar
    cbar = []
    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel('attention weight', rotation=-90, va="bottom", fontsize=20)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)

    # show y ticks
    ylabels = ['%03d'%i for i in range(1, len(pro_seq)+1, width)]
    ax.set_yticks(np.arange(smooth_attn.shape[0]))
    ax.set_yticklabels(ylabels, fontsize=20)

    # Turn spines off and create white grid.
    for key, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_yticks(np.arange(smooth_attn.shape[0] + 1) - .55, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=10, axis='y')  # grid
    ax.tick_params(which="both", top=False, bottom=False, left=False, right=False)

    # 宽高比
    ax.set_aspect(1.5)

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center", fontsize=40, family='serif', alpha=0.65)
    # kw.update(textkw)

    # Loop over the data and create a `Text` for each "pixel".
    texts = []
    cnt = 0
    for i in range(smooth_attn.shape[0]):
        for j in range(smooth_attn.shape[1]):
            # kw.update(color=textcolors[int(im.norm(pro_attn[i, j]) > 0.)])
            text = im.axes.text(j, i, pro_seq[cnt], **kw)
            texts.append(text)
            cnt += 1
            if cnt == len(pro_seq):
                # return im, cbar
                # plt.show() 
                plt.savefig(fig_path, bbox_inches='tight')
                return 




def echarts(pro_seq, attn_weight):
    x = list(pro_seq)
    x_num = list(np.arange(1, len(pro_seq) + 1, 1))
    x_data = [str(i) + '-' + aa for i, aa in zip(x_num, x)]
    y = map(float,attn_weight[:len(pro_seq)])
    y = list([round(i,3) for i in y])
    # print(y)

    line = (
        Line(init_opts=opts.InitOpts(width="800px", height="200px"))
        .add_xaxis(x_data)
        .add_yaxis(
            series_name="attention values",
            y_axis=y,
            # areastyle_opts=opts.AreaStyleOpts(opacity=0.5), # 填充颜色
            linestyle_opts=opts.LineStyleOpts(),
            label_opts=opts.LabelOpts(is_show=False),
            # is_smooth=True,
        )
        .set_global_opts(
        #         title_opts=opts.TitleOpts(
        #             title="Attention Weight Visualizaton",
        #             pos_left="center",
        #             pos_top="top",
        #         ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross"
            ),
            legend_opts=opts.LegendOpts(
                pos_left="center"
            ),
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),
                opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
            ],
            # xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
            yaxis_opts=opts.AxisOpts(
                name="values",
                type_="value",
                max_=round(max(y)+abs(min(y)), 2)
            ),
        )
        .set_series_opts(
            markarea_opts=opts.MarkAreaOpts(
                is_silent=False,
            ),
            axisline_opts=opts.AxisLineOpts(),

        )
        # .render("example.html")
    )
    return line.dump_options()

