"""
Function to reduce and plot data in 2 or 3 dimensions

Example of usage:

>>> from sklearn.datasets import make_blobs

>>> X, y = make_blobs(n_samples=300, n_features=4, centers=3, cluster_std=1.0)

>>> y_conf = []
>>> for i in range(len(y)):
...    if y[i] == 0 or y[i] == 2:
...        y_conf.append(0)
...    else:
...        y_conf.append(1)
>>> y_conf = np.array(y_conf)

>>> scatter_and_color_according_to_y(X, y_conf, col='rainbow', dim_reduct='LDA', projection='3d')


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
import warnings
from os.path import expanduser
from time import gmtime, strftime
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import matplotlib


def scatter_and_color_according_to_y(
    X,
    y=None,
    col='rainbow',
    projection='2d',
    dim_reduct='LDA',
    save=False,
    legend=True,
    saving_loc='/home/chris/',
    saving_name='myplot-',
    plot_tag_name=False,
    super_alpha=10,
    cmap_col='viridis',
    *args,
    **kwargs
):
    """
    :param X: an array of feature vectors
    :param y: an array of tags
    :param col: 'random' or 'rainbow'. Rainbow tends to give more distinct colors
    :param projection: '2d' or '3d'
    :param dim_reduct: 'LDA' or 'PCA', the dimension reduction method used (when needed)
                        anything else and the function will use the first 2 or 3 coordinates by default
    :param iterated: whether or not to use iterated projection in LDA when the number of tags is 2
                     without the iterate projection, one would get only 1 dimension out of lda
                     if set to False and the number of tags is 2 and the original space has dimension
                     more than 2, the first two dimensions will be retained for the scatterplot
                     (i.e. no smart drawing then). This last option can be useful when the iterated
                     projection just yield points on a line.
           fall_back: when LDA cannot produce the required number of dimensions, PCA is used instead.
    :param save: a boolean, whether or not the plot will be saved
    :param saving_loc: a string, the location where the file will be saved. If none is given,
                       it will be saved in the home folder
    :param args: more argument for scatter
    :param kwargs: more keyword argument for scatter
    :return: a plot of 2d scatter plot of X with different colors for each tag
    """

    if projection == '1d':
        proj_dim = 1
    elif projection == '2d':
        proj_dim = 2
    elif projection == '3d':
        proj_dim = 3
    else:
        warnings.warn(
            'The choices for the parameter projectionare'
            " '1d', '2d' or '3d'. Anything else and it will be assumed to be '2d' by default"
        )
        proj_dim = 2

    if y is None:
        y = np.zeros(len(X), dtype='int')
        legend = False
        n_tags = 1
        tags = [0]
        if dim_reduct == 'LDA':
            warnings.warn(
                'LDA cannot be used if no y is provided, will use PCA instead'
            )
            dim_reduct = 'PCA'

    elif isinstance(y[0], float):
        legend = False
        cm = plt.cm.get_cmap(cmap_col)
        colors = None
        col = 'continuous'
        no_tag = True
        if dim_reduct == 'LDA':
            warnings.warn(
                'LDA cannot be used if the y entries are floats, will use PCA instead'
            )
            dim_reduct = 'PCA'

    else:
        tags = np.unique(y)
        n_tags = len(tags)
        if dim_reduct == 'LDA' and proj_dim > n_tags - 1:
            warnings.warn(
                'LDA cannot be used to produce {} dimensions if y has less than {} classes,'
                ' will use PCA instead'.format(proj_dim, proj_dim + 1)
            )
            dim_reduct = 'PCA'

    second_index = 1
    third_index = 2
    n_points, n_dim = X.shape
    alpha = min(super_alpha + 1 / n_points, 1)

    # use LDA/PCA to project on a 2d/3d space if needed
    if n_dim > proj_dim:
        if dim_reduct == 'LDA':
            LDA = LinearDiscriminantAnalysis(n_components=proj_dim)
            X = LDA.fit(X, y).transform(X)

        elif dim_reduct == 'PCA':
            pca = PCA(n_components=proj_dim)
            X = pca.fit_transform(X)

        elif dim_reduct == 'TSNE':
            X = TSNE(n_components=proj_dim).fit_transform(X)

        elif dim_reduct == 'random':
            pass

        else:
            X = X[:, :proj_dim]

    if col is 'rainbow':
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_tags))
    if col is 'random':
        colors = matplotlib.colors.hsv_to_rgb(np.random.rand(n_tags, 3))

    if projection == '1d':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        if n_dim == 1:
            second_index = 0
        if colors is not None:
            for c, i in zip(colors, tags):
                sc = ax.scatter(
                    X[y == i, 0],
                    np.zeros(np.sum(y == i)),
                    c=[c],
                    alpha=alpha,
                    s=10,
                    linewidths=0.05,
                    marker='+',
                    *args,
                    **kwargs
                )
                if legend:
                    handle = mpatches.Patch(color=c, label=i)
                    handles.append(handle)
                    ax.legend(
                        handles=handles, loc='center left', bbox_to_anchor=(1, 0.5)
                    )
        else:
            ax.scatter(X[:, 0], X[:, second_index], c=y, alpha=alpha, *args, **kwargs)
            ax.colorbar()

    if projection == '2d':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        if n_dim == 1:
            second_index = 0
        if colors is not None:
            for c, i in zip(colors, tags):
                sc = ax.scatter(
                    X[y == i, 0],
                    X[y == i, second_index],
                    c=[c],
                    alpha=alpha,
                    *args,
                    **kwargs
                )
                if legend:
                    handle = mpatches.Patch(color=c, label=i)
                    handles.append(handle)
                    ax.legend(
                        handles=handles, loc='center left', bbox_to_anchor=(1, 0.5)
                    )
        else:
            ax.scatter(X[:, 0], X[:, second_index], c=y, alpha=alpha, *args, **kwargs)
            ax.colorbar()

    if projection == '3d':
        handles = []
        if n_dim == 1:
            second_index = 0
            third_index = 0
        if n_dim == 2:
            third_index = 1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if colors is not None:
            for c, i in zip(colors, tags):
                sc = ax.scatter(
                    X[y == i, 0],
                    X[y == i, second_index],
                    X[y == i, third_index],
                    c=[c],
                    alpha=alpha,
                )
                if legend:
                    handle = mpatches.Patch(color=c, label=i)
                    handles.append(handle)
                    ax.legend(
                        handles=handles, loc='center left', bbox_to_anchor=(1, 0.5)
                    )
        else:
            p = ax.scatter(
                X[:, 0],
                X[:, second_index],
                X[:, third_index],
                c=y,
                alpha=alpha,
                *args,
                **kwargs
            )
            fig.colorbar(p)

    if plot_tag_name and not no_tag:
        for tag in range(n_tags):
            tag_center = np.mean(X[y == tag], axis=0)
            if projection == '2d':
                plt.text(tag_center[0], tag_center[1], tags[tag])
            else:
                ax.text(
                    tag_center[0],
                    tag_center[1],
                    tag_center[2],
                    tags[tag],
                    size=20,
                    zorder=1,
                    color='k',
                )

    if save:
        path = (
            saving_loc + saving_name + datetime.datetime.today().strftime('%Y-%m-%d-%r')
        )
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def save_figs_to_pdf(figs, pdf_filepath=None):
    """
    Save figures to a single pdf
    :param figs:
    :param pdf_filepath:
    :return:
    """

    if pdf_filepath is None:
        pdf_filepath = '' + datetime.datetime.today().strftime('%Y-%m-%d-%r') + '.pdf'
    with PdfPages(pdf_filepath) as pdf:
        for fig in figs:
            pdf.savefig(fig)


# def rgb(minimum, maximum, value):
#     minimum, maximum = float(minimum), float(maximum)
#     ratio = 2 * (value-minimum) / (maximum - minimum)
#     b = int(max(0, 255 * (1 - ratio)))
#     r = int(max(0, 255 * (ratio - 1)))
#     g = 255 - b - r
#     return r/255, g/255, b/255, 1


def side_by_side_bar(
    list_of_values_for_bars, width=1, spacing=1, list_names=None, colors=None
):
    """
    A plotting utility making side by side bar graphs from a list of list (of same length) of values.

    :param list_of_values_for_bars: list of list of values for the bar graphs
    :param width: the width of the bar graphs
    :param spacing: the size of the spacing between groups of bars
    :param list_names: the names to assign to the bars, in the same order as in list_of_values_for_bars
    :param list_colors: the colors to use for the bars, in the same order as in list_of_values_for_bars
    :return: a nice plot!
    """

    # if no list_names is specified, the names on the legend will be integer
    if list_names is None:
        list_names = range(len(list_of_values_for_bars))
    # if no list_colors is specified, the colors will be chosen from the rainbow
    n_bars = len(list_of_values_for_bars)
    if colors is None:
        colors = plt.cm.rainbow(np.linspace(0, 1, n_bars))
    else:
        assert (
            len(colors) >= n_bars
        ), "There's not enough colors for the number of bars ({})".format(n_bars)
    ax = plt.subplot(111)
    # making each of the bar plot
    for i, list_of_values_for_bars in enumerate(list_of_values_for_bars):
        x = [
            width * j * n_bars + spacing * j + i * width
            for j in range(len(list_of_values_for_bars))
        ]
        ax.bar(x, list_of_values_for_bars, width=width, color=colors[i], align='center')
    ax.legend(list_names)
    ax.xaxis.set_ticklabels([])


def ratio_comparison_vlines(y1, y2, c1='b', c2='k'):
    """
    Plots vlines of y1/y2.
    :param y1: numerator
    :param y2: denominator
    :param c1: color of numerator
    :param c2: color of denominator (will be a straight horizontal line placed at 1)
    :return: what plt.plot returns
    """
    y = np.array(y1) / np.array(y2)
    plt.vlines(list(range(len(y))), 1, y)
    plt.hlines(1, 0, len(y) - 1, colors=c2)
    return plt.plot(list(range(len(y))), y, 'o', color=c1)
