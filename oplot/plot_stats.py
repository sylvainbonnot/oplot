"""Functions to represent the accuracy of outlier or classification algorithms"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import six
import os
import sympy as sp
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm
import warnings
from oplot.util import fixed_step_chunker



def plot_func_sphinx(arg1: int, arg2: str) -> bool:
    """Summary line.

    Extended description of function.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    """
    return True


def plot_freqs_stats(X, upper_frequency=22050, n_bins=1025, normalized=True):
    """
    X is intended to be the list/array of spectra, the function plots the mean, max and min of each frequency.
    If normalized, the min/max/mean entries are all divided by the the minimum value in the mean array.
    """

    max_each_freq = np.max(X, axis=0)
    min_each_freq = np.min(X, axis=0)
    mean_each_freq = np.mean(X, axis=0)

    if normalized:
        normalization_factor = min(mean_each_freq)
        max_each_freq /= normalization_factor
        min_each_freq /= normalization_factor
        mean_each_freq /= normalization_factor

    plt.figure(figsize=(20, 10))
    plt.plot(np.linspace(0, upper_frequency, n_bins), max_each_freq, label='max')
    plt.plot(np.linspace(0, upper_frequency, n_bins), min_each_freq, label='min')
    plt.plot(np.linspace(0, upper_frequency, n_bins), mean_each_freq, label='mean')
    plt.legend(loc='best')
    plt.xlabel('frequencies')
    plt.ylabel('intensities')
    plt.show()

    plt.figure(figsize=(20, 10))
    freq_var = np.var(X, axis=0)
    plt.plot(np.linspace(0, upper_frequency, n_bins), freq_var, label='variance')
    plt.xlabel('frequencies')
    plt.ylabel('variance')
    plt.show()


def make_heatmap(
    matrix_results,
    tags,
    rounding=4,
    fig_size=(20, 20),
    make_symmetric=False,
    fill_diag=None,
    cmap=plt.cm.Blues,
    name='',
):
    """
    Makes a heatmap plot of the matrix_results where the entries are rounded.
    If matrix_results is upper or lower diagonal and make_symmetric is set to true,
    plot will be made symmetric by copying the strict upper or lower half.
    """

    if make_symmetric:
        matrix_copy = np.copy(matrix_results).T
        matrix_results = matrix_results + matrix_copy
    if fill_diag is not None:
        np.fill_diagonal(matrix_results, fill_diag)
    # rounding its entries for cleaner plot
    rounding_func = np.vectorize(lambda x: round(x, rounding))
    matrix_results = rounding_func(matrix_results)

    # making the plot
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(matrix_results, cmap=cmap)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(tags)))
    ax.set_yticks(np.arange(len(tags)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(tags)
    ax.set_yticklabels(tags)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # Loop over data dimensions and create text annotations.
    for i in range(len(tags)):
        for j in range(len(tags)):
            text = ax.text(
                j, i, matrix_results[i, j], ha='center', va='center', color='w'
            )
    ax.set_title('Pairwise Classification accuracy ' + name)
    fig.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    fig=None,
    ax=None,
    classes=None,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues,
    saving_path=None,
    figsize=(10, 10),
    color_bar=False,
    plot=True,
    cm=False,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Only use the labels that appear in the data
    if classes is None:
        classes = np.unique(y_true)

    # Compute confusion matrix
    if not cm:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = cm.reshape((2, 2))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if color_bar:
        ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label',
    )
    ax.grid(False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # remove this condition if a 0 in each cell with no confusion is wanted
            if cm[i, j] > 0:
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha='center',
                    va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                )
    if fig is not None:
        fig.tight_layout()
        if saving_path is not None:
            fig.savefig(saving_path, bbox_inches='tight', dpi=200)
    if plot:
        plt.show()


def list_mult(l, mult, random_remainder=False):
    """
    Extend the multiplication of lists by an integer naturally to float.
    EX: [1,2] * 1.5 = [1,2,1]
    If random_remainder is set to True, the decimal part of the new list will be chose at random from l.
    """

    mult_in, remain = divmod(mult, 1)
    if random_remainder:
        remainder = random.sample(l, int(np.ceil(len(l) * remain)))
    else:
        remainder = l[: int(np.ceil(len(l) * remain))]
    return l * int(mult_in) + remainder


def rebalancing_normal_outlier_ratio(normal_scores, outlier_scores, percent_outliers):
    """
    Rebalance artificially the ratio outlier/(normal + outlier) to the specified percent_outliers.
    Does this by copying data points, use with caution!
    :param normal_scores:
    :param outlier_scores:
    :param percent_outliers:
    :return:
    """

    n_normal = len(normal_scores)
    n_outlier = len(outlier_scores)
    correction_multiplier = (
        n_outlier / n_normal * (1 - percent_outliers) / percent_outliers
    )
    if correction_multiplier >= 1:
        return (
            np.array(list_mult(list(normal_scores), correction_multiplier)),
            outlier_scores,
        )
    else:
        return (
            normal_scores,
            np.array(list_mult(list(outlier_scores), 1 / correction_multiplier)),
        )


def rebalance_scores(test_scores, test_truth, outlier_proportion):
    """
    Re-balances the ratio of normal/outlier scores by copying the normal/outliers scores when needed.
    This is useful to compute real life precision/recall and other such metrics when the actual proportion
    of outlier/normal is known but not achieved in the test set.
    """

    test_normal_scores = test_scores[test_truth == 0]
    test_anom_scores = test_scores[test_truth == 1]
    normal_scores, anom_scores = rebalancing_normal_outlier_ratio(
        test_normal_scores, test_anom_scores, percent_outliers=outlier_proportion
    )
    bal_test_scores = np.hstack((normal_scores, anom_scores))
    bal_test_truth = np.array([0] * len(normal_scores) + [1] * len(anom_scores))
    return bal_test_scores, bal_test_truth


def get_tn_fp_fn_tp(truth, scores, threshold=2):
    """
    compute the counts of true negative, false positive, false negative and true positive as predicted
    by the outlier score for the given threshold
    :param truth: a list of 0 and 1's, 0 meaning normal and 1 meaning outlier
    :param scores: a list of outlier scores, of same length as truth
    :param threshold: a float, any score under is predicted as normal and above as outlier
    :return: true negative, false positive, false negative, true positive
    """

    thresh_funct = lambda x: 1 if x > threshold else 0
    pred = list(map(thresh_funct, scores))
    print(confusion_matrix(truth, pred).ravel())
    tn, fp, fn, tp = confusion_matrix(truth, pred).ravel()
    return tn, fp, fn, tp


def make_tables_tn_fp_fn_tp(
    truth, scores, threshold_range=None, n_thresholds=10, normalize=False
):
    """
    Make a table of counts of tn, fp, fn, tp for n_thresholds equally spaced in threshold_range

    :param truth:  a list of 0 and 1's, 0 meaning normal and 1 meaning outlier
    :param scores: a list of outlier scores, of same length as truth
    :param threshold_range: two values in a tuple or list. The thresholds will be picked within these.
    :param n_thresholds: the number of thresholds to produce
    :return: a panda dataframe with rows of the form (threshold, tn, tp, fn, fp)
    """

    if threshold_range is None:
        threshold_range = (min(scores[truth == 0]), max(scores[truth == 0]))
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    row = []
    for threshold in thresholds:
        tn, fp, fn, tp = get_tn_fp_fn_tp(truth, scores, threshold)
        row.append([threshold, int(tn), int(fp), int(fn), int(tp)])
    row = row[::-1]
    df = pd.DataFrame(
        row,
        columns=[
            'Threshold',
            'True Negative',
            'False Positive',
            'False Negative',
            'True Positive',
        ],
    )
    if normalize:
        total_positive = np.sum(truth)
        total_negative = len(truth) - total_positive
        df['True Negative'] = df['True Negative'].apply(lambda x: x / total_negative)
        df['False Positive'] = df['False Positive'].apply(lambda x: x / total_negative)
        df['False Negative'] = df['False Negative'].apply(lambda x: x / total_positive)
        df['True Positive'] = df['True Positive'].apply(lambda x: x / total_positive)
    return df


def make_tn_fp_fn_tp_tag_lists(truth, scores, threshold, tags=None):
    """
    Return a dictionary with the keys being tp, tn, fp and fn and the corresponding values are the indices of the scores/truth/tags
    for the given threshold. If a list of tags of the sounds corresponding to each scores is given, the values of the dict will
    be those tags rather than indices
    """
    n = len(truth)
    if tags is None:
        to_pick_from = range(n)
        counter = 0
    else:
        to_pick_from = tags
        counter = 1
    threshold_pred = [1 if score >= threshold else 0 for score in scores]
    true_positive = [
        to_pick_from[i] for i in range(n) if (threshold_pred[i] == 1 and truth[i] == 1)
    ]
    true_negative = [
        to_pick_from[i] for i in range(n) if (threshold_pred[i] == 0 and truth[i] == 0)
    ]
    false_positive = [
        to_pick_from[i] for i in range(n) if (threshold_pred[i] == 1 and truth[i] == 0)
    ]
    false_negative = [
        to_pick_from[i] for i in range(n) if (threshold_pred[i] == 0 and truth[i] == 1)
    ]
    if counter:
        true_positive = Counter(true_positive)
        true_negative = Counter(true_negative)
        false_positive = Counter(false_positive)
        false_negative = Counter(false_negative)

    return {
        'tp': true_positive,
        'tn': true_negative,
        'fp': false_positive,
        'fn': false_negative,
    }


def vlines(
    x,
    ymin=0,
    ymax=None,
    marker='o',
    marker_kwargs=None,
    colors='k',
    linestyles='solid',
    label='',
    data=None,
    **kwargs,
):
    """Plot vlines in a more intuitive way than the default matplotlib version"""
    if ymax is None:
        ymax = x
        x = np.arange(len(ymax))

        if ymax is None:
            raise ValueError('Need to specify ymax')

    if marker is not None:
        if marker_kwargs is None:
            marker_kwargs = {}
        plt.plot(x, ymax, marker, **marker_kwargs)

    return plt.vlines(
        x,
        ymin=ymin,
        ymax=ymax,
        colors=colors,
        linestyles=linestyles,
        label=label,
        data=data,
        **kwargs,
    )


def make_normal_outlier_timeline(
    y,
    scores,
    y_order=None,
    vertical_sep=False,
    saving_path=None,
    fig_size=(16, 5),
    name='normal/outlier scores',
    smooth=False,
    legend_size=10,
    title_font_size=10,
    label_for_y=None,
    legend_n_cols=1,
    xticks=None,
    xticks_labels=None,
    xticks_rotation=90,
):
    """
    Plots all scores grouped by their y values in order specified by y_order or np.unique(y) is left to None.
    :param scores: an array of outlier scores
    :param y: an array of tags, each tag will get its own color on the plot and its name on the legend
              The vertical line will also be grouped according to the tags, in the order given in y_order
    :param fig_size: the size of the plot
    :param line_width: the thickness of the line on the plot
    :return:
    """

    if label_for_y is None:
        label_for_y = lambda x: x

    scores = np.array(scores)
    y = np.array(y)
    if not y_order:
        y_order = np.unique(y)
    else:
        if set(np.unique(y)) != set(y_order):
            warnings.warn('y_order does not include the values present in y')

    if smooth:
        new_scores = []
        new_y = []
        new_y_order = []
        for i in y_order:
            try:
                scores_i = list(smooth_scores(scores[y == i], window_size=smooth))
                new_scores.extend(scores_i)
                new_y += [i] * len(scores_i)
                new_y_order.append(i)
            except ValueError:
                print(
                    f'There are less scores corresponding to {i} than the smoothing window size. '
                    f'These scores will be dropped'
                )
        scores = np.array(new_scores)
        y = np.array(new_y)
        y_order = new_y_order

    colors = cm.rainbow(np.linspace(0, 1, len(y_order)))

    fig, ax1 = plt.subplots(figsize=fig_size)
    ax1.tick_params(labelright=True)
    n_points_drawn = 0

    for i, tag in enumerate(y_order):
        values = scores[y == tag]
        n_points = len(values)
        ax1.vlines(
            np.arange(n_points_drawn, n_points_drawn + n_points),
            ymin=0,
            ymax=values,
            label=label_for_y(tag),
            colors=colors[i],
        )
        n_points_drawn += n_points
    if xticks is not None and xticks_labels is None:
        plt.xticks(ticks=xticks, rotation=xticks_rotation)
    if xticks_labels is not None and xticks is None:
        plt.xticks(
            ticks=plt.xticks()[0], labels=xticks_labels, rotation=xticks_rotation
        )
    if xticks is not None and xticks_labels is not None:
        plt.xticks(ticks=xticks, labels=xticks_labels, rotation=xticks_rotation)
    if vertical_sep == 'auto':
        group_len = apply_function_on_consecutive(y, y, lambda x: len(x))
        vertical_lines_pos = np.cumsum(group_len)
        ax1.vlines(
            vertical_lines_pos,
            ymin=np.min(scores),
            ymax=np.max(scores),
            colors='k',
            linewidth=0.3,
            linestyles='-.',
        )
    elif vertical_sep:
        ax1.vlines(
            vertical_sep,
            ymin=np.min(scores),
            ymax=np.max(scores),
            colors='k',
            linewidth=0.3,
            linestyles='-.',
        )

    if legend_size:
        plt.legend(prop={'size': legend_size}, loc=(1.04, 0), ncol=legend_n_cols)
    plt.title(name, fontsize=title_font_size)
    if saving_path is not None:
        plt.savefig(saving_path, bbox_inches='tight', dpi=200)

    plt.show()


def pick_equally_spaced_points(values, n_points):
    """
    :param values: a list of values
    :param n_points: the number of indices to choose
    :return: a list of indices to choose from the list values in order for these picked values
             to be as equally spaced as possible, starting with the first value and ending with the final value

    DO NOT TRUST THIS FUNCTION! It is made only for increasing values and is only somewhat accurate if the values
    are "fine" compared to the number of point n_points. It works well for what it is intended: finding equally spaced
    points on a precision/recall curve if many points are available on the curve.

    >>> pick_equally_spaced_points([1,2,3], 3)
    array([0, 1, 2])
    >>> pick_equally_spaced_points(range(300), 3)
    array([  0, 150, 299])
    >>> pick_equally_spaced_points([1, 1.1, 1.2, 4, 5, 6], 3)
    array([0, 3, 5])
    """
    min_value = min(values)
    max_value = max(values)
    spacing = (max_value - min_value) / (n_points - 1)
    # first and last are picked by default
    points_idx = [0]
    position = values[1]
    index = 1
    for i in range(n_points - 2):
        while position < spacing * (1 + i):
            index += 1
            position = values[index]
        points_idx.append(index)
    points_idx.append(len(values) - 1)
    return np.array(points_idx)


# TODO: make the integer values show as integer instead of floats
def render_mpl_table(
    data,
    col_width=3.0,
    row_height=0.625,
    font_size=14,
    header_color='#40466e',
    row_colors=['#f1f1f2', 'w'],
    edge_color='w',
    bbox=[0, 0, 1, 1],
    header_columns=0,
    ax=None,
    path_to_save=None,
    round_decimals=3,
    cols_to_round=(),
    cols_to_int='all_other',
    dpi=300,
    **kwargs,
):
    """
    Take a pandas dataframe and represents it with a picture. This allows to save a .png version of the dataframe.
    """
    if round_decimals:
        if len(cols_to_round) == 0:
            data = data.round(decimals=round_decimals)
        else:
            for col in cols_to_round:
                data[col] = data[col].apply(lambda x: round(x, round_decimals))
    if cols_to_int == 'all_others':
        for col in data.columns:
            if col not in cols_to_round:
                data[col] = data[col].apply(lambda x: int(x))
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array(
            [col_width, row_height]
        )
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(
        cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs
    )

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    if path_to_save:
        plt.savefig(path_to_save, bbox_inches='tight', dpi=dpi)
    plt.show()


# all these scores except for MCC gives a score between 0 and 1.
# I normalized MMC into what I call NNMC in order to keep the same scale for all.
base_statistics_dict = {
    'TPR': lambda tn, fp, fn, tp: tp / (tp + fn),
    # sensitivity, recall, hit rate, or true positive rate
    'TNR': lambda tn, fp, fn, tp: tn
    / (tn + fp),  # specificity, selectivity or true negative rate
    'PPV': lambda tn, fp, fn, tp: tp
    / (tp + fp),  # precision or positive predictive value
    'NPV': lambda tn, fp, fn, tp: tn / (tn + fn),  # negative predictive value
    'FNR': lambda tn, fp, fn, tp: fn / (fn + tp),  # miss rate or false negative rate
    'FPR': lambda tn, fp, fn, tp: fp / (fp + tn),  # fall-out or false positive rate
    'FDR': lambda tn, fp, fn, tp: fp / (fp + tp),  # false discovery rate
    'FOR': lambda tn, fp, fn, tp: fn / (fn + tn),  # false omission rate
    'TS': lambda tn, fp, fn, tp: tp / (tp + fn + fp),
    # threat score (TS) or Critical Success Index (CSI)
    'ACC': lambda tn, fp, fn, tp: (tp + tn) / (tp + tn + fp + fn),  # accuracy
    'F1': lambda tn, fp, fn, tp: (2 * tp) / (2 * tp + fp + fn),  # F1 score
    'NMCC': lambda tn, fp, fn, tp: (
        (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + 1
    )
    / 2,
    # NORMALIZED TO BE BETWEEN 0 AND 1 Matthews correlation coefficient
    'BM': lambda tn, fp, fn, tp: tp / (tp + fn) + tn / (tn + fp) - 1,
    # Informedness or Bookmaker Informedness
    'MK': lambda tn, fp, fn, tp: tp / (tp + fp) + tn / (tn + fn) - 1,
}  # Markedness

synonyms = {
    'TPR': ['recall', 'sensitivity', 'true_positive_rate', 'hit_rate', 'tpr'],
    'TNR': ['specificity', 'SPC', 'true_negative_rate', 'selectivity', 'tnr'],
    'PPV': ['precision', 'positive_predictive_value', 'ppv'],
    'NPV': ['negative_predictive_value', 'npv'],
    'FNR': ['miss_rate', 'false_negative_rate', 'fnr'],
    'FPR': ['fall_out', 'false_positive_rate', 'fpr'],
    'FDR': ['false_discovery_rate', 'fdr'],
    'FOR': ['false_omission_rate', 'for'],
    'TS': ['threat_score', 'critical_success_index', 'CSI', 'csi', 'ts'],
    'ACC': ['accuracy', 'acc'],
    'F1': ['f1_score', 'f1', 'F1_score'],
    'NMCC': ['normalized_Matthews_correlation_coefficient', 'nmcc'],
    'BM': ['informedness', 'bookmaker_informedness', 'bi', 'BI', 'bm'],
    'MK': ['markedness', 'mk'],
}


def pair_metrics_to_reference(
    pair_metrics={'x': 'TPR', 'y': 'FPR'},
    outlier_proportion=0.2,
    label='chance line',
    base_statistics_dict=base_statistics_dict,
    synonyms=synonyms,
):
    """
    Utility to compute the reference/chance curve for a pair metrics type curve. Note that for certain combination
    of metrics, the curve may be a point.
    """

    # r = rate of positive of a random outlier model
    r = sp.symbols('r')
    # R = rate of positive in the test set
    R = sp.symbols('R')

    tn = (1 - r) * (1 - R)
    fp = r * (1 - R)
    fn = (1 - r) * R
    tp = r * R

    statistics_dict = dict()
    for k, v in base_statistics_dict.items():
        statistics_dict[k] = v
        for alt in synonyms.get(k, []):
            statistics_dict[k] = v
            statistics_dict[alt] = v

    fx = statistics_dict[pair_metrics['x']]
    fy = statistics_dict[pair_metrics['y']]

    simp_fx = sp.expand(fx(tn, fp, fn, tp))
    simp_fy = sp.expand(fy(tn, fp, fn, tp))

    x_values = np.linspace(0, 1, 40)
    y_values = [
        simp_fy.evalf(subs={R: outlier_proportion, simp_fx: i}) for i in x_values
    ]
    plt.plot(x_values, y_values, '--', c='r', label=label)


def wiggle_values_keep_order(values):
    """
    Wiggles the values in a list of scores so has to remove any duplicate values while keeping the same order.
    This is intended to use with plot_outlier_metric_curve in order to smooth the curve in situations where scores
    are repeating a lot

    >>> scores = [1, 1, 2, 3, 3, 4, 5, 5]
    >>> wiggle_values_keep_order(scores)
    array([1. , 1.5, 2. , 3. , 3.5, 4. , 5. , 5.5])

    :param values: a list of floats in general
    :return: a new list of float, where the values have been moved around a little, without altering their order
    """

    new = []
    current_val = values[0]
    current_val_count = 0
    for val in values:
        if val <= current_val:
            current_val_count += 1
        else:
            next_val = val
            for i in range(current_val_count):
                new.append(
                    current_val + i * (next_val - current_val) / current_val_count
                )
            current_val = next_val
            current_val_count = 1
    len_new = len(new)
    last_wiggle = (values[-1] - values[len_new - 1]) / (len(values) - len_new)
    for i in range(len(values) - len_new):
        new.append(values[-(i + 1)] + i * last_wiggle)

    return np.array(new)


def cumulative_tn_fp_fn_tp(truth, scores):
    """Compute efficiently the cumulative tn, fp, fn and tp """

    truth = np.array(truth)
    scores = np.array(scores)

    sorted_idx = np.argsort(scores, kind='mergesort')
    sorted_truth = truth[sorted_idx]

    total_true_positive = np.sum(sorted_truth)
    total_true_negative = np.sum(np.logical_not(sorted_truth))

    fns = np.concatenate(([0], np.cumsum(sorted_truth)))
    tns = np.concatenate(([0], np.cumsum(np.logical_not(sorted_truth))))
    fps = total_true_negative - tns
    tps = total_true_positive - fns

    return tns, fps, fns, tps


def wiggle_scores(scores, truth):
    """
    Sort scores from low to high while keeping truth aligned with it. The original values
    in scores which are present multiple times are spread out equally between their value and
    the next larger score.

    :param scores: a list of scores
    :param truth: a list of 0/1, for normal/abnormal
    :return: scores and truth list sorted from low to high score and where the scores have been
             wiggled

    >>> wiggle_scores([1, 1, 2], [0, 0, 0])
    (array([1. , 1.5, 2. ]), array([0, 0, 0]))

    >>> wiggle_scores([4, 1, 1, 2], [1, 0, 0, 0])
    (array([1. , 1.5, 2. , 4. ]), array([0, 0, 0, 1]))

    """
    scores = np.array(scores)
    truth = np.array(truth)
    z = list(zip(scores, truth))
    z.sort()
    scores, truth = zip(*z)
    return np.array(wiggle_values_keep_order(scores)), np.array(truth)


def plot_outlier_metric_curve(
    truth,
    scores,
    pair_metrics={'x': 'TPR', 'y': 'PPV'},
    plot_curve=True,
    curve_legend_name=None,
    title=None,
    plot_table_points_on_curve=False,
    plot_chance_line=True,
    plot_table=False,
    n_points_for_table=10,
    axis_name_dict=None,
    saving_root=None,
    outlier_proportion=None,
    wiggle=False,
    table_dpi=300,
    base_statistics_dict=base_statistics_dict,
    synonyms=synonyms,
    return_rauc=True,
    add_point_left=None,
    add_point_right=None,
):
    """
    Plots one outlier scores metric against another one. The metrics name can be any names in the base_statistics_dict
    or the synonyms dict. The chance line/curve is automatically computed and displayed along with a table
    of equally spaced point on the curve.

    :param truth: an array of 0/1, the ground truth: 0 for normal, 1 for outlier
    :param scores: the scores as predicted by our model. Higher scores is expected to correspond to outliers.
    :param pair_metrics: A dictionary with two keys, one for each of the metrics to represent. (x/y on the x/y axis)
    :param plot_curve: boolean, whether to plot the curve
    :param curve_legend_name: the name of the curve as displayed in the legend
    :param title: the title of the curve, by default the name of the metrics
    :param plot_table_points_on_curve: whether to display dots on the rauc curve corresponding to points in the table
    :param plot_chance_line: boolean, whether or not the display the chance line
    :param plot_table: boolean, whether or not to plot the table, useful to share nice pics with customers
    :param n_points_for_table: int, the number of equally spaced points for the table
    :param axis_name_dict: a dictionary specifying the name to display on the x/y axis. If set to None, the names in
                          pair_metrics are used
    :param saving_root: if set, path to the folder where the pictures will be saved.
    :param outlier_proportion: None or a float between 0 and 1. If a float is chosen either the normal scores or the
                               anomaly scores will be copied over to achieve the requested proportion of outlier
    :param wiggle: boolean, whether the scores will be slightly modified in order to avoid duplicate. Can
                          help in drawing the curve when too many scores are the same. Most often
                          the problem wiggle_scores solves arise from using sklearn OneClassSVM for the scores
                          computation
    :param table_dpi: int, the higher the finer the pic
    :param base_statistics_dict: the dictionary of possible metrics with the functions computing the metrics from
                                 the tn_fp_fn_tp counts. See above.
    :param synonyms: a dictionary containing the synonymous, allowing the user to refer to the metrics in different
                     terms
    :param return_rauc: boolean, whether or not to return the area under the curve
    """

    # make a saving folder if specified
    if saving_root and not os.path.isdir(saving_root):
        os.mkdir(saving_root)

    # translate the metrics requested
    statistics_dict = dict()
    for k, v in base_statistics_dict.items():
        statistics_dict[k] = v
        for alt in synonyms.get(k, []):
            statistics_dict[k] = v
            statistics_dict[alt] = v

    scores = np.array(scores)
    truth = np.array(truth)

    # spread out the repeating scores to avoid degenerate cases
    if wiggle:
        scores, truth = wiggle_scores(scores, truth)
    normal_scores = scores[truth == 0]
    outlier_scores = scores[truth == 1]

    # TODO: this is NOT smart and can be slow if the rebalancing is drastic. Need improvement.
    # re-balance the proportion as required by simply copying whichever of outliers/normal elements we need
    # in greater quantities
    if outlier_proportion is not None:
        normal_scores, outlier_scores = rebalancing_normal_outlier_ratio(
            normal_scores, outlier_scores, percent_outliers=outlier_proportion
        )
        truth = np.array([0] * len(normal_scores) + [1] * len(outlier_scores))
        scores = np.array(list(normal_scores) + list(outlier_scores))
    else:
        outlier_proportion = len(outlier_scores) / (
            len(outlier_scores) + len(normal_scores)
        )

    # This is the core of the function, finding the tns, fps, fns, tps and computing the values
    # for the metrics requested
    tns, fps, fns, tps = cumulative_tn_fp_fn_tp(truth, scores)
    x = []
    y = []
    fx = statistics_dict[pair_metrics['x']]
    fy = statistics_dict[pair_metrics['y']]
    for tn, fp, fn, tp in zip(tns, fps, fns, tps):
        x.append(fx(tn, fp, fn, tp))
        y.append(fy(tn, fp, fn, tp))

    # find all the nan and remove them
    nan_idx = list(np.argwhere(np.isnan(x)).flatten()) + list(
        np.argwhere(np.isnan(y)).flatten()
    )
    x = [item for idx, item in enumerate(x) if not idx in nan_idx]
    y = [item for idx, item in enumerate(y) if not idx in nan_idx]

    # sorting the results keeping x and y aligned
    z = list(zip(x, y))
    z = sorted(z, key=itemgetter(0))
    z = sorted(z, key=itemgetter(0), reverse=True)
    x, y = zip(*z)

    if add_point_left:
        x = [add_point_left[0]] + list(x)
        y = [add_point_left[1]] + list(y)
    if add_point_right:
        x = list(x) + [add_point_right[0]]
        y = list(y) + [add_point_right[1]]

    x = np.array(x)
    y = np.array(y)

    # Getting equally spaced points for the table or curve
    if (plot_curve and plot_table_points_on_curve) or plot_table:
        sorted_x, sorted_y = parallel_sort([x, y])
        sorted_x = np.array(sorted_x)
        sorted_y = np.array(sorted_y)

        # note that pick_equally_spaced_points is a quick hack, will not work for less than ideal situation
        idx = pick_equally_spaced_points(sorted_x, n_points_for_table)
        x_points = sorted_x[idx]
        y_points = sorted_y[idx]
        x_nan = np.isnan(x_points)
        y_nan = np.isnan(y_points)
        xy_not_nan = np.logical_not(np.logical_or(x_nan, y_nan))
        x_points = x_points[xy_not_nan]
        y_points = y_points[xy_not_nan]

    if plot_table or plot_curve:
        if not axis_name_dict:
            x_label = pair_metrics['x']
            y_label = pair_metrics['y']
        else:
            x_label = axis_name_dict['x']
            y_label = axis_name_dict['y']

    # plotting the curve
    if plot_curve:
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        curve_name = y_label + '_' + x_label + '_curve'
        if not curve_legend_name:
            curve_legend_name = curve_name
        if not title:
            title = curve_name
        plt.plot(x, y, color='b', alpha=0.2, label=curve_legend_name)
        plt.fill_between(x, y, alpha=0.2, color='b')
        plt.ylim([0.0, 1.1])
        plt.xlim([0.0, 1.0])

        # add a reference line if specified by the user
        if plot_chance_line:
            pair_metrics_to_reference(
                pair_metrics=pair_metrics, outlier_proportion=outlier_proportion
            )
        plt.title(title)

        # adding the points to the curve
        if plot_table_points_on_curve:
            plt.scatter(x_points, y_points, label='table points')
        # saving the curve
        if saving_root:
            path_to_save = os.path.join(saving_root, title)
            plt.savefig(path_to_save, bbox_inches='tight', dpi=200, figsize=(6, 6))
        plt.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
        plt.show()

    if plot_table:
        # making a panda df with the recall/precision pairs of values and saving it as a picture
        rows = list(zip(x_points, y_points))
        df = pd.DataFrame(rows, columns=[x_label, y_label])
        if saving_root:
            path_to_save = os.path.join(saving_root, title + '_table')
        else:
            path_to_save = None
        render_mpl_table(df, path_to_save=path_to_save, dpi=table_dpi)

    if return_rauc:
        return area_under_points(list(zip(x, y))[1:])




def plot_outlier_metric_curve_sphinx_napoleon(
    truth,
    scores,
    pair_metrics={'x': 'TPR', 'y': 'PPV'},
    plot_curve=True,
    curve_legend_name=None,
    title=None,
    plot_table_points_on_curve=False,
    plot_chance_line=True,
    plot_table=False,
    n_points_for_table=10,
    axis_name_dict=None,
    saving_root=None,
    outlier_proportion=None,
    wiggle=False,
    table_dpi=300,
    base_statistics_dict=base_statistics_dict,
    synonyms=synonyms,
    return_rauc=True,
    add_point_left=None,
    add_point_right=None,
):
    """
    Plots one outlier scores metric against another one. The metrics name can be any names in the base_statistics_dict
    or the synonyms dict. The chance line/curve is automatically computed and displayed along with a table
    of equally spaced point on the curve.

    
    Args:
        truth: an array of 0/1, the ground truth: 0 for normal, 1 for outlier
        scores: the scores as predicted by our model. Higher scores is expected to correspond to outliers.
        temporary (bool): Whether or not to delete the file when the File
           instance is destructed

    
    """

    # make a saving folder if specified
    
    pass


def area_under_points(points):
    """
    Given a list of pair corresponding to the x and y coordinates of the points, find the
    area under the curve of the (simplest) piecewise linear function passing by all the points

    >>> area_under_points([(0, 1), (1, 1)])
    1.0
    >>> area_under_points([(0, 1), (1, 2)])
    1.5
    >>> area_under_points([(0, 1), (1, 2), (2, 2)])
    3.5

    """
    # sort the points according to their x coordinates
    sorted_points = sorted(points, key=lambda x: x[0])
    total_area = 0
    left_point = sorted_points[0]
    for right_point in sorted_points[1:]:
        total_area += (
            (right_point[0] - left_point[0]) * (right_point[1] + left_point[1]) / 2
        )
        left_point = right_point

    return total_area

def area_under_points_sphinx(points):
    """
    Given a list of pair corresponding to the x and y coordinates of the points, find the
    area under the curve of the (simplest) piecewise linear function passing by all the points

    .. code-block:: python
   
        def myfunc(x):
           return 2*x
    

    >>> area_under_points([(0, 1), (1, 1)])
    1.0
    >>> area_under_points([(0, 1), (1, 2)])
    1.5
    >>> area_under_points([(0, 1), (1, 2), (2, 2)])
    3.5

    """
    # sort the points according to their x coordinates
    pass


def smooth_scores(scores, window_size=2, window_step=None, smooth_func=np.mean):
    """
    Smooth an iterable of score by applying smooth_funct to each window of size window_size.
    If scores is smaller than window_size, an empty list is returned.

    :param scores: list, the scores to smooth
    :param window_size: int, the size of the window
    :param smooth_funct: function, the function applied to the windows
    :return: a new list of scores


    >>> list(smooth_scores([1], window_size=2))
    []
    >>> list(smooth_scores([1, 2], window_size=2))
    [1.5]
    >>> list(smooth_scores([1, 2, 3], window_size=2, window_step=1, smooth_func=np.max))
    [2, 3]

    """
    if window_step is None:
        window_step = window_size

    win_gen = fixed_step_chunker(scores, chk_size=window_size, chk_step=window_step)
    return map(smooth_func, win_gen)

def smooth_scores_sphinx_napoleon(scores, window_size=2, window_step=None, smooth_func=np.mean):
    """
    Smooth an iterable of score by applying smooth_funct to each window of size window_size.
    If scores is smaller than window_size, an empty list is returned.

    Args:
        scores: list, the scores to smooth
        window_size: int, the size of the window
        smooth_funct: function, the function applied to the windows

    

    Returns:
        a new list of scores


    >>> list(smooth_scores([1], window_size=2))
    []
    >>> list(smooth_scores([1, 2], window_size=2))
    [1.5]
    >>> list(smooth_scores([1, 2, 3], window_size=2, window_step=1, smooth_func=np.max))
    [2, 3]

    """
    if window_step is None:
        window_step = window_size

    win_gen = fixed_step_chunker(scores, chk_size=window_size, chk_step=window_step)
    return map(smooth_func, win_gen)



def split_on_consecutive(arr_to_split, arr_for_consec):
    """
    Split arr_to_split based on the entries of arr_for_consec. Each segment of consecutive
    equal entries in arr_for_consec will induce a split

    >>> split_on_consecutive([1, 1, 2], [1, 1, 2])
    [array([1, 1]), array([2])]
    >>> split_on_consecutive(['a', 'b', 'c', 'd', 'e', 'f', 'g'], [1, 1, 2, 1, 1, 1, 3])
    [array(['a', 'b'], dtype='<U1'), array(['c'], dtype='<U1'), array(['d', 'e', 'f'], dtype='<U1'), array(['g'], dtype='<U1')]

    """
    split_idx = []
    init_value = arr_for_consec[0]
    for idx, val in enumerate(arr_for_consec):
        if val != init_value:
            split_idx.append(idx)
            init_value = val
    return np.split(arr_to_split, split_idx)


def apply_function_on_consecutive(scores, arr_for_consec, func=np.mean):
    """
     Apply func to each block of consecutive value in scores if the corresponding values in arr_for_consec are constant
    :param scores: a list of floats
    :param arr_for_consec: an array of values defining the blocks
    :return: a list the output of func on each block
    """

    split_scores = split_on_consecutive(scores, arr_for_consec)
    smoothed_scores = []

    group_func = lambda x, y: x.extend([y])
    # little hack to see whether the function func returns a value or a list
    if isinstance(func([1, 2, 3]), list):
        group_func = lambda x, y: x.extend(y)

    for scores_group in split_scores:
        group_func(smoothed_scores, func(scores_group))

    return np.array(smoothed_scores)


from operator import itemgetter


def parallel_sort(iterable_list, sort_idx=0):
    """
    Sort several lists in iterable_list in parallel, according to the the list of index sort_idx
    :param iterable_list: list of list, all the lists have the same length
    :param sort_idx: int, the index of the list to sort by
    :return: a list sorted tuples


    >>> parallel_sort([[2, 3, 1], ['a', 'b', 'c']])
    [(1, 2, 3), ('c', 'a', 'b')]

    >>> parallel_sort([[2, 3, 1], ['a', 'b', 'c'], [10, 9, 7]], sort_idx=2)
    [(1, 3, 2), ('c', 'b', 'a'), (7, 9, 10)]

    """
    z = zip(*iterable_list)
    sorted_z = [item for item in sorted(z, key=itemgetter(sort_idx))]
    return list(zip(*sorted_z))
