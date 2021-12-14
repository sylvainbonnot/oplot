"""Visualizing various regions in a list of scores. E.g. adding colored regions
corresponding to percentiles to a timeline of outlier scores"""


import numpy as np
import matplotlib.pyplot as plt


# The code bellow is intended to add some visual interpretation to outliers scores display on a timeline
# See: https://www.dropbox.com/s/g6lxxovop3sibto/013-outlier_scores_functions.ipynb?dl=0
# for a demo of usage


def sort_scores_truth(scores, truth):
    """
    Sort the aligned scores and truth arrays from lowest to largest
    """

    # scores + index pairs
    z = list(enumerate(scores))
    z.sort(key=lambda x: x[1])
    sorted_scores = np.array([score[1] for score in z])
    sorted_truth = np.array([truth[score[0]] for score in z])
    return sorted_scores, sorted_truth


def get_confused_part(sorted_scores, sorted_truth):
    """
    Return the scores in the confused zone, i.e. scores whose range of values contains normal and anormal samples.

    :param sorted_scores: an array of outlier scores, higher is more abnormal
    :param sorted_truth:  an array of 0 for normal and 1 for abnormal
    :return: an array of scores
    """

    max_zero = np.max(sorted_scores[sorted_truth == 0])
    min_one = np.min(sorted_scores[sorted_truth == 1])
    confused_mask = (sorted_scores < max_zero) & (sorted_scores > min_one)
    confused_part = sorted_scores[confused_mask]
    return confused_part


def find_last_normal_idx(sorted_truth):
    """
    Return the index of the last 0 in the sorted_truth array

    :param sorted_truth: an array of 0 or 1
    :return: an int, the last idx of a 0 in sorted_truth
    """

    if sorted_truth[-1] == 0:
        return len(sorted_truth) - 1
    else:
        return len(sorted_truth) - np.argmax(np.diff(sorted_truth)[::-1]) - 2


def find_prop_markers(
    sorted_scores, sorted_truth, ratio_markers=(1, 0.75, 0.5), add_full_out_zone=True
):
    """
    Find the score thresholds starting at which the proportion of n_normal / n_total
    is on or below the values in ratio_markers. If the proportion is never reached, the
    proportions are skipped and the thresholds array is shorter than the ratio_markers tuple
    If add_full_out_zone is set to True, the zone where all scores are from abnormal sound is added and
    any zone above that one is removed.
    """

    tot_normal = 0
    marker_idx = 0
    marker = ratio_markers[marker_idx]
    thresholds = []
    stop_at_idx = len(sorted_scores) - 1
    if add_full_out_zone:
        stop_at_idx = find_last_normal_idx(sorted_truth)
    for idx, val in enumerate(sorted_truth):
        if idx <= stop_at_idx:
            if not val:
                tot_normal += 1
            ratio = tot_normal / (idx + 1)
            if ratio < marker:
                thresholds.append(idx)
                marker_idx += 1
                try:
                    marker = ratio_markers[marker_idx]
                except IndexError:
                    if add_full_out_zone:
                        thresholds.append(stop_at_idx)
                    thresholds = np.array(thresholds)
                    sorted_scores = np.array(sorted_scores)
                    return sorted_scores[thresholds]
        else:
            break
    if add_full_out_zone:
        thresholds.append(stop_at_idx)
    thresholds = np.array(thresholds)
    sorted_scores = np.array(sorted_scores)
    return sorted_scores[thresholds]


def get_percentiles(scores, n_percentiles):
    """
    A function computing the n_percentiles of scores. If n_percentiles is larger than
    len(scores), the scores are interpolated.

    >>> arr = [1, 2, 3, 4]
    >>> get_percentiles(arr, n_percentiles=1)
    array([3])
    >>> get_percentiles(arr, n_percentiles=2)
    array([2, 3])
    >>> get_percentiles(arr, n_percentiles=3)
    array([2, 3, 4])
    >>> get_percentiles(arr, n_percentiles=4)
    array([1, 2, 3, 4])
    >>> get_percentiles(arr, n_percentiles=5)
    array([1.5, 2. , 2.5, 3. , 3.5])
    >>> get_percentiles(arr, n_percentiles=6)
    array([1.5, 2. , 2.5, 3. , 3.5, 4. ])
    >>> get_percentiles(arr, n_percentiles=7)
    array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. ])
    """

    sorted_scores = np.sort(scores)
    n_scores = len(sorted_scores)
    diff = n_percentiles - n_scores
    if diff > 0:
        n_refine = int(np.ceil(diff / (n_scores - 1)))
        scores_steps = [i / (n_refine + 1) for i in np.diff(sorted_scores)]
        extended_scores = []
        for score, step in zip(sorted_scores, scores_steps):
            extended_scores.extend([score + step * i for i in range(n_refine + 1)])
        extended_scores.append(sorted_scores[-1])
        sorted_scores = np.array(extended_scores)
    n_scores = len(sorted_scores)
    percentiles_idx = np.array(
        [int(n_scores / (n_percentiles + 1) * i) for i in range(1, n_percentiles + 1)]
    )
    return sorted_scores[percentiles_idx]


# to use on train data
def get_confusion_zones_percentiles(scores, truth, n_percentiles=1):
    """
    Get the percentiles of the normal scores in the confused zone.

    :param scores: an array of outlier scores
    :param truth: an array of 0 for normal and 1 for abnormal
    :param n_percentiles: the number of percentiles required
    :return: an array of scores marking the boundary of the percentile zones
    """

    scores = np.array(scores)
    truth = np.array(truth)
    scores_normal = scores[truth == 0]
    scores_anomaly = scores[truth == 1]
    max_normal = np.max(scores_normal)
    min_anomaly = np.min(scores_anomaly)
    if max_normal < min_anomaly:
        return (max_normal, min_anomaly)
    elif n_percentiles == 1:
        return (min_anomaly, max_normal)
    else:
        confused_normal_scores = scores_normal[
            (min_anomaly <= scores_normal) & (scores_normal <= max_normal)
        ]
        confused_anomaly_scores = scores_anomaly[
            (min_anomaly <= scores_anomaly) & (scores_normal <= max_normal)
        ]
        if len(confused_normal_scores) > len(confused_anomaly_scores):
            most_numerous = confused_normal_scores
        else:
            most_numerous = confused_anomaly_scores
        percentiles = list(get_percentiles(most_numerous, n_percentiles - 1))
        if not max_normal in percentiles:
            percentiles.append(max_normal)
        if not min_anomaly in percentiles:
            percentiles.append(min_anomaly)
        percentiles.sort()
    return np.array(percentiles)


def get_confusion_zones_std(scores, truth=None, n_zones=6, std_per_zone=0.5):
    """
    Get a list of zones boundaries based on the standard deviation of the normal scores

    :param scores: an array of outlier scores
    :param truth: an array of 0 for normal and 1 for abnormal
    :param n_zones: the number of zones required
    :param std_per_zone: the number of standard deviation per zone
    :return: an array of scores marking the boundary of the percentile zones
    """

    scores = np.array(scores)
    if truth is None:
        truth = np.zeros(len(scores))
    else:
        truth = np.array(truth)
    normal_scores = scores[truth == 0]
    std = np.std(normal_scores)
    mean = np.mean(normal_scores)

    zones = np.array([mean + i * std * std_per_zone for i in range(n_zones)])
    return zones


def plot_scores_and_zones(scores, zones, box=None, title=None, lines=True):
    """
    Plot the scores on a timeline with color according to which zone they belong too, green under the first
    value in zones, red above the last and a shade from green to red.
    
    :param scores: an array of scores
    :param zones: the limit of the zones
    :param box: limits to display the plot
    :param title: name of the plot
    :param lines: whether to show lines at the limit of the zones
    :return:
    """

    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    scores = np.array(scores)

    # plotting all the green points:
    indices = np.arange(0, len(scores))
    green_mask = scores <= zones[0]
    green_scores = scores[green_mask]
    green_idx = indices[green_mask]

    red_mask = scores >= zones[-1]
    red_scores = scores[red_mask]
    red_idx = indices[red_mask]

    green_red_idx = np.concatenate((red_idx, green_idx))
    yellow_idx = indices[
        np.logical_and(np.logical_not(red_mask), np.logical_not(green_mask))
    ]
    yellow_scores = scores[yellow_idx]

    ax.scatter(x=green_idx, y=green_scores, c='g')
    # plotting all the red points:
    ax.scatter(x=red_idx, y=red_scores, c='r')

    # plotting everything in between:
    if len(zones) > 2:
        color = []
        # labels = [100 * 1 / (i + 1) for i in range(len(zones))]
        for score in yellow_scores:
            color.append(np.argmin(zones < score))
    else:
        color = yellow_scores
    if lines:
        for zone in zones:
            ax.axhline(zone, color='b', lw=0.08, alpha=1)

    ax.scatter(x=yellow_idx, y=yellow_scores, c=color, cmap='Wistia')
    if box is None:
        min_ = np.min(scores)
        max_ = np.max(scores)
        diff = max_ - min_
        box = (0, len(scores), min_ - 0.05 * diff, max_ + 0.05 * diff)
    ax.axis(box)
