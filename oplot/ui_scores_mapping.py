"""Functions to create and plot outlier scores (or other) in a fixed bounded range. Intended to use to
show the results of an outlier algorithm in a user friendly UI"""
import numpy as np


def make_linear_part(max_score, min_score):
    """
    :param bottom: the proportion of the graph used for the bottom "sigmoid"
    :param middle: the proportion of the graph used for the middle linear part
    :param top: the proportion of the graph used for the top "sigmoid"
    :param max_score: the maximum score seen on train
    :param min_score: the minimum score seen on train
    :return: the linear part of the ui score mapping
    """

    slope = 1 / (max_score - min_score)

    def linear_part(x):
        return x * slope + 1 - slope * min_score

    return linear_part


def make_top_part(base, max_score, min_score):
    """
    The base has to be between 0 and 1, strictly.
    The function will be of the form -base ** (-x + t) + C, where t and C
    are the two constants to solve for. The constraints are continuity and
    smoothness at max_score when pieced with the linear part
    """

    slope = 1 / (max_score - min_score)
    t = np.log(slope / np.log(base)) / np.log(base) + max_score
    # at the limit when x->inf, the function will approach c
    c = 2 + base ** (-max_score + t)

    def top_part(x):
        return -(base ** (-x + t)) + c

    return top_part, c


def make_bottom_part(base, max_score, min_score):
    """
    The base has to be between 0 and 1, strictly.
    The function will be of the form -base ** (-x + t) + C, where t and C
    are the two constants to solve for. The constraints are continuity and
    smoothness at max_score when pieced with the linear part
    """

    slope = 1 / (max_score - min_score)
    t = np.log(slope / np.log(base)) / np.log(base) - min_score
    # at the limit when x->-inf, the function will approach c
    c = 1 - base ** (min_score + t)

    def bottom_part(x):
        return base ** (x + t) + c

    return bottom_part, c


def make_ui_score_mapping(
    min_lin_score, max_lin_score, top_base=2, bottom_base=2, max_score=10, reverse=False
):
    """
    Plot a sigmoid function to map outlier scores to (by default) the range (0, 10)
    The function is not only continuous but also smooth and the radius of the corners are controlled by the floats
    top_base and bottom_base

    Args:

        min_lin_score: float, the minimum scores which is map with a linear function
        max_lin_score: float, the maximum scores which is map with a linear function
        top_base: float, the base of the exponential function on top of the linear part
        bottom_base:  float, the base of the exponential function on the bottom of the linear part
        max_score: float, the upper bound of the function
        reverse: boolean, whether to mirror the function along its center
    
    Returns:
        a mapping, sigmoid like

    Examples:

    >>> from oplot.ui_scores_mapping import make_ui_score_mapping
    >>> import numpy as np
    >>> import matplotlib,pyplot as plt

    >>> sigmoid_map = make_ui_score_mapping(min_lin_score=1,
    ...                                max_lin_score=9,
    ...                                 top_base=2,
    ...                                 bottom_base=2,
    ...                                 max_score=10)

    >>> x = np.linspace(-5, 15, 100)
    >>> plt.plot(x, [sigmoid_map(i) for i in x])

    """

    linear_part = make_linear_part(max_lin_score, min_lin_score)
    bottom_part, min_ = make_bottom_part(bottom_base, max_lin_score, min_lin_score)
    top_part, max_ = make_top_part(top_base, max_lin_score, min_lin_score)
    if reverse:

        def ui_score_mapping(x):
            if x < min_lin_score:
                return max_score - max_score * (bottom_part(x) - min_) / (max_ - min_)
            if x > max_lin_score:
                return max_score - max_score * (top_part(x) - min_) / (max_ - min_)
            else:
                return max_score - max_score * (linear_part(x) - min_) / (max_ - min_)

    else:

        def ui_score_mapping(x):
            if x < min_lin_score:
                return max_score * (bottom_part(x) - min_) / (max_ - min_)
            if x > max_lin_score:
                return max_score * (top_part(x) - min_) / (max_ - min_)
            else:
                return max_score * (linear_part(x) - min_) / (max_ - min_)

    return ui_score_mapping


def between_percentiles_mean(scores, min_percentile=0.450, max_percentile=0.55):
    """
    Get the mean of the scores between the specified percentiles
    """
    import numpy

    scores = numpy.array(scores)
    sorted_scores = numpy.sort(scores)
    high_scores = sorted_scores[
        int(min_percentile * len(sorted_scores)) : int(
            max_percentile * len(sorted_scores)
        )
    ]
    return numpy.mean(high_scores)


def tune_ui_map(
    scores,
    truth=None,
    all_normal=True,
    min_percentile_normal=0.25,
    max_percentile_normal=0.75,
    min_percentile_abnormal=0.25,
    max_percentile_abnormal=0.75,
    lower_base=10,
    upper_base=10,
    abnormal_fact=2,
):
    """
    Construct a ui scores map spreading out the scores between 0 and 10, where high means normal. Scores is
    an array of raw stroll scores. NOTE: it assumes large scores means abnormal, small means normal!! Need to adapt
    otherwise.

    LOWERING the default range for the normal scores from [0.25, 0.75] to say [0., 0.25] will DECREASE the average
    quality score of normal sounds.

    INCREASING the range for the abnormal scores from [0.25, 0.75] to say [0.5, 1.0] will DECREASE the average quality
    score of abnormal sounds.
    """

    scores = np.array(scores)
    # we have examples of normal and abnormal
    if truth is not None and len(set(truth)) == 2:
        truth = np.array(truth)
        median_normal = between_percentiles_mean(
            scores[truth == 0],
            min_percentile=min_percentile_normal,
            max_percentile=max_percentile_normal,
        )
        median_abnormal = between_percentiles_mean(
            scores[truth == 1],
            min_percentile=min_percentile_abnormal,
            max_percentile=max_percentile_abnormal,
        )

    # if not the scores are all normal
    elif all_normal:
        median_normal = between_percentiles_mean(
            scores,
            min_percentile=min_percentile_normal,
            max_percentile=max_percentile_normal,
        )

        normal_large = between_percentiles_mean(
            scores, min_percentile=0.9, max_percentile=1
        )

        # as an approximation of the median abnormal, we use the media
        median_abnormal = normal_large * abnormal_fact

    # probably never useful, in case all scores are from abnormal
    else:
        median_abnormal = between_percentiles_mean(
            scores,
            min_percentile=min_percentile_abnormal,
            max_percentile=max_percentile_abnormal,
        )
        median_normal = median_abnormal / 10

    return median_normal, median_abnormal, lower_base, upper_base
