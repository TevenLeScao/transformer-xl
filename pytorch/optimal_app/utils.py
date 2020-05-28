import copy
import numpy as np


def clean_run(run):
    return [(a, float(b)) for a, b in run if b != "undefined"]


def param_count(run):
    compute_per_eval = run[0][0]
    return round(compute_per_eval / 4000 / 150 / 60 / 6 * day_ratio)


def convert_to_logspace(run, a, b, c):
    logspace_run = copy.deepcopy(run)
    logspace_run[:, 0] = b * np.log(run[:, 0])
    logspace_run[:, 1] = -np.log(run[:, 1] - c) + np.log(a)
    return logspace_run


# OpenAI used another unit for floating-point operations with a ratio of the number of seconds in a day; we'll display
# the raw number, but do the calculations with the ratio as it can overflow without it (convex hull notably fails)


day_ratio = 24 * 3600
