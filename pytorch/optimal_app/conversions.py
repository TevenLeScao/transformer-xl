import math
import numpy as np
from scipy.optimize import root
from utils import day_ratio

depth_width_ratio = 128

constants_per_gpu = {
    "V100": [2.21527743e+07, 1.18538628e+00, 1.43150104e+00, 1.66015023e+00,
             1.32808220e+00, 5.91503856e+00],
    "V100 (without tensor cores and cudnn.benchmark)": [1.82997989e+07, 1.05349588e+00, 1.25312127e+00, 1.67071294e+00,
                                                        1.44610885e+00, 5.55824273e+00],
    "P100": [6.01863899e+07, 9.23656025e-01, 1.03230702e+00, 1.46733667e+00,
             1.03031298e+00, 5.38021875e+00],
    "P4": [4.84472202e+07, 9.86822195e-01, 1.23474901e+00, 1.38493518e+00,
           1.04630858e+00, 1.03572754e+01],
    "K80": [2.58592374e+07, 6.42050890e-01, 7.06115162e-01, 1.44360777e+00,
            7.50695980e-01, 6.25951436e+00]

}

price_per_gpu = {
    "K80": 0.584,
    "P4": 0.689,
    "V100": 2.005,
    "V100 (without tensor cores and cudnn.benchmark)": 2.005,
    "P100": 1.416,
}

optimal_batch_size_per_gpu = {
    "P4": 16,
    "V100": 64,
    "V100 (without tensor cores and cudnn.benchmark)": 64,
    "P100": 64,
    "K80": 16
}

features_per_amp_mode = {
    "O0": (1, 0, 0),
    "O1": (0, 1, 0),
    "O2": (0, 0, 1)
}


def flo_speed(features, constants):
    k, k1, k2, b, c, layer_base = constants
    o0, o1, o2, x, y, z = features
    return k * np.power(k1, o1) * np.power(k2, o2) * x / (x + layer_base) * np.power(y, b) * np.power(np.log(z + 1), c)


def param_polynomial(width):
    return 7 / depth_width_ratio * (width ** 3) + 8 / depth_width_ratio * (width ** 2 + 3) + 3 * width + 3


def optimal_model_shape(width, param_number):
    depth = width / depth_width_ratio


def hours_to_width(hours, gpu, amp_mode, param_popt):
    seconds = hours * 3600
    d, e, f = param_popt
    constants = constants_per_gpu[gpu]
    amp_features = features_per_amp_mode[amp_mode]

    def equation_function(width):
        return np.power((param_polynomial(width) - f) / d, 1 / e) / flo_speed(
            (*amp_features, width / depth_width_ratio, width, optimal_batch_size_per_gpu[gpu]),
            constants) * day_ratio - seconds

    solution_array = root(equation_function, np.array([127]), method="hybr").x
    width = solution_array[0]
    print("width: {}".format(math.floor(width)))
    print("depth: {}".format(width / depth_width_ratio))
    print("param number: {:.4e}".format(param_polynomial(width)))
    speed = flo_speed((*amp_features, width / depth_width_ratio, width, optimal_batch_size_per_gpu[gpu]), constants)
    print("speed: {:.4e}".format(speed))
    print("flos from speed: {:.4e}".format(seconds * speed))
    print("flos from params: {:.4e}".format(np.power((param_polynomial(width) - f) / d, 1 / e) * day_ratio))
    print("params from flos: {:.4e}".format(np.exp(param_fit(speed * seconds / day_ratio, *param_popt))))
    return solution_array[0]


def width_to_flo(width, param_popt):
    d, e, f = param_popt
    return np.power((param_polynomial(width) - f) / d, 1 / e) * day_ratio


def loss_fit(x, a, b, c):
    return a * np.power(x, -b) + c


def param_fit(x, d, e, f):
    return np.log(d * np.power(x, e) + f)


def hours_to_dollars(hours, gpu):
    return hours * price_per_gpu[gpu]


def dollars_to_hours(dollars, gpu):
    return dollars / price_per_gpu[gpu]


def hours_to_kWh(hours, gpu):
    return hours * 1e13


def hours_to_co2(hours, gpu):
    return hours * 1e13
