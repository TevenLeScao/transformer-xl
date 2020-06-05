import math
import numpy as np
from scipy.optimize import root

day_ratio = 24 * 3600

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

gpu_consumption = {
    "V100": 119.3495934959e-3,
    "V100 (without tensor cores and cudnn.benchmark)": 119.3495934959e-3,
    "K80": 142.42e-3,
    "P4": 55.27e-3,
    "P100": 139.65e-3
}

co2_intensity = 315 * 1e-6


def flo_speed(features, constants):
    k, k1, k2, b, c, layer_base = constants
    o0, o1, o2, x, y, z = features
    return k * np.power(k1, o1) * np.power(k2, o2) * x / (x + layer_base) * np.power(y, b) * np.power(np.log(z + 1), c)


def param_polynomial(width, depth=None, inner=None):
    if depth is not None:
        if inner is not None:
            return 5 * depth * (width ** 2) + 2 * depth * (width * inner) + 7 * depth * width + depth * inner + 3 * width + 3
        else:
            return 7 * depth * (width ** 2) + 8 * depth * width + 3 * width + 3
    else:
        if inner is not None:
            return 5 * depth_width_ratio * (width ** 3) + 2 * depth_width_ratio * (width ** 2 * inner) + 7 * depth_width_ratio * width ** 2 + depth_width_ratio * width * inner + 3 * width + 3
        else:
            return 7 / depth_width_ratio * (width ** 3) + 8 / depth_width_ratio * (width ** 2) + 3 * width + 3


def optimal_model_shape(width, param_number, base=8):
    depth = max(1, math.floor(width / depth_width_ratio))
    poly_params = np.array([depth * 7, depth * 8 + 3, 3 - param_number])
    roots = np.roots(poly_params)
    corresponding_width = int(base * round(max(roots) / base))
    return depth, corresponding_width


def alternate_model_shape(width, param_number, base=8):
    linear_depth = max(1, math.floor(width / depth_width_ratio))
    depth = max(linear_depth + 1, math.floor(0.4 * width ** 1.2 / depth_width_ratio))
    poly_params = np.array([depth * 7, depth * 8 + 3, 3 - param_number])
    roots = np.roots(poly_params)
    corresponding_width = int(base * round(max(roots) / base))
    return depth, corresponding_width


def hours_to_width(hours, gpu, amp_mode, param_popt):
    seconds = hours * 3600
    d, e, f = param_popt
    constants = constants_per_gpu[gpu]
    amp_features = features_per_amp_mode[amp_mode]

    def equation_function(width):
        return np.power((param_polynomial(width) - f) / d, 1 / e) / flo_speed(
            (*amp_features, width / depth_width_ratio, width, optimal_batch_size_per_gpu[gpu]),
            constants) * day_ratio - seconds

    width = iterative_solutions(equation_function, initial_guess=128)
    # print("width: {}".format(math.floor(width)))
    # print("depth: {}".format(width / depth_width_ratio))
    # print("param number: {:.4e}".format(param_polynomial(width)))
    speed = flo_speed((*amp_features, width / depth_width_ratio, width, optimal_batch_size_per_gpu[gpu]), constants)
    # print("speed: {:.4e}".format(speed))
    # print("flos from speed: {:.4e}".format(seconds * speed))
    # print("flos from params: {:.4e}".format(np.power((param_polynomial(width) - f) / d, 1 / e) * day_ratio))
    # print("params from flos: {:.4e}".format(np.exp(param_fit(speed * seconds / day_ratio, *param_popt))))
    return width


def iterative_solutions(equation_function, initial_guess):
    while initial_guess > 16:
        solution_array = root(equation_function, np.array([initial_guess]), method="hybr").x
        width = solution_array[0]
        should_be_zero = equation_function(width)
        if np.abs(should_be_zero) < 1e0:
            return width
        else:
            initial_guess *= 0.5
    return width


def width_to_flo(width, d, e, f):
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
    return hours * 3600 * gpu_consumption[gpu]


def hours_to_co2(hours, gpu):
    return hours * 3600 * gpu_consumption[gpu] * co2_intensity


def loss_to_flo(loss, a, b, c):
    return ((loss - c) / a) ** (-1 / b)


def param_to_flo(param_number, d, e, f):
    return ((param_number - f) / d) ** (1 / e)


def safe_flo_to_param(flo, d, e, f):
    return d * np.power(flo, e) + f


def param_to_width(param_number):
    poly_params = np.array([7 / depth_width_ratio, 8 / depth_width_ratio, 3, 3 - param_number])
    roots = np.roots(poly_params)
    real_roots = [np.real(candidate) for candidate in roots if np.imag(candidate) < 1e-5]
    width = max(real_roots)
    return width


def safe_param_to_width(param_number):
    try:
        return param_to_width(param_number)
    except np.linalg.LinAlgError:
        return safe_param_to_width(1.5 * param_number)


def width_to_hours(width, gpu, amp_mode, param_popt):
    d, e, f = param_popt
    constants = constants_per_gpu[gpu]
    amp_features = features_per_amp_mode[amp_mode]
    flos_from_params = np.power((param_polynomial(width) - f) / d, 1 / e) * day_ratio
    speed = flo_speed((*amp_features, width / depth_width_ratio, width, optimal_batch_size_per_gpu[gpu]), constants)
    seconds = flos_from_params / speed

    return seconds / 3600


def param_prime(width, depth=None):
    if depth is not None:
        return 14 * depth * (width ** 2) + 8 * depth + 3
    else:
        return 21 / depth_width_ratio * (width ** 2) + 16 / depth_width_ratio * width + 3


def flo_speed_prime(width, gpu, amp_mode):
    k, k1, k2, b, c, layer_base = constants_per_gpu[gpu]
    o0, o1, o2 = features_per_amp_mode[amp_mode]
    mult_constant = k * np.power(k1, o1) * np.power(k2, o2) * np.power(np.log(optimal_batch_size_per_gpu[gpu] + 1), c)
    return mult_constant * ((b + 1) * np.power(width, b) / (width + layer_base * depth_width_ratio)
                            - np.power(width, b + 1) / (width + layer_base * depth_width_ratio) ** 2)


# awful equation; we're trying to find the width for which lowering width actually makes the model less efficient
def tipping_point(gpu, amp_mode, param_popt):
    d, e, f = param_popt
    o0, o1, o2 = features_per_amp_mode[amp_mode]

    def equation_function(width):
        return np.power((param_polynomial(width) - f) / d, -1) / e * param_prime(width) / d \
               * flo_speed((o0, o1, o2, width / depth_width_ratio, width, optimal_batch_size_per_gpu[gpu]),
                           constants_per_gpu[gpu]) - \
               flo_speed_prime(width, gpu, amp_mode)

    tipping_width = iterative_solutions(equation_function, initial_guess=100)
    return tipping_width


def update_tip(tip, width, gpu, amp_mode, loss_popt, param_popt):
    a, b, c = loss_popt
    d, e, f = param_popt
    tip["width"] = width
    tip["param_number"] = param_polynomial(width)
    tip["flo"] = np.power((param_polynomial(tip["param_number"]) - f) / d, 1 / e)
    tip["loss"] = loss_fit(tip["flo"], a, b, c)
    tip["hours"] = width_to_hours(width, gpu, amp_mode, param_popt)
