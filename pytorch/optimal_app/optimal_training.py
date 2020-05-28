from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, Select, ColumnDataSource, Span, CustomJS, Div
from bokeh.plotting import figure
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
import randomcolor

from utils import clean_run, param_count, convert_to_logspace, day_ratio
from conversions import width_to_flo, hours_to_dollars, dollars_to_hours, hours_to_width, loss_fit, param_fit

########################################################################################################################
# Set up data
########################################################################################################################

df = pd.read_csv("../loss_vs_compute.csv")
loss_keys = [key for key in df.keys() if "loss" in key]

losses_per_run = {key: np.array(clean_run(list(zip(df["global_step"], df[key])))) for key in loss_keys}
bounds_per_run = {key: [min(value[:, 0]), max(value[:, 0])] for key, value in losses_per_run.items()}
params_per_run = {key: param_count(run) for key, run in losses_per_run.items()}
ordered_keys = sorted(loss_keys, key=lambda x: params_per_run[x])
losses_per_run = [losses_per_run[key] for key in ordered_keys]
bounds_per_run = [bounds_per_run[key] for key in ordered_keys]
params_per_run = [params_per_run[key] for key in ordered_keys]
general_bounds = bounds_per_run[2][0], bounds_per_run[-2][1]
generator = randomcolor.RandomColor(seed=0)
color_list = [generator.generate(hue="blue")[0] for _ in range(len(ordered_keys))]
# there's a bogus point of small coordinates at position 0 to get the ConvexHull facing the origin
# hacky, but it's the syntax here; qhull_options=QG0 means the ConvexHull facing point 0
all_points = np.array([(10e8, 3, -1)] + [(a, b, i) for i, run in enumerate(losses_per_run) for a, b in run if
                                         general_bounds[0] < a < general_bounds[1]])
all_hull = ConvexHull(all_points[:, :2], qhull_options='QG0')
log_points = np.array([(np.log(a), b) for a, b, i in all_points])
log_hull = ConvexHull(log_points, qhull_options='QG0')
indexed_runs = {i: np.array([(a, b) for a, b in run if general_bounds[0] < a < general_bounds[1]]) for i, run in
                enumerate(losses_per_run)}

########################################################################################################################
# Set up loss_plot
########################################################################################################################

loss_plot = figure(plot_height=400, plot_width=1200, title="Training curves and compute frontier",
                   tools="crosshair,pan,reset,save,wheel_zoom",
                   x_range=[general_bounds[0] * day_ratio, general_bounds[1] * day_ratio],
                   y_range=[min(all_points[1:, 1]), max(all_points[1:, 1])],
                   x_axis_type="log", y_axis_type="log",
                   x_axis_label="Floating-point operations (excluding embeddings & softmax)",
                   y_axis_label="Validation loss on Wikitext-103")
for i, run in indexed_runs.items():
    source = ColumnDataSource(data=dict(x=run[:, 0] * day_ratio, y=run[:, 1]))
    loss_plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6, color=color_list[i])
    loss_plot.scatter('x', 'y', source=source, line_width=1, line_alpha=0.6, color=color_list[i])
hull_indices = set(index for pair in all_hull.simplices[all_hull.good] for index in pair)
hull_indices = sorted(hull_indices, key=lambda x: all_points[x, 0])

########################################################################################################################
# Fit frontier
########################################################################################################################

hull_points = np.array([all_points[index] for index in hull_indices])
loss_popt, loss_pcov = curve_fit(loss_fit, hull_points[:, 0], hull_points[:, 1])
a, b, c = loss_popt
print(a, b, c)
display_abscisses = np.array([min(hull_points[:, 0]) / 1.25] + list(hull_points[:, 0]) +
                             [max(hull_points[:, 0]) * 1.25])
source = ColumnDataSource(
    data=dict(x=sorted(display_abscisses * day_ratio), y=loss_fit(sorted(display_abscisses), *loss_popt)))
loss_plot.line('x', 'y', source=source, line_width=1, line_alpha=0.8, color="red")

########################################################################################################################
# Set up param_plot
########################################################################################################################

param_plot = figure(plot_height=400, plot_width=1200,
                    title="Optimal number of parameters per floating-point operations budget",
                    tools="crosshair,pan,reset,save,wheel_zoom",
                    x_range=loss_plot.x_range,
                    y_range=[min(params_per_run), max(params_per_run)],
                    x_axis_type="log", y_axis_type="log",
                    x_axis_label="Floating-point operations (excluding embeddings & softmax)",
                    y_axis_label="Parameter number of the best-performing model")

logspace_points = convert_to_logspace(all_points, *loss_popt)
logspace_losses_per_run = [convert_to_logspace(run, *loss_popt) for run in losses_per_run]
passing_points = []
for run_index, log_run in enumerate(logspace_losses_per_run):
    current_point = None
    passed = False
    difference = log_run[:, 1] - log_run[:, 0]
    passing_points.append(np.argmax(difference))
compute_at_passing_points = np.array([(losses_per_run[i][passing_point, 0], params_per_run[i])
                                      for i, passing_point in enumerate(passing_points)])
compute_at_hull = np.array([(losses_per_run[i][passing_point, 0], params_per_run[i])
                            for i, passing_point in enumerate(passing_points) if i in set(hull_points[:, 2])])
run_indices_at_hull = [i for i, passing_point in enumerate(passing_points) if i in set(hull_points[:, 2])]

param_popt, param_pcov = curve_fit(param_fit, compute_at_hull[:, 0], np.log(compute_at_hull[:, 1]))
d, e, f = param_popt

for i, run_apex in enumerate(compute_at_hull):
    source = ColumnDataSource(data=dict(x=[compute_at_hull[i, 0] * day_ratio], y=[compute_at_hull[i, 1]]))
    param_plot.scatter('x', 'y', source=source, color=color_list[run_indices_at_hull[i]])
display_abscisses = np.array([min(compute_at_hull[:, 0]) / 1.25] + list(compute_at_hull[:, 0]) +
                             [max(compute_at_hull[:, 0]) * 1.25])
source = ColumnDataSource(data=dict(x=display_abscisses * day_ratio,
                                    y=np.exp(param_fit(display_abscisses, d, e, f))))
param_plot.line('x', 'y', source=source, line_width=1, line_alpha=0.8, color="orange")

########################################################################################################################
# Set up widgets
########################################################################################################################

hours_end = 24
gpu_dropdown = Select(title="GPU", options=["V100", "P100", "P4", "K80", "V100 (without tensor cores and cudnn.benchmark)"],
                      value="V100")
amp_mode_dropdown = Select(title="AMP mode", options=["O0", "O1", "O2"], value="O0")
hours = Slider(title="Wall time (hours)", value=1, start=0.0, end=hours_end, step=1 / 100)
dollars = Slider(title="Budget (dollars)", value=0.0, start=0.0, end=hours_to_dollars(hours_end, gpu_dropdown.value),
                 step=1 / 100)
kWh = Slider(title="Power (kWh)", value=0.0, start=0.0, end=5e6, step=1)
co2 = Slider(title="CO2 (Tons)", value=0.0, start=0.0, end=5e3, step=1)
slider_moves = {"hours": 0, "dollars": 0, "kWh": 0, "co2": 0}
n_sliders = len(slider_moves)

width = hours_to_width(hours.value, gpu_dropdown.value, amp_mode_dropdown.value, param_popt)
flo = width_to_flo(width, param_popt)
optimal_params = np.exp(param_fit(flo / 24 / 3600, *param_popt))
final_loss = loss_fit(flo / 24 / 3600, *loss_popt)

flo_line = Span(location=flo, line_alpha=0.7,
                dimension='height', line_color='purple',
                line_dash='dashed', line_width=1)
loss_line = Span(location=final_loss, line_alpha=0.7,
                dimension='width', line_color='red',
                line_dash='dashed', line_width=1)
param_line = Span(location=optimal_params, line_alpha=0.7,
                dimension='width', line_color='orange',
                line_dash='dashed', line_width=1)
loss_plot.add_layout(flo_line)
loss_plot.add_layout(loss_line)
param_plot.add_layout(flo_line)
param_plot.add_layout(param_line)

div = Div(text="""Optimal param number | Expected loss""", width=400, height=400)


def hours_update(attrname, old, new):
    slider_moves["hours"] += 1

    # if hours was the first updated slider
    if sum(slider_moves.values()) <= n_sliders * slider_moves["hours"] - n_sliders + 1:
        dollars.value = hours_to_dollars(hours.value, gpu_dropdown.value)
        kWh.value = hours.value * 20000
        co2.value = hours.value * 10

    width = hours_to_width(new, gpu_dropdown.value, amp_mode_dropdown.value, param_popt)
    flo = width_to_flo(width, param_popt)
    flo_line.location = flo
    optimal_params = np.exp(param_fit(flo / 24 / 3600, *param_popt))
    final_loss = loss_fit(flo / 24 / 3600, *loss_popt)
    loss_line.location = final_loss
    param_line.location = optimal_params
    div.text = "Optimal param number {:.2e} | Expected loss {:.2e}".format(optimal_params, final_loss)


def dollars_update(attrname, old, new):
    slider_moves["dollars"] += 1

    # if hours was the first updated slider
    if sum(slider_moves.values()) <= n_sliders * slider_moves["dollars"] - n_sliders + 1:
        hours.value = dollars_to_hours(dollars.value, gpu_dropdown.value)
        kWh.value = hours.value * 20000
        co2.value = kWh.value / 2000


def kWh_update(attrname, old, new):
    slider_moves["kWh"] += 1

    # if hours was the first updated slider
    if sum(slider_moves.values()) <= n_sliders * slider_moves["kWh"] - n_sliders + 1:
        hours.value = kWh.value / 20000
        dollars.value = hours_to_dollars(hours.value, gpu_dropdown.value)
        co2.value = kWh.value / 2000


def co2_update(attrname, old, new):
    slider_moves["co2"] += 1

    # if hours was the first updated slider
    if sum(slider_moves.values()) <= n_sliders * slider_moves["co2"] - n_sliders + 1:
        hours.value = co2.value / 10
        dollars.value = hours_to_dollars(hours.value, gpu_dropdown.value)
        kWh.value = hours.value * 20000


def gpu_update(attrname, old, new):
    if dollars_to_hours(dollars.value, gpu_dropdown.value) == hours.value:
        width = hours_to_width(hours.value, gpu_dropdown.value, amp_mode_dropdown.value, param_popt)
        flo = width_to_flo(width, param_popt)
        flo_line.location = flo
        optimal_params = np.exp(param_fit(flo / 24 / 3600, *param_popt))
        final_loss = loss_fit(flo / 24 / 3600, *loss_popt)
        loss_line.location = final_loss
        param_line.location = optimal_params
        div.text = "Optimal param number {:.2e} | Expected loss {:.2e}".format(optimal_params, final_loss)
    else:
        dollars.end = hours_to_dollars(hours_end, new)
        hours.value = dollars_to_hours(dollars.value, gpu_dropdown.value)


def amp_update(attrname, old, new):
    width = hours_to_width(hours.value, gpu_dropdown.value, amp_mode_dropdown.value, param_popt)
    flo = width_to_flo(width, param_popt)
    flo_line.location = flo
    optimal_params = np.exp(param_fit(flo / 24 / 3600, *param_popt))
    final_loss = loss_fit(flo / 24 / 3600, *loss_popt)
    loss_line.location = final_loss
    param_line.location = optimal_params
    div.text = "Optimal param number {:.2e} | Expected loss {:.2e}".format(optimal_params, final_loss)


hours.on_change('value', hours_update)
dollars.on_change('value', dollars_update)
kWh.on_change('value', kWh_update)
co2.on_change('value', co2_update)
gpu_dropdown.on_change("value", gpu_update)
amp_mode_dropdown.on_change("value", amp_update)

########################################################################################################################
# Set up layouts and add to document
########################################################################################################################

inputs = column(gpu_dropdown, amp_mode_dropdown, hours, dollars, kWh, co2)

curdoc().add_root(row(column(inputs, div), column(loss_plot, param_plot), width=800))
curdoc().title = "How big should my language model be ?"
