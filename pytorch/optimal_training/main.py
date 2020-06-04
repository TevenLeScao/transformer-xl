from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, Select, ColumnDataSource, Span, Div, Button, LogColorMapper, ColorBar, LogTicker
from bokeh.models.tools import CrosshairTool
from bokeh.plotting import figure
from bokeh.events import Tap
from bokeh.transform import log_cmap
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
from time import sleep

from utils import *
from conversions import *

########################################################################################################################
# Basic dimensions
########################################################################################################################

plot_width = 1200
plot_height = 400
sidebar_width = 400
in_text_plot_width = 800
in_text_plot_height = 300

########################################################################################################################
# Set up data
########################################################################################################################

df = pd.read_csv("optimal_training/static/loss_vs_compute.csv")
loss_keys = [key for key in df.keys() if "loss" in key]

losses_per_run = {key: np.array(clean_run(list(zip(df["global_step"], df[key])))) for key in loss_keys}
losses_per_run = {k: v for k, v in losses_per_run.items() if len(v) > 5}
bounds_per_run = {key: [min(value[:, 0]), max(value[:, 0])] for key, value in losses_per_run.items()}
params_per_run = {key: param_count(run) for key, run in losses_per_run.items()}
ordered_keys = sorted(losses_per_run, key=lambda x: params_per_run[x])
losses_per_run = [losses_per_run[key] for key in ordered_keys]
bounds_per_run = [bounds_per_run[key] for key in ordered_keys]
params_per_run = [params_per_run[key] for key in ordered_keys]
palette = "Viridis256"
color_mapper = LogColorMapper(palette=palette, low=min(params_per_run), high=max(params_per_run))
general_bounds = bounds_per_run[2][0], bounds_per_run[-2][1]
print("{:.4e}, {:.4e}".format(general_bounds[0] * day_ratio, general_bounds[1] * day_ratio))
color_list = ["#000000" in params_per_run]
# there's a bogus point of small coordinates at position 0 to get the ConvexHull facing the origin
# hacky, but it's the syntax here, qhull_options=QG0 means the ConvexHull facing point 0
bounded_points = np.array([(10e8, 3, -1)] + [(a, b, i) for i, run in enumerate(losses_per_run) for a, b in run if
                                             general_bounds[0] < a < general_bounds[1]])
all_points = np.array([(a, b, i) for i, run in enumerate(losses_per_run) for a, b in run])
all_hull = ConvexHull(bounded_points[:, :2], qhull_options='QG0')
log_points = np.array([(np.log(a), b) for a, b, i in bounded_points])
log_hull = ConvexHull(log_points, qhull_options='QG0')
indexed_runs = [np.array([(a, b) for a, b in run]) for run in losses_per_run]

########################################################################################################################
# Set up loss_plot
########################################################################################################################

color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(), label_standoff=12,
                     border_line_color=None, location=(0, 0), title="Num of params")
loss_plot = figure(plot_height=plot_height, plot_width=plot_width,
                   title="Validation loss during training for an array of models of different sizes",
                   tools="pan,reset,save,wheel_zoom,tap", active_scroll="wheel_zoom",
                   x_range=[min(all_points[:, 0]) * day_ratio, max(all_points[:, 0]) * day_ratio],
                   y_range=[min(all_points[:, 1]), max(all_points[:, 1])],
                   x_axis_type="log", y_axis_type="log",
                   x_axis_label="Floating-point operations (excluding embeddings & softmax)",
                   y_axis_label="Validation loss on Wikitext-103")
loss_plot.add_tools(CrosshairTool(dimensions="width", line_alpha=0.2))
loss_plot.add_layout(color_bar, "left")
# for i, run in indexed_runs.items():
#     source = ColumnDataSource(data=dict(x=run[:, 0] * day_ratio, y=run[:, 1]))
#     loss_plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6, color=color_list[i])
#     loss_plot.scatter('x', 'y', source=source, line_width=1, line_alpha=0.6, color=color_list[i])

source = ColumnDataSource(data=dict(
    xs=[run[:, 0] * day_ratio for run in indexed_runs],  # x coords for each line (list of lists)
    ys=[run[:, 1] for run in indexed_runs],  # y coords for each line (list of lists)
    params=params_per_run  # data to use for colormapping
))
loss_plot.multi_line('xs', 'ys', source=source,
                     color=log_cmap('params', palette, min(params_per_run), max(params_per_run)))
source = ColumnDataSource(data=dict(
    x=[compute for run in indexed_runs for compute in run[:, 0] * day_ratio],  # x coords for each line (list of lists)
    y=[loss for run in indexed_runs for loss in run[:, 1] ],  # y coords for each line (list of lists)
    params=[repeated_params for i, params in enumerate(params_per_run)
            for repeated_params in [params] * len(indexed_runs[i])]  # data to use for colormapping
))
loss_plot.scatter('x', 'y', source=source,
                  color=log_cmap('params', palette, min(params_per_run), max(params_per_run)), size=3)

hull_indices = set(index for pair in all_hull.simplices[all_hull.good] for index in pair)
hull_indices = sorted(hull_indices, key=lambda x: bounded_points[x, 0])

########################################################################################################################
# Fit frontier
########################################################################################################################

hull_points = np.array([bounded_points[index] for index in hull_indices])
loss_popt, loss_pcov = curve_fit(loss_fit, hull_points[:, 0], hull_points[:, 1])
a, b, c = loss_popt
print(a, b, c)
display_abscisses = np.array([min(all_points[:, 0]) / 1.25] + sorted(list(all_points[:, 0])) +
                             [max(all_points[:, 0]) * 1.25])
source = ColumnDataSource(
    data=dict(x=sorted(display_abscisses * day_ratio), y=loss_fit(sorted(display_abscisses), *loss_popt)))
loss_plot.line('x', 'y', source=source, line_width=1, line_alpha=0.8, color="red")

########################################################################################################################
# Set up param_plot
########################################################################################################################

param_plot = figure(plot_height=plot_height, plot_width=plot_width,
                    title="Optimal number of non-embedding parameters per floating-point operations budget",
                    tools="pan,reset,save,wheel_zoom,tap", active_scroll="wheel_zoom",
                    x_range=loss_plot.x_range,
                    y_range=[min(params_per_run), max(params_per_run)],
                    x_axis_type="log", y_axis_type="log",
                    x_axis_label="Floating-point operations (excluding embeddings & softmax)",
                    y_axis_label="Optimal number of non-embedding parameters")
param_plot.add_tools(CrosshairTool(dimensions="width", line_alpha=0.2))
param_plot.add_layout(color_bar, "left")

logspace_points = convert_to_logspace(bounded_points, *loss_popt)
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

source = ColumnDataSource(data=dict(x=compute_at_hull[:, 0] * day_ratio,
                                    y=compute_at_hull[:, 1],
                                    params=[params for i, params in enumerate(params_per_run) if
                                            i in set(hull_points[:, 2])]))
param_plot.scatter('x', 'y', source=source,
                   color=log_cmap('params', palette, min(params_per_run), max(params_per_run)))
display_abscisses = np.array([min(compute_at_hull[:, 0]) / 1.25] + sorted(list(compute_at_hull[:, 0])) +
                             [max(compute_at_hull[:, 0]) * 1.25])
source = ColumnDataSource(data=dict(x=display_abscisses * day_ratio,
                                    y=safe_flo_to_param(display_abscisses, d, e, f)))
param_plot.line('x', 'y', source=source, line_width=1, line_alpha=0.8, color="orange")

########################################################################################################################
# Set up widgets
########################################################################################################################

hours_end = 24
hours_initial = 3.23
gpu_dropdown = Select(title="GPU",
                      options=["V100", "P100", "P4", "K80", ],
                      value="V100", width=sidebar_width, sizing_mode="stretch_width")
amp_mode_dropdown = Select(title="AMP mode", options=["O0", "O1", "O2"], value="O0", width=sidebar_width,
                           sizing_mode="stretch_width")
tipping_width = tipping_point(gpu_dropdown.value, amp_mode_dropdown.value, param_popt)
tip = {}
update_tip(tip, tipping_width, gpu_dropdown.value, amp_mode_dropdown.value, loss_popt, param_popt)
hours_slider = Slider(title="Wall time (hours)", value=hours_initial, start=tip["hours"], end=hours_end, step=1 / 100,
                      width=sidebar_width, sizing_mode="stretch_width")
dollars_slider = Slider(title="Budget (dollars)", value=hours_to_dollars(hours_initial, gpu_dropdown.value),
                        start=dollars_to_hours(tip["hours"], gpu_dropdown.value),
                        end=hours_to_dollars(hours_end, gpu_dropdown.value),
                        step=1 / 100, width=sidebar_width, sizing_mode="stretch_width")
input_buffer = Div(text="", width=sidebar_width, height=10,
                   style={"display": "block", "margin": "0 auto", "width": f"{sidebar_width}px",
                          "text-align": 'center'})
top_sidebar_div_style = {"display": "block", "margin": "0 auto", 'font-size': "125%",
                         "width": f"{sidebar_width}px", "text-align": 'center'}
kWh_text = Div(text=kWh_fill(hours_to_kWh(hours_slider.value, gpu_dropdown.value)), width=sidebar_width, height=45,
               style=top_sidebar_div_style)
co2_text = Div(text=co2_fill(hours_to_co2(hours_slider.value, gpu_dropdown.value)), width=sidebar_width, height=45,
               style=top_sidebar_div_style)
slider_moves = {"hours": 0, "dollars": 0, "kWh": 0, "co2": 0}
n_sliders = len(slider_moves)

width = hours_to_width(hours_slider.value, gpu_dropdown.value, amp_mode_dropdown.value, param_popt)
flo = width_to_flo(width, *param_popt)
optimal_params = safe_flo_to_param(flo / 24 / 3600, *param_popt)
final_loss = loss_fit(flo / 24 / 3600, *loss_popt)
example_shape = {}
example_shape['example_depth'], example_shape['example_width'] = optimal_model_shape(width, optimal_params)
example_shape['alternate_depth'], example_shape['alternate_width'] = alternate_model_shape(width, optimal_params)

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

sidebar_div_style = {"display": "block", "margin": "0 auto", "width": f"{sidebar_width}px", "text-align": 'center'}
big_sidebar_div_style = {"display": "block", "margin": "0 auto", "width": f"{sidebar_width}px",
                         "text-align": 'center', 'font-size': "200%", 'font-weight': "bold"}
static_loss_text = Div(text="Expected wt-103 validation loss:", width=sidebar_width, height=10, style=sidebar_div_style)
optimal_loss_text = Div(text="{:.2f}".format(final_loss), width=sidebar_width, height=45,
                        style={"display": "block", "margin": "0 auto", 'font-size': "200%",
                               'font-weight': "bold", "width": f"{sidebar_width}px", "text-align": 'center'})
static_param_text = Div(text="Optimal number of non-embedding parameters:", width=sidebar_width, height=10,
                        style=sidebar_div_style)
optimal_param_text = Div(text="{:.2e}".format(optimal_params), width=sidebar_width, height=45,
                         style=big_sidebar_div_style)
static_shape_text = Div(text="For example, this could be a model of", width=sidebar_width, height=10,
                        style=sidebar_div_style)
optimal_shape_text = Div(text=f"{example_shape['example_depth']} layers of {example_shape['example_width']} dimensions",
                         width=sidebar_width, height=30, style=big_sidebar_div_style)
static_altshape_text = Div(text="Or a model of", width=sidebar_width, height=10, style=sidebar_div_style)
optimal_altshape_text = Div(
    text=f"{example_shape['alternate_depth']} layers of {example_shape['alternate_width']} dimensions",
    width=sidebar_width, height=30, style=big_sidebar_div_style)


def compare_and_update(width):
    if width >= tip["width"]:
        update_width(width)
        hours = width_to_hours(width, gpu_dropdown.value, amp_mode_dropdown.value, param_popt)
        hours_slider.value = hours
    else:
        width = min(tip["width"], width + 5)
        update_width(width)
        compare_and_update(width)


def update_width(width):
    flo = width_to_flo(width, *param_popt)
    flo_line.location = flo
    optimal_params = safe_flo_to_param(flo / 24 / 3600, *param_popt)
    final_loss = loss_fit(flo / 24 / 3600, *loss_popt)
    loss_line.location = final_loss
    param_line.location = optimal_params
    example_shape['example_depth'], example_shape['example_width'] = optimal_model_shape(width, optimal_params)
    example_shape['alternate_depth'], example_shape['alternate_width'] = alternate_model_shape(width, optimal_params)
    optimal_shape_text.text = f"{example_shape['example_depth']} layers of {example_shape['example_width']} dimensions"
    optimal_altshape_text.text = f"{example_shape['alternate_depth']} layers of {example_shape['alternate_width']} dimensions"
    optimal_param_text.text = "{:.2e}".format(optimal_params)
    optimal_loss_text.text = "{:.2f}".format(final_loss)


def hours_update(attrname, old, new):
    slider_moves["hours"] += 1

    # if hours was the first updated slider
    if sum(slider_moves.values()) <= n_sliders * slider_moves["hours"] - n_sliders + 1:
        dollars_slider.value = hours_to_dollars(hours_slider.value, gpu_dropdown.value)
        kWh_text.text = kWh_fill(hours_to_kWh(hours_slider.value, gpu_dropdown.value))
        co2_text.text = co2_fill(hours_to_co2(hours_slider.value, gpu_dropdown.value))

    width = hours_to_width(hours_slider.value, gpu_dropdown.value, amp_mode_dropdown.value, param_popt)
    update_width(width)


def dollars_update(attrname, old, new):
    slider_moves["dollars"] += 1

    # if hours was the first updated slider
    if sum(slider_moves.values()) <= n_sliders * slider_moves["dollars"] - n_sliders + 1:
        hours_slider.value = dollars_to_hours(dollars_slider.value, gpu_dropdown.value)
        kWh_text.text = kWh_fill(hours_to_kWh(hours_slider.value, gpu_dropdown.value))
        co2_text.text = co2_fill(hours_to_co2(hours_slider.value, gpu_dropdown.value))


def gpu_update(attrname, old, new):
    update_tip(tip, tipping_point(gpu_dropdown.value, amp_mode_dropdown.value, param_popt), gpu_dropdown.value,
               amp_mode_dropdown.value, loss_popt, param_popt)
    hours_slider.start = tip["hours"]
    dollars_slider.start = hours_to_dollars(tip["hours"], gpu_dropdown.value)
    if dollars_to_hours(dollars_slider.value, gpu_dropdown.value) == hours_slider.value:
        width = hours_to_width(hours_slider.value, gpu_dropdown.value, amp_mode_dropdown.value, param_popt)
        compare_and_update(width)
    else:
        dollars_slider.end = hours_to_dollars(hours_end, new)
        hours_slider.value = dollars_to_hours(dollars_slider.value, gpu_dropdown.value)
    kWh_text.text = kWh_fill(hours_to_kWh(hours_slider.value, gpu_dropdown.value))
    co2_text.text = co2_fill(hours_to_co2(hours_slider.value, gpu_dropdown.value))


def amp_update(attrname, old, new):
    update_tip(tip, tipping_point(gpu_dropdown.value, amp_mode_dropdown.value, param_popt), gpu_dropdown.value,
               amp_mode_dropdown.value, loss_popt, param_popt)
    width = hours_to_width(hours_slider.value, gpu_dropdown.value, amp_mode_dropdown.value, param_popt)
    hours_slider.start = tip["hours"]
    dollars_slider.start = hours_to_dollars(tip["hours"], gpu_dropdown.value)
    compare_and_update(width)
    kWh_text.text = kWh_fill(hours_to_kWh(hours_slider.value, gpu_dropdown.value))
    co2_text.text = co2_fill(hours_to_co2(hours_slider.value, gpu_dropdown.value))


def loss_tap(event):
    _, loss = event.x, event.y
    flo = loss_to_flo(loss, *loss_popt)
    param_number = safe_flo_to_param(flo, *param_popt)
    width = param_to_width(param_number)
    compare_and_update(width)


loss_plot.on_event(Tap, loss_tap)


def param_tap(event):
    _, param_number = event.x, event.y
    width = param_to_width(param_number)
    hours = width_to_hours(width, gpu_dropdown.value, amp_mode_dropdown.value, param_popt)
    hours_slider.value = hours


param_plot.on_event(Tap, param_tap)

hours_slider.on_change('value', hours_update)
dollars_slider.on_change('value', dollars_update)
gpu_dropdown.on_change("value", gpu_update)
amp_mode_dropdown.on_change("value", amp_update)


########################################################################################################################
# Buttons
########################################################################################################################

def on_optimal_click():
    code_box.text = hf_code(example_shape['example_width'], example_shape['example_depth'])


def on_alternate_click():
    code_box.text = hf_code(example_shape['alternate_width'], example_shape['alternate_depth'])


input_text = Div(text="Choose a GPU, AMP mode, and budget:", width=sidebar_width, height=30,
                 style={"display": "block", "margin": "0 auto", 'font-size': "125%",
                        'font-weight': "bold", "width": f"{sidebar_width}px", "text-align": 'center'})
initialize_optimal = Button(width=175, label="Initialize in 🤗transformers!")
initialize_optimal.align = "center"
initialize_optimal.on_click(on_optimal_click)
results_buffer = Div(text="", width=sidebar_width, height=5, style=sidebar_div_style)
initialize_alternate = Button(width=175, label="Initialize in 🤗transformers!")
initialize_alternate.align = "center"
initialize_alternate.on_click(on_alternate_click)

code_box_style = {"display": "block", "margin": "0 auto", "width": f"{sidebar_width + plot_width}px",
                  "text-align": 'center',
                  "white-space": "pre-wrap", "background": "#f4f4f4",
                  "border": "1px solid #ddd",
                  "border-left": "3px solid #f36d33",
                  "color": "#666",
                  "page-break-inside": "avoid",
                  "font-family": "monospace",
                  "font-size": "15px",
                  "line-height": "1.6",
                  "max-width": "100%",
                  "overflow": "hidden",
                  "min-height": "30px",
                  "word-wrap": "break-word"}
code_box = Div(text="Find the right model for you with the curves and sliders then click the buttons to display the "
                    "corresponding 🤗transformers code here!", width=sidebar_width + plot_width, style=code_box_style,
               sizing_mode="scale_width")
code_box.align = "center"

########################################################################################################################
# Add write-up text
########################################################################################################################

text_width = "800px"
main_text_style = {"min-height": "100px",
                   "overflow": "hidden",
                   "display": "block",
                   "margin": "auto",
                   "width": text_width,
                   "font-size": "18px"}

formula_img_style_1 = {"min-height": "25px",
                       "display": "block",
                       "margin": "0 auto",
                       "width": text_width,
                       "height": "auto",
                       "max-width": "100%",
                       "max-height": "100%"}

formula_img_style_2 = {"min-height": "50px",
                       "display": "block",
                       "margin": "0 auto",
                       "width": text_width,
                       "height": "auto",
                       "max-width": "100%",
                       "max-height": "100%"}

text_1 = Div(text=md1, style=main_text_style)
text_2 = Div(text=md2, style=main_text_style)
text_3 = Div(text=md3, style=main_text_style)
text_4 = Div(text=md4, style=main_text_style)

########################################################################################################################
# Loss plot in write-up
########################################################################################################################

in_text_loss_plot = figure(plot_height=in_text_plot_height, plot_width=in_text_plot_width,
                           title="Validation loss during training for an array of models of different sizes",
                           tools="pan,reset,save,wheel_zoom,tap", active_scroll="wheel_zoom",
                           x_range=[min(all_points[:, 0]) * day_ratio, max(all_points[:, 0]) * day_ratio],
                           y_range=[min(all_points[:, 1]), max(all_points[:, 1])],
                           x_axis_type="log", y_axis_type="log",
                           x_axis_label="Floating-point operations (excluding embeddings & softmax)",
                           y_axis_label="Validation loss on Wikitext-103")
in_text_loss_plot.add_layout(color_bar, "left")
in_text_loss_plot.align = "center"

source = ColumnDataSource(data=dict(
    xs=[run[:, 0] * day_ratio for run in indexed_runs],  # x coords for each line (list of lists)
    ys=[run[:, 1] for run in indexed_runs],  # y coords for each line (list of lists)
    params=params_per_run  # data to use for colormapping
))
in_text_loss_plot.multi_line('xs', 'ys', source=source,
                             color=log_cmap('params', palette, min(params_per_run), max(params_per_run)))
source = ColumnDataSource(data=dict(
    x=[compute for run in indexed_runs for compute in run[:, 0] * day_ratio],  # x coords for each line (list of lists)
    y=[loss for run in indexed_runs for loss in run[:, 1] ],  # y coords for each line (list of lists)
    params=[repeated_params for i, params in enumerate(params_per_run)
            for repeated_params in [params] * len(indexed_runs[i])]  # data to use for colormapping
))
in_text_loss_plot.scatter('x', 'y', source=source,
                  color=log_cmap('params', palette, min(params_per_run), max(params_per_run)), size=3)
# for i, run in indexed_runs.items():
#     source = ColumnDataSource(data=dict(x=run[:, 0] * day_ratio, y=run[:, 1]))
#     in_text_loss_plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6, color=color_list[i])
#     in_text_loss_plot.scatter('x', 'y', source=source, line_width=1, line_alpha=0.6, color=color_list[i])

in_text_param_plot = figure(plot_height=in_text_plot_height, plot_width=in_text_plot_width,
                            title="Optimal number of non-embedding parameters per floating-point operations budget",
                            tools="pan,reset,save,wheel_zoom,tap", active_scroll="wheel_zoom",
                            x_range=in_text_loss_plot.x_range,
                            y_range=[min(params_per_run), max(params_per_run)],
                            x_axis_type="log", y_axis_type="log",
                            x_axis_label="Floating-point operations (excluding embeddings & softmax)",
                            y_axis_label="Optimal number of non-embedding parameters")
in_text_param_plot.add_layout(color_bar, "left")
in_text_param_plot.align = "center"
# for i, run_apex in enumerate(compute_at_hull):
#     source = ColumnDataSource(data=dict(x=[compute_at_hull[i, 0] * day_ratio], y=[compute_at_hull[i, 1]]))
#     in_text_param_plot.scatter('x', 'y', source=source, color=color_list[run_indices_at_hull[i]])

source = ColumnDataSource(data=dict(x=compute_at_hull[:, 0] * day_ratio, y=compute_at_hull[:, 1],
                                    params=[params for i, params in enumerate(params_per_run) if
                                            i in set(hull_points[:, 2])]))
in_text_param_plot.scatter('x', 'y', source=source,
                           color=log_cmap('params', palette, min(params_per_run), max(params_per_run)))

training_button = Button(width=175, label="Fit!")
training_button.align = "center"
fit_button = Button(width=175, label="Fit!")
fit_button.align = "center"


def on_train_click():
    display_abscisses = np.array([min(all_points[:, 0]) / 1.25] + sorted(list(all_points[:, 0])) +
                                 [max(all_points[:, 0]) * 1.25])
    source = ColumnDataSource(
        data=dict(x=sorted(display_abscisses * day_ratio), y=loss_fit(sorted(display_abscisses), *loss_popt)))
    in_text_loss_plot.line('x', 'y', source=source, line_width=1, line_alpha=1, color="red")


def on_fit_click():
    display_abscisses = np.array([min(compute_at_hull[:, 0]) / 1.25] + sorted(list(compute_at_hull[:, 0])) +
                                 [max(compute_at_hull[:, 0]) * 1.25])
    source = ColumnDataSource(data=dict(x=display_abscisses * day_ratio,
                                        y=safe_flo_to_param(display_abscisses, d, e, f)))
    in_text_param_plot.line('x', 'y', source=source, line_width=1, line_alpha=0.8, color="orange")


training_button.on_click(on_train_click)
fit_button.on_click(on_fit_click)

before_text = column(text_1, training_button, in_text_loss_plot, text_2, fit_button, in_text_param_plot, text_3)
after_text = column(text_4)

########################################################################################################################
# Set up layouts and add to document
########################################################################################################################

inputs = column(input_text, gpu_dropdown, amp_mode_dropdown, hours_slider, dollars_slider, input_buffer, kWh_text,
                co2_text, sizing_mode="scale_width", width=sidebar_width, height=plot_height)

results = column(static_loss_text,
                 optimal_loss_text,
                 static_param_text,
                 optimal_param_text,
                 static_shape_text,
                 optimal_shape_text,
                 initialize_optimal,
                 results_buffer,
                 static_altshape_text,
                 optimal_altshape_text,
                 initialize_alternate, sizing_mode="scale_width", width=sidebar_width, height=plot_height)

# app = column(row(inputs, loss_plot, sizing_mode="scale_width"), row(results, param_plot, sizing_mode="scale_width"),
#              code_box, sizing_mode="scale_width")
app = column(row(column(inputs, results, sizing_mode="fixed"),
                 column(loss_plot, param_plot, sizing_mode="stretch_width", )),
             code_box, sizing_mode="scale_width")
before_text.align = "center"
app.align = "center"
after_text.align = "center"

main_body = column(before_text, app, after_text, sizing_mode="scale_width")

curdoc().add_root(main_body)
curdoc().title = "How big should my language model be ?"
