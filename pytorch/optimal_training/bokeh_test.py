''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput
from bokeh.plotting import figure

# Set up data
N = 200
x = np.linspace(0, 4 * np.pi, N)
y = np.sin(x)
source = ColumnDataSource(data=dict(x=x, y=y))

# Set up plot
plot = figure(plot_height=400, plot_width=400, title="my sine wave",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, 4 * np.pi], y_range=[-2.5, 2.5])

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

# Set up widgets
text = TextInput(title="title", value='my sine wave')
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)
amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
phase = Slider(title="phase", value=0.0, start=0.0, end=2 * np.pi)
freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)
slider_moves = {"offset": 0, "amplitude": 0, "phase": 0, "freq": 0}


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value


text.on_change('value', update_title)


def update_data(attrname, old, new):
    # Get the current slider values
    a = amplitude.value
    b = offset.value
    w = phase.value
    k = freq.value

    # Generate the new curve
    x = np.linspace(0, 4 * np.pi, N)
    y = a * np.sin(k * x + w) + b

    source.data = dict(x=x, y=y)


def offset_force(attrname, old, new):
    slider_moves["offset"] += 1

    if slider_moves["amplitude"] < slider_moves["offset"]:
        a = amplitude.value = offset.value
        w = phase.value = offset.value
        k = freq.value = offset.value
        b = offset.value
        x = np.linspace(0, 4 * np.pi, N)
        y = a * np.sin(k * x + w) + b

        source.data = dict(x=x, y=y)


def amp_force(attrname, old, new):
    slider_moves["amplitude"] += 1

    if slider_moves["offset"] < slider_moves["amplitude"]:
        b = offset.value = amplitude.value * 2
        w = phase.value = amplitude.value * 2
        k = freq.value = amplitude.value * 2
        a = amplitude.value
        x = np.linspace(0, 4 * np.pi, N)
        y = a * np.sin(k * x + w) + b

        source.data = dict(x=x, y=y)


for w in [phase, freq]:
    w.on_change('value', update_data)

offset.on_change('value', offset_force)
amplitude.on_change('value', amp_force)

# Set up layouts and add to document
inputs = column(text, offset, amplitude, phase, freq)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sliders"
