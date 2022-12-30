import numpy as np
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots

number_kwargs = dict(type="number", value=0, style={'marginRight': '10px'})
app = Dash(__name__)
app.layout = html.Div([
    html.H3('Fourier Transformation'),
    dcc.Graph(id="graph"),
    html.P('Max HZ'),
    dcc.Slider(id='max_hz', min=0, max=24, value=0, step=1),
    html.P(["samples:    ", dcc.Input(id='num', type='number', min=2, max=1024, step=2, value=64, debounce=True)]),
    html.P(["Bias:       ", dcc.Input(id='bias', type='number', min=-5, max=5, value=0, debounce=True)]),
    html.P(["Amplitudes: ", dcc.Input(type='text', value='1 0 2 1.5', id='amplitudes', debounce=True), ]),
    html.P(["Phases:     ", dcc.Input(type='text', value='1.57 0 0.42 0.2', id='phases', debounce=True), ]),
])


@app.callback(
    Output("graph", "figure"),
    Input("max_hz", "value"),
    Input("num", "value"),
    Input('bias', 'value'),
    Input("amplitudes", "value"),
    Input("phases", "value")
)
def plot_sin(i, n_samples, bias, amps, phases):
    amps = list(map(float, amps.strip().split(' ')))
    phases = list(map(float, phases.strip().split(' ')))
    phases.extend([0] * (len(amps) - len(phases)))

    x = np.pi * np.linspace(0, 2, n_samples, endpoint=False)
    signals = [a * np.sin(x * (f + 1) + phi) for f, (a, phi) in enumerate(zip(amps, phases))]
    y = np.array(signals).sum(axis=0) + bias
    i += 1
    rft = np.fft.rfft(y)
    rft[i:] = 0
    smooth_y = np.fft.irfft(rft)

    fig = make_subplots(rows=1, cols=3, column_widths=[0.5, 0.25, 0.25],
                        subplot_titles=['Given signal', 'Complex amplitudes', 'Phases and Amplitudes'])
    fig.add_traces(data=[go.Scatter(x=x, y=y, mode='markers', name='data'),
                         go.Scatter(x=x, y=smooth_y, name='approximation', line=go.scatter.Line(width=3))])
    xrange = np.arange(len(x))
    fft = np.fft.fft(y)
    fig.add_trace(go.Scatter(x=xrange, y=rft.real, name='real'), col=2, row=1)
    fig.add_trace(go.Scatter(x=xrange, y=rft.imag, name='imag',
                             line=go.scatter.Line(dash='dot')), col=2, row=1)

    amplitudes = 2 * np.abs(rft) / len(x)
    amplitudes[0] /= 2
    frequencies = np.fft.fftfreq(len(x)) * len(x)
    phases = np.arctan2(rft.imag, rft.real)
    phases += np.pi / 2
    phases[amplitudes < 0.1] = 0
    phases[0] = 0
    fig.add_trace(go.Scatter(x=frequencies[:len(frequencies) // 2],
                             y=amplitudes[:len(fft) // 2], name='Amplitude'), col=3, row=1)
    fig.add_trace(go.Scatter(x=frequencies[:len(frequencies) // 2],
                             y=phases[:len(fft) // 2], name='Phase', line=go.scatter.Line(dash='dot')), col=3, row=1)

    for n, signal in enumerate(signals):
        fig.add_trace(go.Scatter(x=x, y=signal + bias, opacity=0.3, name=f'signal {n + 1}'))

    fig.update_xaxes(range=[0, 32], col=3)
    fig.update_yaxes(range=[y.min() * 1.2, y.max() * 1.2], col=1)
    fig.update_xaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=3)
    fig.update_xaxes(title_text="Frequency", row=1, col=3)
    fig.update_yaxes(title_text="Signal Value", row=1, col=1)
    fig.update_xaxes(title_text="time", row=1, col=1)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
