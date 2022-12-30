import numpy as np
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots


number_kwargs = dict(type="number", value=0, style={'marginRight': '10px'})
app = Dash(__name__)
app.layout = html.Div([
    html.H2('Fourier Transformation'),
    html.P("samples"),
    dcc.Input(id='num', type='number', min=2, max=1024, step=2, value=64),
    html.P("Amplitudes (sep='  ')"),
    dcc.Input(type='text', value='1 1 1', id='amplitudes', debounce=True),
    dcc.Graph(id="graph"),
    dcc.Slider(id='max_hz', min=0, max=32, value=0, step=1),
])


@app.callback(
    Output("graph", "figure"),
    Input("max_hz", "value"),
    Input("num", "value"),
    Input("amplitudes", "value")
)
def plot_sin(i, n_samples, amps):
    amps = list(map(float, amps.strip().split(' ')))
    x = np.pi * np.linspace(0, 2, n_samples)
    signals = [a * np.sin(x * (f + 1)) for f, a in enumerate(amps[1:])]
    y = np.array(signals).sum(axis=0) + amps[0]
    i += 1
    fig = make_subplots(rows=1, cols=3, column_widths=[0.5, 0.25, 0.25])
    rft = np.fft.rfft(y)
    rft[i:] = 0
    smooth_y = np.fft.irfft(rft)
    fig.add_traces(data=[go.Scatter(x=x, y=y, mode='markers', name='data'),
                         go.Scatter(x=x, y=smooth_y, name='approximation', line=go.scatter.Line(width=3))
                         ])
    xrange = np.arange(len(x))
    fft = np.fft.fft(y)
    fig.add_trace(go.Scatter(x=xrange, y=rft.real, name='real'), col=2, row=1)
    fig.add_trace(go.Scatter(x=xrange, y=rft.imag, name='imag',
                             line=go.scatter.Line(dash='dot')), col=2, row=1)

    amplitudes = 2 * np.abs(rft) / len(x)
    amplitudes[0] /= 2
    frequencies = np.fft.fftfreq(len(x)) * len(x)
    fig.add_trace(go.Scatter(x=frequencies[:len(frequencies) // 2],
                             y=amplitudes[:len(fft) // 2], name='Amplitude'), col=3, row=1)

    for n, signal in enumerate(signals):
        fig.add_trace(go.Scatter(x=x, y=signal + amps[0], opacity=0.3, name=f'signal {n + 1}'))
    fig.update_yaxes(range=[y.min() * 1.2, y.max() * 1.2], col=1)
    fig.update_xaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=3)
    fig.update_xaxes(title_text="Frequency", row=1, col=3)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="time", row=1, col=1)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
