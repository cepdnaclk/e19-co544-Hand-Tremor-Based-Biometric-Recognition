from flask import Flask, render_template
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    # Generate random data
    x = np.linspace(0, 10, 100)
    y1 = np.random.rand(100)
    y2 = np.random.rand(100)
    y3 = np.random.rand(100)
    y4 = np.random.rand(100)

    # Create traces for each line graph
    trace1 = go.Scatter(x=x, y=y1, mode='lines', name='X Data')
    trace2 = go.Scatter(x=x, y=y2, mode='lines', name='Y Data')
    trace3 = go.Scatter(x=x, y=y3, mode='lines', name='Z Data')
    trace4 = go.Scatter(x=x, y=y4, mode='lines', name='Mixed Data')

    # Create a layout
    layout = go.Layout(title='Random Data Line Graphs')

    # Create a figure for each graph
    fig1 = go.Figure(data=[trace1], layout=layout)
    fig2 = go.Figure(data=[trace2], layout=layout)
    fig3 = go.Figure(data=[trace3], layout=layout)
    fig4 = go.Figure(data=[trace4], layout=layout)

    # Generate HTML divs
    graph1 = pyo.plot(fig1, output_type='div', include_plotlyjs=False)
    graph2 = pyo.plot(fig2, output_type='div', include_plotlyjs=False)
    graph3 = pyo.plot(fig3, output_type='div', include_plotlyjs=False)
    graph4 = pyo.plot(fig4, output_type='div', include_plotlyjs=False)

    return render_template('index.html', graph1=graph1, graph2=graph2, graph3=graph3, graph4=graph4)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
