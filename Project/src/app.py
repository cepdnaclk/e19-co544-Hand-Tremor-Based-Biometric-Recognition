from flask import Flask, render_template, jsonify
import numpy as np
from src.components.data_ingestion import remove_outliers_iqr
from src.components.data_ingestion import calculate_metrics
from src.pipeline.predict_pipeline import predict
import pandas as pd
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    # Generate random data
    x = np.linspace(0, 100, 100)
    y1 = np.random.rand(100).tolist()
    y1 = y1*0.2
    y2 = np.random.rand(100).tolist()
    y2=y2*0.4
    y3 = np.random.rand(100).tolist()
    y3=y3*0.4
    y4 = np.random.rand(100).tolist()
    y4=y4*0.6

    metrics = []
    for col in [y1, y2, y3, y4]:
        window = col
        mean, std_dev, energy, entropy, num_peaks = calculate_metrics(window)
        metrics.append(mean)
        metrics.append(std_dev)
        metrics.append(energy)
        metrics.append(entropy)
        metrics.append(num_peaks)
    metrics = np.array(metrics)
    input = pd.DataFrame(metrics)
    target = predict(input)
    print(target)

    return jsonify({'x': x.tolist(), 'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4, 'target': target.tolist()[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
