from flask import Flask, render_template, jsonify
import numpy as np
from src.components.data_ingestion import remove_outliers_iqr
from src.components.data_ingestion import calculate_metrics
from src.pipeline.predict_pipeline import predict
import pandas as pd
from collections import defaultdict
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

df = pd.read_csv('artifacts/dataset.csv')

@app.route('/data')
def data():
    target_idx = np.random.randint(1,5)
    person_df = df[df['ClassLabel'] == target_idx]
    for col in ['X', 'Y', 'Z', 'Mixed']:
        person_df = remove_outliers_iqr(person_df, col)

    d = {}

    x = np.linspace(0, 100, 100)

    
    i = np.random.randint(0, 350)
    metrics = defaultdict(list)
    for col in ['X', 'Y', 'Z', 'Mixed']:
        window = person_df[col].iloc[i:i + 100]
        d[col] = window
        mean, std_dev, energy, entropy, num_peaks = calculate_metrics(window)
        metrics[f'Mean_{col}'].append(mean)
        metrics[f'Std Dev_{col}'].append(std_dev)
        metrics[f'Energy_{col}'].append(energy)
        metrics[f'Entropy_{col}'].append(entropy)
        metrics[f'Peaks_{col}'].append(num_peaks)
    input = pd.DataFrame(metrics)
    target = predict(input)

    return jsonify({'x': x.tolist(), 'y1': d['X'].tolist(), 'y2': d['Y'].tolist(), 'y3': d['Z'].tolist(), 
                    'y4': d['Mixed'].tolist(), 'target': target[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
