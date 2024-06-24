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
person1 = df[df['ClassLabel'] == 1]
person2 = df[df['ClassLabel'] == 2]
person3 = df[df['ClassLabel'] == 2]
person4 = df[df['ClassLabel'] == 4]
person5 = df[df['ClassLabel'] == 5]

personas = [person1, person2, person3, person4, person5]

# for index, person_df in enumerate(personas):
#     # Remove outliers for each column
#     for col in ['X', 'Y', 'Z', 'Mixed']:
#         df[df['ClassLabel']==index+1] = remove_outliers_iqr(person_df, col)

@app.route('/data')
def data():
    target_idx = np.random.randint(1,5)
    # selected_data = df.iloc[start_idx:start_idx + 100]
    # print(start_idx, selected_data['ClassLabel'].values)
    person_df = df[df['ClassLabel'] == target_idx]
    for col in ['X', 'Y', 'Z', 'Mixed']:
        person_df = remove_outliers_iqr(person_df, col)

    
    # Generate random data
    # d = {'X': np.random.rand(100)*0.6-0.1, 
    #      'Y': np.random.rand(100)*0.4-0.4, 
    #      'Z': np.random.rand(100)*0.4-0.002, 
    #      'Mixed': np.random.rand(100)}
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
    print(target)

    return jsonify({'x': x.tolist(), 'y1': d['X'].tolist(), 'y2': d['Y'].tolist(), 'y3': d['Z'].tolist(), 
                    'y4': d['Mixed'].tolist(), 'target': target[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
