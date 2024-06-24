from flask import Flask, render_template, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    # Generate random data
    x = np.linspace(0, 10, 100)
    y1 = np.random.rand(100).tolist()
    y2 = np.random.rand(100).tolist()
    y3 = np.random.rand(100).tolist()
    y4 = np.random.rand(100).tolist()

    return jsonify({'x': x.tolist(), 'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
