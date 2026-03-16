import os
import json
import subprocess
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

training_process = None

@app.route('/api/status', methods=['GET'])
def get_status():
    global training_process
    is_running = training_process is not None and training_process.poll() is None
    return jsonify({"is_running": is_running})

@app.route('/api/start', methods=['POST'])
def start_training():
    global training_process
    if training_process is not None and training_process.poll() is None:
        return jsonify({"error": "Training already running"}), 400
    
    # Initialize empty log files before starting
    with open("training_metrics.json", "w") as f:
        json.dump([], f)
    log_file = open('training_terminal.log', 'w')
    
    python_exec = r"C:\unsloth_env\Scripts\python.exe"
    if not os.path.exists(python_exec):
        python_exec = "python" # fallback
        
    training_process = subprocess.Popen(
        [python_exec, "train.py"],
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    return jsonify({"message": "Training started"})

@app.route('/api/stop', methods=['POST'])
def stop_training():
    global training_process
    if training_process is not None and training_process.poll() is None:
        training_process.terminate()
        return jsonify({"message": "Training stopped"})
    return jsonify({"error": "Training not running"}), 400

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    try:
        if os.path.exists("training_metrics.json"):
            with open("training_metrics.json", "r") as f:
                data = json.load(f)
            return jsonify(data)
    except Exception as e:
        pass
    return jsonify([])

@app.route('/api/terminal', methods=['GET'])
def get_terminal():
    try:
        if os.path.exists("training_terminal.log"):
            with open("training_terminal.log", "r", errors="ignore") as f:
                lines = f.readlines()[-100:]
                return jsonify({"log": "".join(lines)})
    except Exception as e:
        pass
    return jsonify({"log": ""})

if __name__ == '__main__':
    with open("training_metrics.json", "w") as f:
        json.dump([], f)
    with open("training_terminal.log", "w") as f:
        f.write("")
    app.run(port=5000, debug=True)
