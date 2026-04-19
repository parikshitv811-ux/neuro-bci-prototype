"""
BCI Prototype - Web Interface
Serves the BCI project dashboard and allows running the pipeline.
"""

import subprocess
import threading
import json
import os
import time
from flask import Flask, render_template, jsonify, send_file, Response

app = Flask(__name__)

pipeline_output = []
pipeline_running = False
pipeline_lock = threading.Lock()


def run_pipeline_bg(script):
    global pipeline_output, pipeline_running
    with pipeline_lock:
        pipeline_output = []
        pipeline_running = True
    try:
        proc = subprocess.Popen(
            ["python3", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            with pipeline_lock:
                pipeline_output.append(line.rstrip())
        proc.wait()
    except Exception as e:
        with pipeline_lock:
            pipeline_output.append(f"Error: {e}")
    finally:
        with pipeline_lock:
            pipeline_running = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run/<script>", methods=["POST"])
def run_script(script):
    allowed = {"bci_core": "bci_core.py", "benchmark": "benchmark.py", "run_all": "run_all.py"}
    if script not in allowed:
        return jsonify({"error": "Unknown script"}), 400
    global pipeline_running
    with pipeline_lock:
        if pipeline_running:
            return jsonify({"error": "Pipeline already running"}), 409
    t = threading.Thread(target=run_pipeline_bg, args=(allowed[script],), daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/output")
def get_output():
    with pipeline_lock:
        return jsonify({
            "lines": list(pipeline_output),
            "running": pipeline_running
        })


@app.route("/api/results")
def get_results():
    results_file = "tsta_results.json"
    if os.path.exists(results_file):
        with open(results_file) as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route("/dashboard.png")
def dashboard():
    if os.path.exists("bci_dashboard.png"):
        return send_file("bci_dashboard.png", mimetype="image/png")
    return "No dashboard image yet", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
