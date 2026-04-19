"""
BCI Prototype - Web Interface
Serves the BCI + TSTA project dashboard.
"""

import subprocess
import threading
import json
import os
import glob
from flask import Flask, render_template, jsonify, send_file

app = Flask(__name__)

pipeline_output = []
pipeline_running = False
pipeline_lock = threading.Lock()


ALLOWED_SCRIPTS = {
    # Legacy scripts
    "bci_core":   ("python3", "bci_core.py"),
    "benchmark":  ("python3", "benchmark.py"),
    # New tsta_project scripts
    "debug":      ("python3", "-m", "tsta_project.scripts.debug"),
    "synthetic":  ("python3", "-m", "tsta_project.scripts.run_synthetic"),
    "real":       ("python3", "-m", "tsta_project.scripts.run_real"),
    "full":       ("python3", "-m", "tsta_project.scripts.run_full_pipeline"),
}


def run_pipeline_bg(cmd: tuple):
    global pipeline_output, pipeline_running
    with pipeline_lock:
        pipeline_output = []
        pipeline_running = True
    try:
        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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
    if script not in ALLOWED_SCRIPTS:
        return jsonify({"error": "Unknown script"}), 400
    global pipeline_running
    with pipeline_lock:
        if pipeline_running:
            return jsonify({"error": "Pipeline already running"}), 409
    cmd = ALLOWED_SCRIPTS[script]
    t = threading.Thread(target=run_pipeline_bg, args=(cmd,), daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/output")
def get_output():
    with pipeline_lock:
        return jsonify({"lines": list(pipeline_output), "running": pipeline_running})


@app.route("/api/results")
def get_results():
    results = {}
    # Load synthetic results
    syn_path = "tsta_project/outputs/logs/synthetic_results.json"
    if os.path.exists(syn_path):
        with open(syn_path) as f:
            results["synthetic"] = json.load(f)
    # Load real results
    real_path = "tsta_project/outputs/logs/real_results.json"
    if os.path.exists(real_path):
        with open(real_path) as f:
            results["real"] = json.load(f)
    # Legacy
    if os.path.exists("tsta_results.json"):
        with open("tsta_results.json") as f:
            results["legacy"] = json.load(f)
    return jsonify(results)


@app.route("/api/figures")
def list_figures():
    figs = glob.glob("tsta_project/outputs/figures/*.png")
    return jsonify([os.path.basename(f) for f in figs])


@app.route("/figures/<name>")
def serve_figure(name):
    path = os.path.join("tsta_project/outputs/figures", name)
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return "Not found", 404


@app.route("/dashboard.png")
def legacy_dashboard():
    if os.path.exists("bci_dashboard.png"):
        return send_file("bci_dashboard.png", mimetype="image/png")
    return "No dashboard image yet", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
