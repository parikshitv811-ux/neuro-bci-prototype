"""
TSTA Research Platform — Flask Web Server
==========================================
Serves the interactive 4-tab research dashboard with:
  - Pipeline runner (live terminal output)
  - Real-time EEG simulation via Server-Sent Events
  - Research results API
  - Model Explorer API
"""

import subprocess
import threading
import json
import os
import glob
import time
import queue
import numpy as np
from flask import Flask, render_template, jsonify, send_file, Response, request

app = Flask(__name__)

# ── Shared state ─────────────────────────────────────────────────────────────
pipeline_output  = []
pipeline_running = False
pipeline_lock    = threading.Lock()

# Real-time SSE state
rt_queue      = queue.Queue(maxsize=200)
rt_running    = False
rt_lock       = threading.Lock()
_rt_thread    = None

ALLOWED_SCRIPTS = {
    "debug":    ("python3", "-m", "tsta_project.scripts.debug"),
    "synthetic":("python3", "-m", "tsta_project.scripts.run_synthetic"),
    "real":     ("python3", "-m", "tsta_project.scripts.run_real"),
    "full":     ("python3", "-m", "tsta_project.scripts.run_full_pipeline"),
    "advanced": ("python3", "-m", "tsta_project.scripts.run_advanced"),
    "research": ("python3", "-m", "tsta_project.scripts.run_research"),
    "realtime": ("python3", "-m", "tsta_project.scripts.run_realtime"),
    "demo":     ("python3", "-m", "tsta_project.scripts.demo_mode"),
}


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline_bg(cmd: tuple):
    global pipeline_output, pipeline_running
    with pipeline_lock:
        pipeline_output = []
        pipeline_running = True
    try:
        proc = subprocess.Popen(
            list(cmd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
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


@app.route("/api/run/<script>", methods=["POST"])
def run_script(script):
    if script not in ALLOWED_SCRIPTS:
        return jsonify({"error": "Unknown script"}), 400
    global pipeline_running
    with pipeline_lock:
        if pipeline_running:
            return jsonify({"error": "Pipeline already running"}), 409
    t = threading.Thread(target=_run_pipeline_bg,
                         args=(ALLOWED_SCRIPTS[script],), daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/output")
def get_output():
    with pipeline_lock:
        return jsonify({"lines": list(pipeline_output),
                        "running": pipeline_running})


# ── Real-time EEG SSE ─────────────────────────────────────────────────────────

def _rt_worker(duration_s: float, intent_seq: list):
    global rt_running
    try:
        import torch
        from tsta_project.config                     import TSTAConfig
        from tsta_project.model                      import TSTA
        from tsta_project.data.synthetic.generator   import SyntheticEEGGenerator
        from tsta_project.data.preprocess            import Preprocessor
        from tsta_project.training.trainer           import TSTATrainer
        from tsta_project.realtime.stream_simulator  import EEGStreamSimulator
        from tsta_project.realtime.realtime_inference import RealTimeInference

        # Quick data + model setup
        gen  = SyntheticEEGGenerator(n_subjects=2, n_per_class=20, seed=42)
        ds   = gen.get_dataset()
        prep = Preprocessor(sfreq=ds.sfreq)
        ds   = prep.process_dataset(ds)

        cfg = TSTAConfig()
        cfg.update_from_dataset(ds.n_channels, ds.sfreq, ds.n_samples)

        model_path = os.path.join("tsta_project/outputs/models", "tsta_subj01.pt")
        model = TSTA(cfg)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            rt_queue.put(json.dumps({"status": "training",
                                      "msg": "No saved model — training fast model..."}))
            trainer = TSTATrainer(cfg, "cpu")
            mask    = ds.subjects == 1
            model, _ = trainer.train(ds.X[mask], ds.y[mask], epochs=12, tag="[RT]")
        model.eval()

        sim = EEGStreamSimulator(cfg, chunk_s=0.5, n_subjects=2, seed=42)
        rt  = RealTimeInference(model, cfg, "cpu", buffer_size=16)

        for chunk, label, t_ms in sim.stream(duration_s=duration_s,
                                              intent_seq=intent_seq):
            if not rt_running:
                break
            result = rt.infer(chunk, true_label=label)
            # Add a compact EEG signal for display (8 ch, downsampled)
            ch_sel = [0, 8, 16, 20, 24, 28, 32, 48]
            eeg_disp = chunk[ch_sel, ::4].tolist()   # 8 ch, T/4 samples

            payload = {
                "step":       result["step"],
                "t_ms":       round(t_ms, 1),
                "pred_class": result["pred_class"],
                "pred_intent":result["pred_intent"],
                "true_label": label,
                "true_intent":_intent_name(label),
                "confidence": result["confidence"],
                "correct":    result["correct"],
                "all_sims":   result["all_sims"],
                "traj_point": result["traj_point"],
                "trajectory": result["trajectory"],
                "eeg_disp":   eeg_disp,
                "smooth_dir": result["smooth_dir"],
            }
            rt_queue.put(json.dumps(payload))
            time.sleep(0.12)   # ~8 fps

        rt_queue.put(json.dumps({"status": "done"}))
    except Exception as e:
        rt_queue.put(json.dumps({"status": "error", "msg": str(e)}))
    finally:
        with rt_lock:
            rt_running = False


def _intent_name(cls: int) -> str:
    names = ["communication", "navigation", "action", "selection", "idle"]
    return names[cls] if 0 <= cls < len(names) else "?"


@app.route("/api/realtime/start", methods=["POST"])
def start_realtime():
    global rt_running, _rt_thread
    with rt_lock:
        if rt_running:
            return jsonify({"error": "Already running"}), 409
        rt_running = True
    # Drain old queue
    while not rt_queue.empty():
        try: rt_queue.get_nowait()
        except: break
    data       = request.get_json(silent=True) or {}
    duration_s = float(data.get("duration_s", 15.0))
    intent_seq = data.get("intent_seq", [0, 1, 2, 3, 4, 0, 1, 2])
    _rt_thread = threading.Thread(target=_rt_worker,
                                   args=(duration_s, intent_seq), daemon=True)
    _rt_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/realtime/stop", methods=["POST"])
def stop_realtime():
    global rt_running
    with rt_lock:
        rt_running = False
    return jsonify({"status": "stopped"})


@app.route("/api/realtime/stream")
def realtime_stream():
    def _gen():
        while True:
            try:
                msg = rt_queue.get(timeout=2.0)
                yield f"data: {msg}\n\n"
                if json.loads(msg).get("status") in ("done", "error"):
                    break
            except queue.Empty:
                yield "data: {\"heartbeat\": true}\n\n"
    return Response(_gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── Results & figures API ─────────────────────────────────────────────────────

@app.route("/api/results")
def get_results():
    results = {}
    for key, path in [
        ("synthetic",       "tsta_project/outputs/logs/synthetic_results.json"),
        ("real",            "tsta_project/outputs/logs/real_results.json"),
        ("advanced",        "tsta_project/outputs/logs/advanced_results.json"),
        ("research_report", "tsta_project/outputs/logs/research_report.json"),
        ("demo",            "tsta_project/outputs/logs/demo_results.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                results[key] = json.load(f)
    return jsonify(results)


@app.route("/api/figures")
def list_figures():
    figs = sorted(glob.glob("tsta_project/outputs/figures/*.png"))
    return jsonify([os.path.basename(f) for f in figs])


@app.route("/figures/<name>")
def serve_figure(name):
    path = os.path.join("tsta_project/outputs/figures", name)
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return "Not found", 404


# ── Model explorer API ────────────────────────────────────────────────────────

@app.route("/api/explorer/subjects")
def explorer_subjects():
    models = glob.glob("tsta_project/outputs/models/tsta_subj*.pt")
    ids    = sorted([int(os.path.basename(m)[9:11]) for m in models])
    return jsonify({"subjects": ids})


@app.route("/api/explorer/trajectory", methods=["POST"])
def explorer_trajectory():
    """Encode a single subject+class sample and return trajectory data."""
    try:
        import torch
        import torch.nn.functional as F
        from tsta_project.config                   import TSTAConfig
        from tsta_project.model                    import TSTA
        from tsta_project.data.synthetic.generator import SyntheticEEGGenerator
        from tsta_project.data.preprocess          import Preprocessor

        data  = request.get_json(silent=True) or {}
        subj  = int(data.get("subject", 1))
        cls   = int(data.get("cls", 0))

        gen   = SyntheticEEGGenerator(n_subjects=subj, n_per_class=8, seed=42)
        ds    = gen.get_dataset()
        prep  = Preprocessor(sfreq=ds.sfreq)
        ds    = prep.process_dataset(ds)

        cfg   = TSTAConfig()
        cfg.update_from_dataset(ds.n_channels, ds.sfreq, ds.n_samples)

        model_path = f"tsta_project/outputs/models/tsta_subj{subj:02d}.pt"
        model = TSTA(cfg)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            return jsonify({"error": f"No saved model for subject {subj}. Run synthetic pipeline first."}), 404

        model.eval()
        mask = (ds.subjects == subj) & (ds.y == cls)
        if mask.sum() == 0:
            return jsonify({"error": "No data for this subject/class"}), 400

        x_sample = torch.tensor(ds.X[mask][:8], dtype=torch.float32)
        y_sample = torch.tensor(ds.y[mask][:8], dtype=torch.long)

        with torch.no_grad():
            patch_tok = model.patcher(x_sample)
            trans_tok = model.transformer(patch_tok)   # (B, P, D)
            dirs, _,_ = model(x_sample, y_sample)
            ids = torch.arange(cfg.N_CLASSES)
            txt = model.encode_text(ids)

        # Build trajectory: patch tokens projected to 2D (first 2 dims)
        traj = trans_tok[0].numpy()   # (P, D) — first sample
        traj_2d = traj[:, :2].tolist()

        dir_np  = F.normalize(dirs, dim=-1).numpy()
        sims    = (dir_np @ txt.numpy().T).tolist()

        return jsonify({
            "subject":   subj,
            "cls":       cls,
            "intent":    _intent_name(cls),
            "trajectory_2d": traj_2d,       # (P, 2)
            "direction":     dir_np[0].tolist(),
            "all_sims":      sims[0],
            "n_patches": int(traj.shape[0]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
