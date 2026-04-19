"""
scripts/run_realtime.py
========================
Real-time EEG simulation + TSTA inference demo (CLI version).
Loads a saved model (or trains fast), then streams synthetic EEG.

Usage:
    python -m tsta_project.scripts.run_realtime
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import torch
import numpy as np

from tsta_project.config                     import TSTAConfig, MODELS_DIR
from tsta_project.utils                      import seed_everything, get_device, banner, section
from tsta_project.data.synthetic.generator   import SyntheticEEGGenerator
from tsta_project.data.preprocess            import Preprocessor
from tsta_project.model                      import TSTA
from tsta_project.training.trainer           import TSTATrainer
from tsta_project.realtime.stream_simulator  import EEGStreamSimulator
from tsta_project.realtime.realtime_inference import RealTimeInference
from tsta_project.data.synthetic.profiles    import CATEGORIES


INTENT_CYCLE = [0, 1, 2, 3, 4, 0, 1, 2]   # cycle through intents


def _load_or_train(cfg: TSTAConfig, ds, device: str):
    path = os.path.join(MODELS_DIR, "tsta_subj01.pt")
    model = TSTA(cfg).to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"  [Loaded] {path}")
    else:
        print("  Training fast model (15 epochs)...")
        trainer = TSTATrainer(cfg, device)
        mask    = ds.subjects == 1
        model, _ = trainer.train(ds.X[mask], ds.y[mask], epochs=15, tag="[RT]")
    return model


def main():
    seed_everything(42)
    device = get_device()
    banner("TSTA — REAL-TIME EEG SIMULATION")

    section("Data & Model")
    gen  = SyntheticEEGGenerator(n_subjects=2, n_per_class=20, seed=42)
    ds   = gen.get_dataset()
    prep = Preprocessor(sfreq=ds.sfreq)
    ds   = prep.process_dataset(ds)

    cfg = TSTAConfig()
    cfg.update_from_dataset(ds.n_channels, ds.sfreq, ds.n_samples)
    model = _load_or_train(cfg, ds, device)

    section("Stream Simulator + Real-Time Inference")
    sim = EEGStreamSimulator(cfg, chunk_s=0.5, n_subjects=2, seed=42)
    rt  = RealTimeInference(model, cfg, device, buffer_size=16, ema_alpha=0.35)

    intent_seq = INTENT_CYCLE
    correct, total = 0, 0

    COLS = 60
    print(f"\n  {'Step':>5}  {'Time':>7}  {'True':>14}  {'Pred':>14}  {'Conf':>6}  OK?")
    print(f"  {'─' * COLS}")

    for chunk, label, t_ms in sim.stream(duration_s=8.0, intent_seq=intent_seq):
        r     = rt.infer(chunk, true_label=label)
        ok    = "✓" if r["correct"] else "✗"
        if r["correct"] is not None:
            correct += int(r["correct"])
            total   += 1
        if r["step"] % 3 == 0:
            print(f"  {r['step']:>5}  {t_ms:>5.0f}ms  "
                  f"{CATEGORIES[label]:>14}  "
                  f"{r['pred_intent']:>14}  "
                  f"{r['confidence']:>6.3f}  {ok}")

    acc = correct / max(total, 1)
    print(f"\n  Real-time accuracy : {acc*100:.1f}%  ({correct}/{total})")
    print("  Simulation complete.")


if __name__ == "__main__":
    main()
