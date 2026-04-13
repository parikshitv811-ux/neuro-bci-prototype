"""
BCI Level 2 — Production-Ready Pipeline
=========================================
Integrates:
  - EEGNet CNN intent classifier (from Level 1)
  - SSVEP/P300 paradigm detectors (high reliability)
  - Claude AI agent (intent expansion → structured plan)
  - Confirmation engine (blink / timeout / auto)
  - Autonomous execution engine (simulated or real)

Run modes:
  python bci_level2_pipeline.py                 # full simulated demo
  python bci_level2_pipeline.py --ssvep         # SSVEP paradigm demo
  python bci_level2_pipeline.py --p300          # P300 paradigm demo
  python bci_level2_pipeline.py --api-key KEY   # use real Claude API
  python bci_level2_pipeline.py --execute       # real OS execution (CAUTION)
"""

import sys, argparse, time, json
sys.path.insert(0, '/home/claude/bci_prototype')  # Level 1 core
sys.path.insert(0, '/home/claude/bci_level2')

import numpy as np
import torch

# Level 1 imports
from bci_core import EEGSimulator, Preprocessor, EEGNet, BCITrainer
from torch.utils.data import DataLoader, TensorDataset

# Level 2 imports
from agents.claude_agent import ClaudeIntentAgent, UserContext, ConfirmationEngine
from hardware.hardware_layer import SimulatedEEGSource, SSVEPDetector, SSVEPConfig, P300Detector, P300Config
from execution.execution_engine import ExecutionEngine


# ─────────────────────────────────────────────────────────────
# LEVEL 2 PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────
class BCILevel2Pipeline:
    """
    Full production-ready BCI pipeline.
    
    Flow:
        EEG source → paradigm detector or EEGNet CNN
             ↓
        Coarse intent label
             ↓
        Claude agent → structured action plan (with context)
             ↓
        Confirmation engine (if needed)
             ↓
        Execution engine → OS actions
             ↓
        RL feedback → update reward weights
    """

    def __init__(self,
                 use_real_api: bool = False,
                 api_key: str = None,
                 execute_real: bool = False,
                 paradigm: str = "eegnet",
                 confirm_mode: str = "auto"):

        print("\n" + "═" * 65)
        print("  BCI LEVEL 2 — INITIALIZING PRODUCTION PIPELINE")
        print("═" * 65)

        # EEG source (hardware abstraction)
        self.eeg_source = SimulatedEEGSource(sfreq=256, n_channels=14)
        print(f"  EEG source : Simulated (replace with BrainFlowSource for hardware)")

        # Paradigm
        self.paradigm = paradigm
        if paradigm == "ssvep":
            cfg = SSVEPConfig(
                frequencies=(8.0, 10.0, 12.0, 15.0),
                labels=("open_app", "scroll_down", "scroll_up", "click")
            )
            self.detector = SSVEPDetector(cfg, sfreq=256)
            print("  Paradigm   : SSVEP (4-class)")
        elif paradigm == "p300":
            self.detector = P300Detector(P300Config(), sfreq=256)
            print("  Paradigm   : P300 (binary target/non-target)")
        else:
            self.detector = None
            print("  Paradigm   : EEGNet CNN (5-class motor imagery)")

        # Train EEGNet for motor imagery fallback
        self.model = None
        self.prep = Preprocessor(sfreq=256)
        if paradigm == "eegnet":
            self._train_eegnet()

        # Claude agent
        self.context = UserContext(name="Demo User")
        self.agent = ClaudeIntentAgent(
            context=self.context,
            use_real_api=use_real_api,
            api_key=api_key
        )
        api_mode = "Claude API" if use_real_api else "mock (set --api-key for real Claude)"
        print(f"  AI agent   : {api_mode}")

        # Confirmation engine
        self.confirmer = ConfirmationEngine(mode=confirm_mode)
        print(f"  Confirm    : {confirm_mode} mode")

        # Execution engine
        self.executor = ExecutionEngine(simulate=not execute_real)
        exec_mode = "REAL OS execution" if execute_real else "simulation (safe)"
        print(f"  Execution  : {exec_mode}")

        # Metrics
        self.session_log = []
        print("═" * 65 + "\n")

    def _train_eegnet(self):
        """Quick 15-epoch training for the demo."""
        print("  Training EEGNet (15 epochs)...")
        sim = EEGSimulator()
        X, y = sim.generate_dataset(n_per_class=100)
        X_p = self.prep.process_batch(X)
        split = int(0.8 * len(y))
        X_tr, y_tr = X_p[:split], y[:split]
        X_te, y_te = X_p[split:], y[split:]
        Xtr_t = torch.tensor(X_tr[:, np.newaxis]).float()
        ytr_t = torch.tensor(y_tr).long()
        Xte_t = torch.tensor(X_te[:, np.newaxis]).float()
        yte_t = torch.tensor(y_te).long()
        tr_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)
        te_loader = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=32)
        self.model = EEGNet(n_classes=5, n_channels=14, n_times=X_p.shape[-1])
        trainer = BCITrainer(self.model, 'cpu')
        for ep in range(15):
            trainer.train_epoch(tr_loader)
        preds, labels = trainer.evaluate(te_loader)
        acc = (preds == labels).mean()
        print(f"  EEGNet ready — test accuracy: {acc*100:.1f}%\n")
        self.reward_weights = np.ones(5)

    def _get_intent_eegnet(self) -> tuple:
        """Motor imagery classification via EEGNet."""
        from bci_core import EEGSimulator
        sim = EEGSimulator()
        true_cls = np.random.randint(0, 5)
        epoch = sim.generate_epoch(true_cls)
        epoch_p = self.prep.process(epoch)
        tensor = torch.tensor(epoch_p[np.newaxis, np.newaxis]).float()
        with torch.no_grad():
            logits = self.model(tensor)[0]
            w = torch.tensor(self.reward_weights, dtype=torch.float32)
            probs = torch.softmax(logits * w, dim=0).numpy()
        pred = int(probs.argmax())
        conf = float(probs[pred])
        label = EEGSimulator.CLASSES[pred]
        true_label = EEGSimulator.CLASSES[true_cls]
        return label, conf, true_label

    def _get_intent_ssvep(self) -> tuple:
        """SSVEP paradigm classification."""
        target_freq = np.random.choice(self.detector.cfg.frequencies)
        epoch = self.detector.simulate_response(target_freq)
        result = self.detector.classify(epoch)
        return result["label"], result["confidence"], result["label"]

    def _get_intent_p300(self) -> tuple:
        """P300 paradigm classification."""
        epochs, labels = self.detector.simulate_epochs(n_target=2, n_nontarget=8)
        result = self.detector.run_speller_step(epochs, labels)
        label = "click" if result.get("selected") else "idle"
        conf = result.get("confidence", 0.5)
        return label, conf, label

    def run_session(self, n_trials: int = 10):
        """Run a full BCI session with N trials."""
        print(f"{'─'*65}")
        print(f"  SESSION START — {n_trials} trials, paradigm: {self.paradigm.upper()}")
        print(f"{'─'*65}\n")

        for trial_n in range(1, n_trials + 1):
            print(f"  ── Trial {trial_n}/{n_trials} ──")

            # Step 1: Acquire and classify EEG intent
            t0 = time.perf_counter()
            if self.paradigm == "ssvep":
                intent, conf, true_label = self._get_intent_ssvep()
            elif self.paradigm == "p300":
                intent, conf, true_label = self._get_intent_p300()
            else:
                intent, conf, true_label = self._get_intent_eegnet()
            latency_ms = (time.perf_counter() - t0) * 1000

            print(f"  EEG intent : {intent}  (conf={conf:.2f}, latency={latency_ms:.1f}ms)")

            # Skip low-confidence or idle
            if conf < 0.60 or intent == "idle":
                reason = "low confidence" if conf < 0.60 else "idle"
                print(f"  Skipped    : {reason}\n")
                continue

            # Step 2: Claude agent — expand intent → structured plan
            print(f"  Claude     : expanding '{intent}' with user context...")
            plan = self.agent.resolve(intent)
            print(f"  Plan       : {plan['interpreted_action']}")
            print(f"  Steps      : {len(plan['steps'])} step(s)")

            # Step 3: Confirmation (for irreversible actions)
            if plan.get("confirmation_required"):
                confirmed = self.confirmer.request_confirmation(
                    plan.get("confirmation_message", "Proceed?")
                )
                if not confirmed:
                    print("  Cancelled  : user aborted\n")
                    continue

            # Step 4: Execute
            results = self.executor.run_plan(plan)
            successes = sum(1 for r in results if r.success)
            print(f"  Executed   : {successes}/{len(results)} steps succeeded")

            # Step 5: Log and RL feedback
            self.context.add_action(intent)
            if self.paradigm == "eegnet" and intent in ["open_app","scroll_down","scroll_up","click","idle"]:
                from bci_core import EEGSimulator
                idx = list(EEGSimulator.CLASSES.values()).index(intent) if intent in EEGSimulator.CLASSES.values() else 4
                reward = 1.0 if successes == len(results) else -0.5
                alpha = 0.1
                self.reward_weights[idx] = (1 - alpha) * self.reward_weights[idx] + alpha * (1 + reward)
                self.reward_weights = np.clip(self.reward_weights, 0.5, 2.0)

            self.session_log.append({
                "trial": trial_n,
                "intent": intent,
                "confidence": conf,
                "latency_ms": round(latency_ms, 2),
                "plan_steps": len(plan["steps"]),
                "steps_ok": successes,
                "action": plan["interpreted_action"]
            })
            print()

        self._print_summary()
        return self.session_log

    def _print_summary(self):
        print(f"\n{'═'*65}")
        print("  SESSION SUMMARY")
        print(f"{'═'*65}")
        total = len(self.session_log)
        if total == 0:
            print("  No trials completed.")
            return
        avg_conf = np.mean([r["confidence"] for r in self.session_log])
        avg_lat  = np.mean([r["latency_ms"] for r in self.session_log])
        full_ok  = sum(1 for r in self.session_log if r["steps_ok"] == r["plan_steps"])
        print(f"  Trials completed : {total}")
        print(f"  Avg confidence   : {avg_conf:.2f}")
        print(f"  Avg EEG latency  : {avg_lat:.1f} ms")
        print(f"  Full success rate: {full_ok}/{total} ({100*full_ok/total:.0f}%)")
        print(f"\n  Actions executed:")
        for r in self.session_log:
            status = "✓" if r["steps_ok"] == r["plan_steps"] else "✗"
            print(f"    {status} Trial {r['trial']:>2}: [{r['intent']:>11}] {r['action'][:55]}")
        print(f"{'═'*65}\n")

        # Save log
        with open("/home/claude/bci_level2/session_log.json", "w") as f:
            json.dump(self.session_log, f, indent=2)
        print("  Log saved → bci_level2/session_log.json\n")


# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCI Level 2 Pipeline")
    parser.add_argument("--ssvep",    action="store_true", help="Use SSVEP paradigm")
    parser.add_argument("--p300",     action="store_true", help="Use P300 paradigm")
    parser.add_argument("--execute",  action="store_true", help="Real OS execution (careful!)")
    parser.add_argument("--api-key",  default=None,        help="Anthropic API key")
    parser.add_argument("--trials",   type=int, default=8, help="Number of trials")
    parser.add_argument("--confirm",  default="auto",      choices=["auto","deny","timeout","blink"])
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    paradigm = "ssvep" if args.ssvep else ("p300" if args.p300 else "eegnet")
    use_api  = args.api_key is not None
    pipeline = BCILevel2Pipeline(
        use_real_api=use_api,
        api_key=args.api_key,
        execute_real=args.execute,
        paradigm=paradigm,
        confirm_mode=args.confirm
    )
    pipeline.run_session(n_trials=args.trials)
