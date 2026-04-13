"""
BCI Prototype - Core System
============================
Simulates EEG acquisition, runs preprocessing pipeline,
trains EEGNet-inspired CNN, then does real-time inference
and maps predictions to device commands.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt, welch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import time, warnings, json
from collections import deque
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. SYNTHETIC EEG DATA GENERATOR
# ─────────────────────────────────────────────
class EEGSimulator:
    """
    Simulates a 14-channel EEG device at 256 Hz.
    Generates realistic oscillatory + artifact signals
    for 5 cognitive intent classes.
    """
    CLASSES = {
        0: "open_app",
        1: "scroll_down",
        2: "scroll_up",
        3: "click",
        4: "idle"
    }
    # Each class has a dominant frequency signature
    CLASS_FREQ = {
        0: (12, 15),   # SMR band – focused intent
        1: (8, 10),    # Alpha – rhythmic downward imagery
        2: (10, 12),   # Alpha-high – upward imagery
        3: (15, 25),   # Beta – sharp motor click
        4: (2, 6),     # Theta/Delta – idle/resting
    }
    N_CHANNELS = 14
    SFREQ = 256        # Hz
    EPOCH_LEN = 2.0    # seconds

    def generate_epoch(self, class_id, add_artifacts=True):
        t = np.linspace(0, self.EPOCH_LEN, int(self.SFREQ * self.EPOCH_LEN))
        lo, hi = self.CLASS_FREQ[class_id]
        freq = np.random.uniform(lo, hi)
        data = np.zeros((self.N_CHANNELS, len(t)))
        for ch in range(self.N_CHANNELS):
            phase = np.random.uniform(0, 2*np.pi)
            amp   = np.random.uniform(0.8, 1.2)
            # dominant component
            data[ch] = amp * np.sin(2*np.pi*freq*t + phase)
            # background broadband noise
            data[ch] += 0.25 * np.random.randn(len(t))
            # weak harmonics
            data[ch] += 0.15 * np.sin(2*np.pi*(freq*2)*t + phase)
        if add_artifacts:
            # blink spike on frontal channels (0,1)
            if np.random.rand() < 0.2:
                blink_idx = np.random.randint(10, len(t)-10)
                for fch in [0, 1]:
                    data[fch, blink_idx:blink_idx+8] += 3.5
        return data.astype(np.float32)

    def generate_dataset(self, n_per_class=200):
        X, y = [], []
        for cls in range(len(self.CLASSES)):
            for _ in range(n_per_class):
                X.append(self.generate_epoch(cls))
                y.append(cls)
        X = np.array(X)   # (N, C, T)
        y = np.array(y)
        idx = np.random.permutation(len(y))
        return X[idx], y[idx]


# ─────────────────────────────────────────────
# 2. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
class Preprocessor:
    """
    Bandpass filter → artifact rejection → normalization
    """
    def __init__(self, sfreq=256, lo=1.0, hi=40.0):
        self.sfreq = sfreq
        self.lo, self.hi = lo, hi
        b, a = butter(4, [lo/(sfreq/2), hi/(sfreq/2)], btype='band')
        self.b, self.a = b, a
        self.scaler = StandardScaler()

    def bandpass(self, epoch):
        return filtfilt(self.b, self.a, epoch, axis=-1)

    def reject_artifacts(self, epoch, threshold=4.0):
        """Simple peak-to-peak amplitude-based artifact rejection"""
        channel_std = epoch.std(axis=-1, keepdims=True)
        channel_std = np.where(channel_std < 1e-10, 1e-10, channel_std)
        z = np.abs(epoch) / channel_std
        epoch = np.where(z > threshold, 0.0, epoch)
        return epoch

    def normalize(self, epoch):
        shape = epoch.shape
        flat = epoch.reshape(-1, shape[-1])
        # z-score per channel
        mu   = flat.mean(axis=-1, keepdims=True)
        sig  = flat.std(axis=-1, keepdims=True) + 1e-8
        return ((flat - mu) / sig).reshape(shape)

    def process(self, epoch):
        epoch = self.bandpass(epoch)
        epoch = self.reject_artifacts(epoch)
        epoch = self.normalize(epoch)
        return epoch

    def process_batch(self, X):
        return np.array([self.process(ep) for ep in X])


# ─────────────────────────────────────────────
# 3. FEATURE EXTRACTION
# ─────────────────────────────────────────────
class FeatureExtractor:
    """
    PSD band power + simple CSP-like variance features
    """
    BANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 40),
    }
    def __init__(self, sfreq=256):
        self.sfreq = sfreq

    def band_power(self, epoch):
        freqs, psd = welch(epoch, fs=self.sfreq, nperseg=128, axis=-1)
        features = []
        for lo, hi in self.BANDS.values():
            mask = (freqs >= lo) & (freqs < hi)
            features.append(np.log1p(psd[:, mask].mean(axis=-1)))
        return np.concatenate(features)   # shape: (C*5,)

    def variance_features(self, epoch):
        return np.log1p(epoch.var(axis=-1))  # per-channel variance

    def extract(self, epoch):
        bp = self.band_power(epoch)
        vf = self.variance_features(epoch)
        return np.concatenate([bp, vf])

    def extract_batch(self, X):
        return np.array([self.extract(ep) for ep in X])


# ─────────────────────────────────────────────
# 4. EEGNET-INSPIRED CNN MODEL
# ─────────────────────────────────────────────
class EEGNet(nn.Module):
    """
    Compact CNN architecture based on EEGNet (Lawhern et al., 2018)
    adapted for 14-channel, 512-sample epochs and 5 classes.
    """
    def __init__(self, n_classes=5, n_channels=14, n_times=512,
                 F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        # Block 1: temporal convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        # Block 2: depthwise spatial convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # Block 3: separable convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        # Calculate classifier input size
        dummy = torch.zeros(1, 1, n_channels, n_times)
        x = self.block1(dummy)
        x = self.block2(x)
        x = self.block3(x)
        flat_size = x.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, n_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
# 5. TRAINER
# ─────────────────────────────────────────────
class BCITrainer:
    def __init__(self, model, device='cpu'):
        self.model  = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            out  = self.model(X_batch)
            loss = self.criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)
        self.scheduler.step()
        return total_loss/len(loader), correct/total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        for X_batch, y_batch in loader:
            out = self.model(X_batch.to(self.device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y_batch.numpy())
        return np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────
# 6. REAL-TIME INFERENCE ENGINE + COMMAND ROUTER
# ─────────────────────────────────────────────
class RealTimeInferenceEngine:
    """
    Processes rolling 250ms windows, applies confidence threshold,
    debouncing, and RL-style adaptive weighting.
    """
    ACTIONS = {
        0: "open_app",
        1: "scroll_down",
        2: "scroll_up",
        3: "click",
        4: "idle"
    }
    COMMANDS = {
        "open_app":    "pyautogui.hotkey('ctrl','alt','t')  # open terminal",
        "scroll_down": "pyautogui.scroll(-3)               # scroll down",
        "scroll_up":   "pyautogui.scroll(3)                # scroll up",
        "click":       "pyautogui.click()                  # left click",
        "idle":        "# no action — idle state"
    }
    def __init__(self, model, preprocessor, confidence_threshold=0.80,
                 debounce_ms=500, device='cpu'):
        self.model      = model.eval()
        self.prep       = preprocessor
        self.threshold  = confidence_threshold
        self.debounce   = debounce_ms / 1000.0
        self.device     = device
        self.last_action_time = 0
        self.action_log = []
        # RL: per-class reward weights (updated by feedback)
        self.reward_weights = np.ones(5)

    @torch.no_grad()
    def predict(self, epoch_raw):
        """Full pipeline: raw epoch → command"""
        epoch = self.prep.process(epoch_raw)
        tensor = torch.tensor(epoch[np.newaxis, np.newaxis]).float().to(self.device)
        logits = self.model(tensor)[0]
        probs  = torch.softmax(logits * torch.tensor(self.reward_weights, dtype=torch.float32), dim=0)
        probs  = probs.cpu().numpy()
        pred   = int(probs.argmax())
        conf   = float(probs[pred])
        return pred, conf, probs

    def dispatch(self, epoch_raw, execute=False):
        pred, conf, probs = self.predict(epoch_raw)
        now    = time.time()
        action = self.ACTIONS[pred]
        result = {
            "timestamp":   now,
            "action":      action,
            "confidence":  round(conf, 4),
            "probs":       {self.ACTIONS[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "dispatched":  False,
            "reason":      ""
        }
        if conf < self.threshold:
            result["reason"] = f"confidence {conf:.2f} < threshold {self.threshold}"
        elif action == "idle":
            result["reason"] = "idle state — no action"
        elif (now - self.last_action_time) < self.debounce:
            result["reason"] = f"debounce ({(now-self.last_action_time)*1000:.0f}ms < {self.debounce*1000:.0f}ms)"
        else:
            result["dispatched"] = True
            result["command"]    = self.COMMANDS[action]
            self.last_action_time = now
            if execute:
                self._execute_command(action)
        self.action_log.append(result)
        return result

    def _execute_command(self, action):
        """Real execution — only called when execute=True"""
        try:
            import pyautogui
            if action == "scroll_down": pyautogui.scroll(-3)
            elif action == "scroll_up": pyautogui.scroll(3)
            elif action == "click":     pyautogui.click()
            elif action == "open_app":  pyautogui.hotkey('ctrl', 'alt', 't')
        except Exception as e:
            pass

    def rl_feedback(self, action_idx, reward):
        """
        Reinforcement: positive reward (+1) for correct, negative (-1) for wrong.
        Updates reward weights with exponential moving average.
        """
        alpha = 0.1
        self.reward_weights[action_idx] = (
            (1 - alpha) * self.reward_weights[action_idx] + alpha * (1 + reward)
        )
        self.reward_weights = np.clip(self.reward_weights, 0.5, 2.0)


# ─────────────────────────────────────────────
# 7. FULL PIPELINE RUN
# ─────────────────────────────────────────────
def run_full_pipeline():
    print("=" * 60)
    print("  BCI PROTOTYPE — FULL PIPELINE")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/6] Generating synthetic EEG dataset...")
    sim   = EEGSimulator()
    X, y  = sim.generate_dataset(n_per_class=160)
    print(f"      Dataset: {X.shape}  |  Classes: {np.unique(y)}")

    # Step 2: Preprocess
    print("\n[2/6] Preprocessing (bandpass + artifact rejection + norm)...")
    prep  = Preprocessor(sfreq=sim.SFREQ)
    X_p   = prep.process_batch(X)
    print(f"      Preprocessed: {X_p.shape}")

    # Step 3: Train/test split
    split = int(0.8 * len(y))
    X_tr, y_tr = X_p[:split], y[:split]
    X_te, y_te = X_p[split:], y[split:]

    # Step 4: Build model and train
    print("\n[3/6] Building EEGNet model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = EEGNet(n_classes=5, n_channels=sim.N_CHANNELS, n_times=X_p.shape[-1])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {total_params:,}  |  Device: {device}")

    # DataLoaders
    Xtr_t = torch.tensor(X_tr[:, np.newaxis]).float()
    Xte_t = torch.tensor(X_te[:, np.newaxis]).float()
    ytr_t = torch.tensor(y_tr).long()
    yte_t = torch.tensor(y_te).long()
    tr_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)
    te_loader = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=32)

    trainer = BCITrainer(model, device)

    print("\n[4/6] Training EEGNet (30 epochs)...")
    print(f"      {'Epoch':>5}  {'Train Loss':>11}  {'Train Acc':>10}  {'Val Acc':>8}")
    print("      " + "-"*45)
    for epoch in range(1, 31):
        loss, acc = trainer.train_epoch(tr_loader)
        if epoch % 5 == 0:
            preds, labels = trainer.evaluate(te_loader)
            val_acc = (preds == labels).mean()
            print(f"      {epoch:>5}  {loss:>11.4f}  {acc*100:>9.1f}%  {val_acc*100:>7.1f}%")

    # Step 5: Evaluation
    print("\n[5/6] Evaluation on test set...")
    preds, labels = trainer.evaluate(te_loader)
    acc = (preds == labels).mean()
    print(f"\n      Overall Accuracy: {acc*100:.1f}%\n")
    cr = classification_report(labels, preds,
                               target_names=list(EEGSimulator.CLASSES.values()),
                               digits=3)
    for line in cr.split('\n'):
        print("      " + line)

    # Step 6: Real-time simulation
    print("\n[6/6] Real-time inference simulation (20 epochs)...")
    engine = RealTimeInferenceEngine(model, prep, confidence_threshold=0.70, debounce_ms=400)
    dispatched_count, total_count = 0, 0
    print(f"\n      {'#':>3}  {'True Class':>12}  {'Predicted':>12}  {'Conf':>6}  {'Dispatched':>10}")
    print("      " + "-"*55)
    for i in range(20):
        true_cls = i % 5
        raw_epoch = sim.generate_epoch(true_cls)
        result    = engine.dispatch(raw_epoch, execute=False)
        dispatched = "✓" if result["dispatched"] else "✗"
        total_count += 1
        dispatched_count += int(result["dispatched"])
        print(f"      {i+1:>3}  {EEGSimulator.CLASSES[true_cls]:>12}  "
              f"{result['action']:>12}  {result['confidence']:>6.3f}  {dispatched:>10}")
        # Simulate RL feedback: reward if correct
        action_idx = list(EEGSimulator.CLASSES.values()).index(result['action'])
        reward = 1.0 if result['action'] == EEGSimulator.CLASSES[true_cls] else -0.5
        engine.rl_feedback(action_idx, reward)
        time.sleep(0.01)  # simulate 10ms between epochs

    print(f"\n      Dispatch rate: {dispatched_count}/{total_count} "
          f"({100*dispatched_count/total_count:.0f}%)")
    print(f"      RL reward weights: {engine.reward_weights.round(3)}")

    # Save results
    report = {
        "model_params": total_params,
        "test_accuracy": float(round(acc, 4)),
        "n_classes": 5,
        "classes": list(EEGSimulator.CLASSES.values()),
        "sfreq": sim.SFREQ,
        "n_channels": sim.N_CHANNELS,
        "epoch_len_s": sim.EPOCH_LEN,
        "dispatch_rate": dispatched_count / total_count,
        "rl_weights": engine.reward_weights.tolist(),
    }
    with open("/home/claude/bci_prototype/results.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Test Accuracy : {acc*100:.1f}%")
    print(f"  Model Params  : {total_params:,}")
    print(f"  Latency target: <50ms per inference")
    print("=" * 60)
    return report

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    run_full_pipeline()
