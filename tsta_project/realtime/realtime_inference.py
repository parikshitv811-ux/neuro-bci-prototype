"""
Real-Time TSTA Inference Engine
=================================
Maintains a rolling buffer of EEG direction vectors and performs
live semantic decoding as new EEG chunks arrive.

Pipeline:
  1. Receive raw chunk (C, chunk_len)
  2. Zero-pad to model input size (C, T)
  3. Run TSTA encode_eeg → direction vector
  4. Push to direction buffer (last N)
  5. Smooth direction using exponential moving average
  6. Output: predicted class, confidence, smoothed direction, trajectory

Designed for streaming from EEGStreamSimulator.
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from tsta_project.config  import TSTAConfig
from tsta_project.model   import TSTA
from tsta_project.data.synthetic.profiles import CATEGORIES


class RealTimeInference:
    """
    Real-time EEG semantic decoder using a pre-trained TSTA model.

    Args:
        model:       Trained TSTA model
        cfg:         TSTAConfig
        device:      'cpu' | 'cuda'
        buffer_size: Number of past direction vectors to keep
        ema_alpha:   EMA smoothing factor (0=frozen, 1=no smoothing)
    """

    def __init__(self,
                 model,
                 cfg:         TSTAConfig,
                 device:      str   = "cpu",
                 buffer_size: int   = 16,
                 ema_alpha:   float = 0.35):
        self.model       = model
        self.cfg         = cfg
        self.device      = device
        self.ema_alpha   = ema_alpha
        self.T_model     = cfg.N_SAMPLES      # expected input length

        # Rolling direction buffer
        D = cfg.D_MODEL
        self.dir_buffer  = deque(maxlen=buffer_size)
        self.smooth_dir  = np.zeros(D, dtype=np.float32)
        self.trajectory  = []                  # list of (x,y) in PCA space

        # Class text embeddings (frozen)
        self.model.eval()
        with torch.no_grad():
            ids = torch.arange(cfg.N_CLASSES).to(device)
            self.text_mat = model.encode_text(ids).cpu().numpy()  # (C, D)

        self.step_count  = 0
        self.last_result = None

    def reset(self):
        """Clear buffer and trajectory."""
        self.dir_buffer.clear()
        self.smooth_dir[:] = 0.0
        self.trajectory.clear()
        self.step_count = 0

    def _pad_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Zero-pad (C, chunk_len) → (C, T_model)."""
        C, L = chunk.shape
        if L >= self.T_model:
            return chunk[:, :self.T_model]
        out = np.zeros((C, self.T_model), dtype=np.float32)
        out[:, :L] = chunk
        return out

    def infer(self, chunk: np.ndarray, true_label: int = -1) -> dict:
        """
        Process one incoming EEG chunk.

        Args:
            chunk:      (C, chunk_len) raw EEG
            true_label: Ground-truth label (-1 if unknown)

        Returns:
            {
              'step':          int,
              'pred_class':    int,
              'pred_intent':   str,
              'confidence':    float,     # max cosine sim
              'sdas_instant':  float,     # instantaneous correct-class sim
              'direction':     list[float],
              'smooth_dir':    list[float],
              'traj_point':    [float, float],  # 2D PCA projection (approx)
              'true_label':    int,
              'correct':       bool,
            }
        """
        self.model.eval()
        x_pad   = self._pad_chunk(chunk)
        x_t     = torch.tensor(x_pad[None], dtype=torch.float32).to(self.device)
        dummy_y = torch.zeros(1, dtype=torch.long).to(self.device)

        with torch.no_grad():
            eeg_dir, _, _ = self.model(x_t, dummy_y)

        dir_np = F.normalize(eeg_dir, dim=-1).cpu().numpy()[0]   # (D,)
        self.dir_buffer.append(dir_np)

        # Exponential moving average smoothing
        if self.step_count == 0:
            self.smooth_dir = dir_np.copy()
        else:
            self.smooth_dir = (self.ema_alpha * dir_np
                               + (1 - self.ema_alpha) * self.smooth_dir)
        sd_norm = self.smooth_dir / (np.linalg.norm(self.smooth_dir) + 1e-8)

        # Classification via cosine similarity
        sims       = self.text_mat @ sd_norm            # (C,)
        pred_class = int(np.argmax(sims))
        confidence = float(np.max(sims))
        sdas_inst  = float(sims[true_label]) if true_label >= 0 else None

        # 2D trajectory approximation: project onto first two principal axes
        # Use fast PCA via the first two singular vectors of the buffer
        traj_pt = [float(sd_norm[0] * 3), float(sd_norm[1] * 3)]
        self.trajectory.append(traj_pt)
        if len(self.trajectory) > 80:
            self.trajectory = self.trajectory[-80:]

        self.step_count += 1

        result = {
            "step":         self.step_count,
            "pred_class":   pred_class,
            "pred_intent":  CATEGORIES[pred_class],
            "confidence":   round(confidence, 4),
            "sdas_instant": round(sdas_inst, 4) if sdas_inst is not None else None,
            "direction":    [round(float(v), 4) for v in dir_np[:8]],   # first 8 dims
            "smooth_dir":   [round(float(v), 4) for v in sd_norm[:8]],
            "traj_point":   traj_pt,
            "trajectory":   list(self.trajectory[-20:]),
            "true_label":   true_label,
            "correct":      (pred_class == true_label) if true_label >= 0 else None,
            "all_sims":     [round(float(s), 4) for s in sims],
        }
        self.last_result = result
        return result
