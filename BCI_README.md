# BCI Prototype — Brain-Computer Interface System

## What This Is
A fully working Python prototype of an end-to-end BCI pipeline that:
- Simulates 14-channel EEG data at 256 Hz
- Runs a bandpass filter + ICA artifact rejection + normalization
- Trains an **EEGNet** CNN (6,453 parameters) to classify 5 brain intents
- Performs real-time inference with debouncing and confidence thresholding
- Uses a Reinforcement Learning feedback loop to adapt reward weights

## Results
| Metric             | Value       |
|--------------------|-------------|
| Test accuracy      | **95.6%**   |
| Model parameters   | 6,453       |
| Median latency     | ~1.6 ms     |
| P95 latency        | ~56 ms      |
| Classes            | 5           |

## Classified Actions
| Class ID | Intent       | Device Command                      |
|----------|--------------|-------------------------------------|
| 0        | open_app     | Ctrl+Alt+T (open terminal)          |
| 1        | scroll_down  | pyautogui.scroll(-3)                |
| 2        | scroll_up    | pyautogui.scroll(+3)                |
| 3        | click        | pyautogui.click()                   |
| 4        | idle         | no action                           |

## Architecture

```
[EEG Wearable 256Hz]
        ↓
[Bandpass Filter 1-40Hz]
[ICA Artifact Removal]
[Epoch Normalization]
        ↓
[Feature Extraction: PSD + CSP variance]
        ↓
[EEGNet CNN - 3 blocks]
  Block1: Temporal Conv (1×64)
  Block2: Depthwise Spatial Conv (14×1)
  Block3: Separable Conv + AvgPool
        ↓
[Softmax → 5 classes]
        ↓
[Confidence threshold (>70%)]
[Debounce (400ms)]
        ↓
[Command Router → OS API]
        ↓
[RL Feedback Loop (reward weights)]
```

## Files
- `bci_core.py`      — Full pipeline: simulator, preprocessor, EEGNet, trainer, inference engine
- `benchmark.py`     — Latency benchmarking (200 inference runs)
- `visualize.py`     — Dashboard plots (EEG signals, PSD, confusion matrix, training curve)
- `bci_dashboard.png`— Output visualization
- `results.json`     — Saved pipeline metrics

## How to Run

```bash
# Install dependencies
pip install numpy scipy mne torch scikit-learn matplotlib

# Run full prototype pipeline
python3 bci_core.py

# Run latency benchmark
python3 benchmark.py

# Generate visualizations
python3 visualize.py
```

## Connecting Real Hardware
To connect a real EEG device, replace `EEGSimulator.generate_epoch()` with:

```python
# OpenBCI
from brainflow.board_shim import BoardShim, BrainFlowInputParams
params = BrainFlowInputParams(); params.serial_port = '/dev/ttyUSB0'
board = BoardShim(BoardIds.CYTON_BOARD, params)
board.prepare_session(); board.start_stream()
data = board.get_current_board_data(512)  # 512 samples = 2s at 256Hz

# Muse (via muse-lsl)
from muselsl import stream, list_muses
muses = list_muses(); stream(muses[0]['address'])
```

## Real Device Control
The `RealTimeInferenceEngine.dispatch(epoch, execute=True)` call will
fire real OS-level commands via PyAutoGUI when confidence > threshold.

## Next Steps for Production
1. Collect real EEG data (5-10 min per user, calibration)
2. Fine-tune on subject-specific data (transfer learning)
3. Export to ONNX for Raspberry Pi / Jetson Nano deployment
4. Add SSVEP or P300 paradigm for higher accuracy
5. Build mobile companion app via AccessibilityService (Android)
