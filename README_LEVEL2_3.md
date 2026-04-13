# BCI Prototype — Level 2 & 3 Upgrade

## What Was Added

Building on the Level 1 EEGNet prototype (95.6% accuracy, ~1.6ms median latency),
this upgrade adds a full production-ready pipeline (Level 2) and a semantic
alignment research scaffold (Level 3).

---

## Part A — Level 2: Production-Ready System

### New Components

| File | Purpose |
|------|---------|
| `bci_level2/bci_level2_pipeline.py` | Main orchestrator — runs the full end-to-end session |
| `bci_level2/agents/claude_agent.py` | Claude AI agent: coarse intent → structured action plan |
| `bci_level2/hardware/hardware_layer.py` | BrainFlow connector + SSVEP + P300 paradigm detectors |
| `bci_level2/execution/execution_engine.py` | OS-level execution: click, scroll, type, email, hotkey |

### Architecture

```
[EEG Hardware / Simulated Source]  ← BrainFlow, Muse, OpenBCI
           ↓
[Paradigm Detector]
  ├── SSVEP (4-class, occipital, 70-90% acc)
  ├── P300  (binary target, parietal, high SNR)
  └── EEGNet CNN (5-class motor imagery, 95.6% sim)
           ↓
[Coarse Intent Label]  e.g. "communication", "scroll_down"
           ↓
[Claude AI Agent]
  ├── Injects user context (contacts, recent apps, prefs)
  ├── Expands intent → structured JSON action plan
  └── Sets confirmation_required for irreversible actions
           ↓
[Confirmation Engine]
  ├── auto   — always confirm (testing)
  ├── blink  — detect alpha-burst eye blink in EEG
  └── timeout — auto-confirm after N seconds
           ↓
[Execution Engine]   simulate=True (safe) | simulate=False (real OS)
  ├── open_app   → subprocess / os.startfile
  ├── type_text  → pyautogui.typewrite
  ├── click      → pyautogui.click
  ├── scroll     → pyautogui.scroll
  ├── hotkey     → pyautogui.hotkey
  └── send_email → mailto: URL / direct API
           ↓
[RL Feedback Loop]  reward weights updated via EMA
```

### Running Level 2

```bash
# Demo (fully simulated, no hardware, no API key)
python3 bci_level2/bci_level2_pipeline.py

# With SSVEP paradigm (most reliable for real hardware)
python3 bci_level2/bci_level2_pipeline.py --ssvep

# With P300 paradigm
python3 bci_level2/bci_level2_pipeline.py --p300

# With real Claude API (intent expansion via LLM)
python3 bci_level2/bci_level2_pipeline.py --api-key YOUR_KEY

# Real OS execution (CAUTION — will move mouse/keyboard)
python3 bci_level2/bci_level2_pipeline.py --execute

# Confirmation modes: auto | deny | timeout | blink
python3 bci_level2/bci_level2_pipeline.py --confirm timeout
```

### Connecting Real Hardware

Replace `SimulatedEEGSource` in the pipeline with `BrainFlowSource`:

```python
from hardware.hardware_layer import BrainFlowSource
src = BrainFlowSource(board_id=0, serial_port='/dev/ttyUSB0')  # OpenBCI Cyton
# src = BrainFlowSource(board_id=38)  # Muse 2 (Bluetooth)
src.start()
epoch = src.get_epoch(512)  # 2s at 256Hz
src.stop()
```

### Claude API Integration

The `ClaudeIntentAgent` calls `claude-sonnet-4-20250514` with:
- A strict JSON output system prompt
- The user's context (contacts, recent apps, preferences)
- The coarse EEG intent label

Example response for `intent = "communication"`:
```json
{
  "intent_label": "communication",
  "interpreted_action": "Compose email to alice@example.com",
  "steps": [
    {"step": 1, "os_command": "open_app", "params": {"app": "email_client"}},
    {"step": 2, "os_command": "send_email",
     "params": {"to": "alice@example.com", "subject": "Quick message"}}
  ],
  "confirmation_required": true,
  "confirmation_message": "Send email to alice@example.com?"
}
```

---

## Part B — Level 3: Semantic Alignment Research

### File

`bci_level3/semantic_alignment.py`

### Research Question

Can EEG embeddings learned from guided motor-imagery tasks cluster
near semantically related text phrase embeddings, using contrastive
(CLIP-style) training?

### Architecture

```
EEG Epoch (14ch × 512t)
       ↓
[EEGTransformerEncoder]
  - Patch embed: (C × patch_size) → d_model
  - CLS token + positional encoding
  - 3× TransformerEncoderLayer
  - Project → 128-dim L2-normalized embedding
       ↓
[128-dim EEG embedding]  ←──── InfoNCE contrastive loss ────→  [128-dim text embedding]
                                                                        ↑
                                                        [TextEmbeddingModel]
                                                          - 18-phrase vocabulary
                                                          - 5 semantic categories
                                                          - Category-aware init
```

### Results (simulated)

| Epoch | Train Loss | Top-1 Acc | Category Acc |
|-------|-----------|-----------|--------------|
| 1     | 2.78      | 8.1%      | 29.3% |
| 10    | 0.87      | 17.9%     | 53.7% |
| 20    | 0.53      | 21.5%     | **56.9%** |

- Random baseline (5 categories): 20.0%
- Simulated result: **56.9% category accuracy** — meaningful above random
- Top-1 phrase matching: ~21% (harder, expected with noisy EEG)

### Running Level 3

```bash
python3 bci_level3/semantic_alignment.py
```

### Research Interpretation

| Result | Meaning |
|--------|---------|
| Cat acc ≈ 20% (random) | No semantic structure found |
| Cat acc 40–60% | Coarse categorical clustering (promising) |
| Cat acc > 60% on new subjects | Generalizable semantic BCI signal |
| Top-1 > 30% | Phrase-level decoding beginning to work |

Real feasibility bar: **≥60% category accuracy on held-out subjects**
with real EEG data collected under controlled paradigm conditions.

---

## Roadmap to Production

### Immediate (0–3 months)
- [ ] Collect 5–10 min calibration data per user with SSVEP or P300 paradigm
- [ ] Fine-tune EEGNet on subject-specific data (transfer learning)
- [ ] Deploy on Raspberry Pi 5 / Jetson Nano (export to ONNX)
- [ ] Build confirmation UI (audio/visual feedback to user)

### Near-term (3–9 months)
- [ ] Android companion app via AccessibilityService API
- [ ] Real EEG data collection for Level 3 contrastive training
- [ ] Multimodal fusion: EEG + eye tracking (gaze as X/Y, EEG as trigger)
- [ ] Subject adaptation: online fine-tuning after each session

### Research (9–24 months)
- [ ] Evaluate semantic alignment on real EEG data
- [ ] Pre-train EEG foundation model (similar to BENDR, LaBraM)
- [ ] Cross-subject generalization via domain adaptation
- [ ] P300 speller → word-level BCI typing system

---

## Key Constraints (Honest Assessment)

1. **Semantic decoding is unsolved** — current non-invasive EEG cannot
   reliably decode arbitrary semantic thought. This system relies on
   constrained paradigms (motor imagery, SSVEP, P300).

2. **Inter-subject variability** — EEG patterns differ significantly
   between people. Each user needs calibration.

3. **Signal-to-noise** — consumer EEG devices (Muse, OpenBCI) have
   much lower SNR than research-grade equipment. Expect lower accuracy.

4. **Latency** — the current median (~1.6ms) is for inference only.
   Include signal buffering (250–500ms epoch) for realistic end-to-end latency.

5. **The hybrid approach is not a compromise** — it is the correct
   engineering choice. EEG as trigger/intent activator + voice/eye for
   semantic content is what state-of-the-art BCIs actually use.
