"""
Synthetic EEG — Category Profiles
===================================
Defines the 5 semantic categories and their EEG signal properties.
Each category has:
  - band:       dominant oscillatory frequency range (Hz)
  - drift:      (low_hz_shift, high_hz_shift) — the TEMPORAL DIRECTION SIGNAL
  - erp_ms:     ERP component latency in milliseconds (None for idle)
  - laterality: which electrode region is active
"""

CATEGORIES = {
    0: "communication",
    1: "navigation",
    2: "action",
    3: "selection",
    4: "idle",
}

CATEGORY_PROFILES = {
    0: {
        "band":       (8,  12),
        "drift":      (+2.0, -1.5),
        "erp_ms":     300,
        "laterality": "left",
        "label":      "communication",
    },
    1: {
        "band":       (12, 18),
        "drift":      (-1.5, +2.0),
        "erp_ms":     250,
        "laterality": "right",
        "label":      "navigation",
    },
    2: {
        "band":       (15, 25),
        "drift":      (+3.0, +1.0),
        "erp_ms":     200,
        "laterality": "bilateral",
        "label":      "action",
    },
    3: {
        "band":       (10, 15),
        "drift":      (-2.0, -2.0),
        "erp_ms":     350,
        "laterality": "parietal",
        "label":      "selection",
    },
    4: {
        "band":       (6,   9),
        "drift":      (0.0,  0.0),
        "erp_ms":     None,
        "laterality": "none",
        "label":      "idle",
    },
}
