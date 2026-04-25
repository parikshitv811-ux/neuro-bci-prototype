import sys, os, torch
sys.path.insert(0, '/content/neuro-bci-prototype')
# Attempting to load from established modules
try:
    from tsta_project.models_v2 import TSTAEncoder
    from tsta_project.losses_v2 import CompositeLoss
    from tsta_project.trainer_v2 import train_loso
except ImportError:
    # Fallback to single-script trainer if modules were lost
    pass 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

config = {
    'data': {'dataset_name': 'PhysionetMI', 'subjects': [1, 2, 3, 4, 5], 'fmin': 4, 'fmax': 40, 'tmin': 0, 'tmax': 4},
    'model': {'n_channels': 64, 'n_timepoints': 160, 'n_classes': 4, 'embed_dim': 128, 'dropout': 0.5},
    'lr': 5e-4,
    'criterion': None # Initialized inside trainer_v2 if modular
}

if __name__ == '__main__':
    try:
        print('🚀 Starting TSTA LOSO Re-run...')
        # Re-running logic here
        !python3 run_fixed_pipeline.py
    except Exception as e:
        print(f'❌ Error: {e}')
