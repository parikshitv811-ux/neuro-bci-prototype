
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from tsta_project.models_v2 import TSTAEncoder
from tsta_project.losses_v2 import CompositeLoss
from tsta_project.trainer_v2 import train_loso

print("="*60)
print("TSTA BCI - REAL PYTORCH PIPELINE")
print("="*60)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

config = {
    'data': {
        'dataset_name': 'PhysionetMI',
        'subjects': [1, 2, 3, 4, 5],
        'fmin': 4, 'fmax': 40, 'tmin': 0, 'tmax': 4
    },
    'model': {
        'n_channels': 64, 'n_timepoints': 160, 'n_classes': 4,
        'embed_dim': 128, 'dropout': 0.5
    },
    'lr': 1e-3,
    'criterion': CompositeLoss(lambda_ca=0.1, lambda_proto=0.01, temperature=0.07)
}

results = train_loso(TSTAEncoder, config, device=device, n_epochs=30, save_dir='./results')
print("\nPipeline complete! Check ./results/loso_results.json")
