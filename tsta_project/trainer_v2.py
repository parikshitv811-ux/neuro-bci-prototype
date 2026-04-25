
import torch, torch.optim as optim, numpy as np, os, json

def train_epoch(model, dataloader, optimizer, criterion, device='cpu'):
    model.train(); total_loss, n_batches = 0, 0
    for batch in dataloader:
        x, y, subj = batch['eeg'].to(device), batch['label'].to(device), batch['subject'].to(device)
        optimizer.zero_grad(); logits, emb = model(x, subject_ids=subj)
        loss = criterion(emb, y, subj, model.prototypes)['total']
        loss.backward(); optimizer.step(); total_loss += loss.item(); n_batches += 1
    return total_loss / n_batches

def train_loso(model_class, config, device='cpu', n_epochs=100, save_dir='./results'):
    from tsta_project.data_loader_v2 import get_loso_loaders
    from tsta_project.evaluator_v2 import evaluate_model
    os.makedirs(save_dir, exist_ok=True); splits = get_loso_loaders(**config['data']); results = []
    print(f"\n{'='*60}\nLOSO TRAINING | Device: {device} | Epochs: {n_epochs} | Folds: {len(splits)}\n{'='*60}")
    for fold_idx, split in enumerate(splits):
        test_subj = split['test_subject']
        print(f"\nFOLD {fold_idx+1}/{len(splits)} | Test Subject: {test_subj}")
        model = model_class(n_subjects=split['n_train_subjects'], **config['model']).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        criterion = config['criterion']; scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        best_acc, best_state = 0, None
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, split['train'], optimizer, criterion, device); scheduler.step()
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                metrics = evaluate_model(model, split['test'], device)
                if metrics['accuracy'] > best_acc:
                    best_acc = metrics['accuracy']; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  Ep {epoch:3d}: loss={train_loss:.4f} | acc={metrics['accuracy']:.3f} | cdas={metrics['cdas']:.4f}")
        if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        final = evaluate_model(model, split['test'], device); final['test_subject'] = test_subj; results.append(final)
        print(f"  FINAL: acc={final['accuracy']:.3f} | cdas={final['cdas']:.4f}")
    mean_acc = np.mean([r['accuracy'] for r in results]); std_acc = np.std([r['accuracy'] for r in results]); mean_cdas = np.mean([r['cdas'] for r in results])
    print(f"\n{'='*60}\nFINAL: Acc={mean_acc:.3f}±{std_acc:.3f} | CDAS={mean_cdas:.4f}\n{'='*60}")
    with open(os.path.join(save_dir, 'loso_results.json'), 'w') as f:
        json.dump({'per_fold': results, 'aggregate': {'mean_accuracy': float(mean_acc), 'std_accuracy': float(std_acc), 'mean_cdas': float(mean_cdas)}}, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    return results
