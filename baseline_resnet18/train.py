"""Train a patient-grouped ResNet-18 baseline on data/{train,test}/{class}."""
import argparse, csv, hashlib, json, random, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ['NORMAL', 'PNEUMONIA']

def patient(path):
    name = path.stem.lower()
    match = re.match(r'(person\d+)', name)
    if match:
        return match.group(1)
    # NORMAL2-IM-xxxx-0001-0001 and -0002 are images of the same study.
    match = re.match(r'((?:normal2-)?im-\d+-\d+)', name)
    return match.group(1) if match else 'file:' + name

def read_data(root, split):
    rows = []
    for label, name in enumerate(CLASSES):
        folder = root / split / name
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing folder: {folder}')
        for p in sorted(folder.rglob('*')):
            if p.suffix.lower() not in ('.jpeg', '.jpg', '.png'): continue
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            rows.append((p, label, patient(p), h))
    if not rows: raise ValueError(f'No images found in {root / split}')
    return rows

def groups(rows):
    # Union patient groups and exact duplicate hashes, including transitive links.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in ('patient:' + row[2], 'hash:' + row[3]):
            if key in seen: parent[find(i)] = find(seen[key])
            else: seen[key] = i
    return [find(i) for i in range(len(rows))]

class Images(Dataset):
    def __init__(self, rows, transform): self.rows, self.transform = rows, transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path, label, _, _ = self.rows[i]
        with Image.open(path) as im: image = self.transform(im.convert('RGB'))
        return image, label

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    p.add_argument('--epochs', type=int, default=10); p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--allow-test-overlap', action='store_true', help='Evaluate despite potential patient/duplicate overlap; results may be biased')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--workers', type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    a.output.mkdir(parents=True, exist_ok=True)
    rows = read_data(a.data, 'train'); test = read_data(a.data, 'test')
    group = groups(rows)
    labels = [r[1] for r in rows]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=a.seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(rows)), labels, group))
    train = [rows[i] for i in train_idx]; val = [rows[i] for i in val_idx]
    # Never silently claim a clean held-out test if the original split overlaps.
    overlaps = set(r[3] for r in rows) & set(r[3] for r in test)
    shared_patients = set(r[2] for r in rows) & set(r[2] for r in test)
    print(f'Train/validation/test: {len(train)}/{len(val)}/{len(test)}')
    print(f'Exact train-test overlaps: {len(overlaps)}; shared filename-derived groups: {len(shared_patients)}')
    if overlaps or shared_patients:
        print('WARNING: Potential train/test leakage; test evaluation will be skipped unless --allow-test-overlap is supplied.')
    with (a.output / 'split.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f); writer.writerow(['split','path','label','group','sha256'])
        for name, subset in [('train', train), ('validation', val), ('test', test)]:
            for path, label, group_id, digest in subset:
                writer.writerow([name, str(path.relative_to(a.data)), CLASSES[label], group_id, digest])
    weights = models.ResNet18_Weights.DEFAULT
    eval_transform = weights.transforms()
    train_transform = transforms.Compose([transforms.Resize((256,256)), transforms.RandomResizedCrop(224, scale=(0.9,1.0)), transforms.RandomRotation(5), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    loaders = {name: DataLoader(Images(subset, train_transform if name == 'train' else eval_transform), batch_size=a.batch_size, shuffle=(name == 'train'), num_workers=a.workers) for name, subset in [('train',train),('validation',val),('test',test)]}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=weights)
    for param in model.parameters(): param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2); model.to(device)
    counts = Counter(r[1] for r in train)
    if len(counts) != 2: raise ValueError('Training split must contain both classes')
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([len(train)/(2*counts[i]) for i in range(2)], device=device))
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    best_loss = float('inf'); best_epoch = 0
    def run(name, training=False):
        model.eval()  # Keep frozen backbone batch-normalization statistics fixed.
        predictions, truth, probabilities = [], [], []
        total_loss = total = 0
        with torch.set_grad_enabled(training):
            for x, y in loaders[name]:
                x, y = x.to(device), y.to(device)
                if training: optimizer.zero_grad()
                logits = model(x); loss = criterion(logits, y)
                if training: loss.backward(); optimizer.step()
                total_loss += loss.item() * len(y); total += len(y)
                truth.extend(y.cpu().tolist()); predictions.extend(logits.argmax(1).cpu().tolist())
                probabilities.extend(logits.softmax(1)[:,1].detach().cpu().tolist())
        return total_loss / total, truth, predictions, probabilities
    for epoch in range(1, a.epochs + 1):
        train_loss, *_ = run('train', True)
        val_loss, *_ = run('validation')
        print(f'Epoch {epoch}: train loss {train_loss:.4f}, validation loss {val_loss:.4f}', flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save({'model_state_dict': model.state_dict(), 'classes': CLASSES, 'epoch': epoch, 'seed': a.seed, 'weights': str(weights)}, a.output / 'best_model.pt')
    checkpoint = torch.load(a.output / 'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    if (overlaps or shared_patients) and not a.allow_test_overlap:
        print('Best checkpoint saved. Test evaluation skipped because of potential overlap.')
        return
    loss, y, pred, prob = run('test')
    report = classification_report(y, pred, labels=[0,1], target_names=CLASSES, output_dict=True, zero_division=0)
    result = {'best_epoch': best_epoch, 'validation_loss': best_loss, 'test_loss': loss, 'confusion_matrix': confusion_matrix(y,pred,labels=[0,1]).tolist(), 'classification_report': report, 'roc_auc': roc_auc_score(y,prob) if len(set(y)) == 2 else None, 'device': str(device), 'train_test_exact_overlap': len(overlaps), 'train_test_shared_groups': len(shared_patients)}
    (a.output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
