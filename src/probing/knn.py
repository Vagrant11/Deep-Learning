import torch
import torch.nn.functional as F


def extract_features(encoder, loader, device, max_samples=None):
    encoder.eval()
    features = []
    labels = []
    seen = 0

    with torch.no_grad():
        for x, y in loader:
            if max_samples is not None and seen >= max_samples:
                break

            if max_samples is not None:
                remaining = max_samples - seen
                x = x[:remaining]
                y = y[:remaining]

            x = x.to(device)
            z = encoder(x).view(x.size(0), -1)
            z = F.normalize(z, dim=1)

            features.append(z.cpu())
            labels.append(y.cpu())
            seen += x.size(0)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def knn_accuracy(train_features, train_labels, test_features, test_labels, k=20, chunk_size=512):
    correct = 0
    total = 0
    num_classes = int(train_labels.max().item()) + 1

    train_features = train_features.t().contiguous()

    for start in range(0, test_features.size(0), chunk_size):
        end = min(start + chunk_size, test_features.size(0))
        sims = test_features[start:end].mm(train_features)
        topk_labels = train_labels[sims.topk(k, dim=1).indices]

        preds = []
        for labels in topk_labels:
            preds.append(torch.bincount(labels, minlength=num_classes).argmax())
        preds = torch.stack(preds)

        correct += (preds == test_labels[start:end]).sum().item()
        total += end - start

    return correct / total
