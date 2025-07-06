import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def multilabel_weighted_cross_entropy(preds, targets):
    """
    Weighted cross-entropy for multi-label classification.
    preds: list of [B x 1] tensors from each head
    targets: [B x 5] tensor
    """
    losses = []
    B, b = targets.shape
    m = B
    for i in range(b):
        y_true = targets[:, i].float()
        y_pred = preds[i].squeeze(1)

        si = y_true.sum()
        ti = m - si
        wsi = si / m
        wti = ti / m

        loss = - (wsi * y_true * torch.log(y_pred + 1e-6) +
                  wti * (1 - y_true) * torch.log(1 - y_pred + 1e-6)).mean()
        losses.append(loss)
    return torch.stack(losses).sum()


def conditional_probability_loss(batch_preds, batch_targets):
    """
    Calculates L1 loss between observed and true conditional probabilities.
    """
    # Convert list of [B x 1] to [B x 5]
    pred_bin = torch.cat([(p > 0.5).float() for p in batch_preds], dim=1)
    target_bin = batch_targets.float()

    def conditional_prob(a, b):
        joint = ((a == 1) & (b == 1)).sum().item()
        cond = (b == 1).sum().item()
        return joint / (cond + 1e-6)

    indices = {'H': 0, 'C': 1, 'X': 2, 'M': 3, 'T': 4}
    P = lambda i, j: conditional_prob(pred_bin[:, indices[i]], pred_bin[:, indices[j]])
    Pr = lambda i, j: conditional_prob(target_bin[:, indices[i]], target_bin[:, indices[j]])

    loss = sum([
        abs(Pr('T', 'H') - P('T', 'H')),
        abs(Pr('M', 'H') - P('M', 'H')),
        abs(Pr('C', 'X') - P('C', 'X')),
        abs(Pr('X', 'H') - P('X', 'H')),
        abs(Pr('T', 'C') - P('T', 'C'))
    ])
    return torch.tensor(loss, requires_grad=True).to(batch_targets.device)


def spectral_graph_loss(preds, targets):
    """
    Graph similarity loss using Laplacian eigenvalues.
    preds, targets: binary vectors of shape [B x 5]
    """
    B, b = targets.shape
    total_loss = 0.0
    for i in range(B):
        v = preds[i].float()
        vr = targets[i].float()
        W = torch.exp(-2 * torch.abs(v.unsqueeze(0) - v.unsqueeze(1)))
        Wr = torch.exp(-2 * torch.abs(vr.unsqueeze(0) - vr.unsqueeze(1)))

        D = torch.diag(W.sum(1))
        Dr = torch.diag(Wr.sum(1))

        L = torch.eye(b).to(v.device) - torch.linalg.multi_dot([torch.linalg.inv(D).sqrt(), W, torch.linalg.inv(D).sqrt()])
        Lr = torch.eye(b).to(v.device) - torch.linalg.multi_dot([torch.linalg.inv(Dr).sqrt(), Wr, torch.linalg.inv(Dr).sqrt()])

        eigvals = torch.linalg.eigvalsh(L)
        eigvals_r = torch.linalg.eigvalsh(Lr)

        k = min(b, int(0.9 * b))  # top 90% energy
        total_loss += F.mse_loss(eigvals[:k], eigvals_r[:k])
    return total_loss / B


def composite_loss(preds, targets, alpha=0.5, beta=0.2, gamma=0.3):
    lc = multilabel_weighted_cross_entropy(preds, targets)
    lp = conditional_probability_loss(preds, targets)
    pred_bin = torch.cat([(p > 0.5).float() for p in preds], dim=1)
    lg = spectral_graph_loss(pred_bin, targets)
    return alpha * lc + beta * lp + gamma * lg
