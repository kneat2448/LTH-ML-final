"""Pilot: do lottery tickets fail at high LR because sparse tickets are too sharp?

Iterative magnitude pruning (IMP, reset to theta_0) on Fashion-MNIST with a small
CNN. For every sparsity level we record:
  * lambda_max of the training-loss Hessian at the masked init (theta_0 * m)
  * test accuracy / macro precision / recall / F1 / one-vs-rest AUC
  * the same metrics for a random-reinit baseline (theta_0' * m)
  * early-training loss spike (catapult indicator)
  * Liu et al. overlap ratio R_p(theta_0*m, theta_T) at p = 0.2
Usage: python lt_sharp.py --lr 0.1 --warmup 0 --tag high_nowarm
"""
import argparse, gzip, json, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, default=0.01)
p.add_argument("--warmup", type=float, default=0.0, help="warmup length in epochs")
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--rounds", type=int, default=6)
p.add_argument("--rate", type=float, default=0.5)
p.add_argument("--ntrain", type=int, default=30000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", type=str, required=True)
p.add_argument("--probe", action="store_true", help="dense-only LR probe")
p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()
torch.set_num_threads(2)
torch.manual_seed(a.seed); np.random.seed(a.seed)
D = os.path.join(os.path.dirname(__file__), "data")

def load(kind):
    with gzip.open(f"{D}/{kind}-images-idx3-ubyte.gz") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    with gzip.open(f"{D}/{kind}-labels-idx1-ubyte.gz") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8)
    x = (torch.tensor(x, dtype=torch.float32) / 255.0 - 0.2860) / 0.3530
    return x, torch.tensor(y, dtype=torch.long)

Xtr, Ytr = load("train"); Xte, Yte = load("t10k")
perm = torch.randperm(len(Xtr))[: a.ntrain]; Xtr, Ytr = Xtr[perm], Ytr[perm]
Xtr, Ytr, Xte, Yte = (t.to(a.device) for t in (Xtr, Ytr, Xte, Yte))
Xh, Yh = Xtr[:1024], Ytr[:1024]  # fixed batch for Hessian

class Net(nn.Module):  # Conv-4 style, no BatchNorm
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv2d(1, 16, 3, padding=1); s.c2 = nn.Conv2d(16, 16, 3, padding=1)
        s.c3 = nn.Conv2d(16, 32, 3, padding=1); s.c4 = nn.Conv2d(32, 32, 3, padding=1)
        s.f1 = nn.Linear(32 * 7 * 7, 128); s.f2 = nn.Linear(128, 10)
    def forward(s, x):
        x = F.max_pool2d(F.relu(s.c2(F.relu(s.c1(x)))), 2)
        x = F.max_pool2d(F.relu(s.c4(F.relu(s.c3(x)))), 2)
        return s.f2(F.relu(s.f1(x.flatten(1))))

def prunable(m):
    return [n for n, q in m.named_parameters() if q.dim() > 1]

def apply_mask(m, mask):
    with torch.no_grad():
        for n, q in m.named_parameters():
            if n in mask: q.mul_(mask[n])

def sharpness(m, mask, iters=20):
    """Top Hessian eigenvalue of the loss on a fixed batch, restricted to unpruned weights."""
    params = [q for q in m.parameters()]
    names = [n for n, _ in m.named_parameters()]
    loss = F.cross_entropy(m(Xh), Yh)
    g = torch.autograd.grad(loss, params, create_graph=True)
    v = [torch.randn_like(q) * (mask[n] if n in mask else 1) for n, q in zip(names, params)]
    def power(shift, v):
        lam = 0.0
        for _ in range(iters):
            nv = math.sqrt(sum((x * x).sum() for x in v)); v = [x / nv for x in v]
            hv = torch.autograd.grad(g, params, grad_outputs=v, retain_graph=True)
            hv = [(h - shift * x) * (mask[n] if n in mask else 1) for n, h, x in zip(names, hv, v)]
            lam = sum((h * x).sum() for h, x in zip(hv, v)).item(); v = [h.detach() for h in hv]
        return lam + shift
    lam = power(0.0, v)
    if lam < 0:  # power iteration found the most negative eigenvalue; shift to get lambda_max
        lam = power(lam, [torch.randn_like(q) * (mask[n] if n in mask else 1) for n, q in zip(names, params)])
    return lam

def evaluate(m):
    m.eval()
    with torch.no_grad():
        logits = torch.cat([m(Xte[i:i + 2000]) for i in range(0, len(Xte), 2000)])
    pr = F.softmax(logits, 1).cpu().numpy(); yp = pr.argmax(1); y = Yte.cpu().numpy()
    if not np.isfinite(pr).all():  # diverged run: metrics undefined
        return dict(acc=float("nan"), prec=float("nan"), rec=float("nan"), f1=float("nan"), auc=float("nan"))
    P, R, F1, _ = precision_recall_fscore_support(y, yp, average="macro", zero_division=0)
    return dict(acc=accuracy_score(y, yp), prec=P, rec=R, f1=F1,
                auc=roc_auc_score(y, pr, multi_class="ovr", average="macro"))

lam_early = []
def train(m, mask):
    opt = torch.optim.SGD(m.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4)
    bs = 128; spe = math.ceil(a.ntrain / bs); total = spe * a.epochs; wu = int(a.warmup * spe)
    step, early, diverged = 0, [], False
    lam_early.clear()
    for ep in range(a.epochs):
        m.train(); idx = torch.randperm(a.ntrain)
        for i in range(spe):
            b = idx[i * bs:(i + 1) * bs]
            lr = a.lr * min(1.0, (step + 1) / wu) if wu > 0 else a.lr
            if step >= int(0.75 * total): lr *= 0.1
            for gp in opt.param_groups: gp["lr"] = lr
            loss = F.cross_entropy(m(Xtr[b]), Ytr[b])
            if not torch.isfinite(loss): diverged = True; break
            opt.zero_grad(); loss.backward(); opt.step(); apply_mask(m, mask)
            if step < 200: early.append(loss.item())
            step += 1
            if step == 100: lam_early.append(sharpness(m, mask, iters=10)); m.train()
        if diverged: break
    spike = max(early) / early[0] if early else float("nan")
    return spike, diverged

def overlap(w0, wT, mask, pfrac=0.2):
    """Liu et al. (2021) correlation indicator R_p, pooled over prunable layers."""
    num = den = 0
    for n in mask:
        keep = mask[n].bool().flatten(); a0 = w0[n].flatten()[keep].abs(); aT = wT[n].flatten()[keep].abs()
        k = max(1, int(pfrac * keep.sum().item()))
        s0 = set(torch.topk(a0, k).indices.tolist()); sT = set(torch.topk(aT, k).indices.tolist())
        num += len(s0 & sT); den += k
    return num / den

net = Net().to(a.device); theta0 = {k: v.clone() for k, v in net.state_dict().items()}
names = prunable(net)
mask = {n: torch.ones_like(dict(net.named_parameters())[n]) for n in names}
rows = []; t0 = time.time(); lam_early = []
rounds = 0 if a.probe else a.rounds
for r in range(rounds + 1):
    remain = sum(mask[n].sum().item() for n in names) / sum(mask[n].numel() for n in names)
    # --- winning ticket: theta_0 * m
    net.load_state_dict(theta0); apply_mask(net, mask)
    lam = sharpness(net, mask)
    spike, div = train(net, mask); met = evaluate(net)
    wT = {n: q.detach().clone() for n, q in net.named_parameters()}
    R = overlap(theta0, wT, mask); lam100 = lam_early[0] if lam_early else float('nan')
    row = dict(round=r, remain=remain, lam0=lam, lam100=lam100, thresh=(2 + 2 * 0.9) / a.lr, spike=spike,
               diverged=div, overlap=R, **{f"ticket_{k}": v for k, v in met.items()})
    # --- random reinit baseline: theta_0' * m
    if r > 0:
        rnd = Net().to(a.device); apply_mask(rnd, mask)
        row["lam0_reinit"] = sharpness(rnd, mask)
        s2, d2 = train(rnd, mask); met2 = evaluate(rnd)
        row.update({f"reinit_{k}": v for k, v in met2.items()}, reinit_spike=s2, lam100_reinit=lam_early[0] if lam_early else float('nan'))
    rows.append(row)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
          f"t={time.time() - t0:.0f}s", flush=True)
    # --- prune: global magnitude over surviving weights of the trained ticket
    alive = torch.cat([wT[n][mask[n].bool()].abs() for n in names])
    k = int(a.rate * alive.numel()); thr = torch.kthvalue(alive, k).values
    mask = {n: (mask[n].bool() & (wT[n].abs() > thr)).float() for n in names}

json.dump(dict(args=vars(a), rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pilot_results", f"res_{a.tag}.json"), "w"), indent=1)
