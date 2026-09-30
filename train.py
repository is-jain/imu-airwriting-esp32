import argparse
import csv
import os
import sys

import numpy as np

N_PTS, N_AXES = 40, 6
N_FEAT = N_PTS * N_AXES
H1, H2 = 64, 32

def load(path):
    if not os.path.exists(path):
        sys.exit("No data at %s. Run collect.py first." % path)
    labels, feats = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) != N_FEAT + 1:
                continue
            labels.append(row[0])
            feats.append([float(v) for v in row[1:]])
    X = np.array(feats, dtype=np.float32)
    names = sorted(set(labels), key=lambda s: (len(s), s))
    y = np.array([names.index(l) for l in labels])
    return X, y, names

def normalize(X):
    Xr = X.reshape(len(X), N_PTS, N_AXES)
    Xr = Xr - Xr.mean(axis=1, keepdims=True)
    acc = np.abs(Xr[:, :, :3]).max(axis=(1, 2), keepdims=True) + 1e-6
    gyr = np.abs(Xr[:, :, 3:]).max(axis=(1, 2), keepdims=True) + 1e-6
    Xr = np.concatenate([Xr[:, :, :3] / acc, Xr[:, :, 3:] / gyr], axis=2)
    return Xr.reshape(len(X), N_FEAT)

def augment(X, y, copies, rng):
    outX, outY = [X], [y]
    t = np.linspace(0, 1, N_PTS)
    for _ in range(copies):
        Xr = X.reshape(len(X), N_PTS, N_AXES).copy()
        for i in range(len(Xr)):
            k = rng.uniform(-0.3, 0.3)
            warped = t + k * np.sin(np.pi * t)
            warped = (warped - warped.min()) / (warped.max() - warped.min())
            for a in range(N_AXES):
                Xr[i, :, a] = np.interp(warped, t, Xr[i, :, a])
            s = rng.integers(-2, 3)
            if s:
                Xr[i] = np.roll(Xr[i], s, axis=0)
                if s > 0:
                    Xr[i, :s] = Xr[i, s]
                else:
                    Xr[i, s:] = Xr[i, s - 1]
        Xr *= rng.uniform(0.85, 1.15, size=(len(Xr), 1, N_AXES))
        Xr += rng.normal(0, 0.03, size=Xr.shape)
        outX.append(normalize(Xr.reshape(len(Xr), N_FEAT)))
        outY.append(y)
    return np.concatenate(outX), np.concatenate(outY)

def stratified_split(y, val_frac, rng):
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac))) if len(idx) > 2 else 0
        va.extend(idx[:n_val])
        tr.extend(idx[n_val:])
    return np.array(tr), np.array(va)

class MLP:
    def __init__(self, n_in, h1, h2, n_out, rng):
        def init(i, o):
            return (rng.normal(0, np.sqrt(2.0 / i), size=(o, i))).astype(np.float32)
        self.params = {
            "W1": init(n_in, h1), "B1": np.zeros(h1, np.float32),
            "W2": init(h1, h2), "B2": np.zeros(h2, np.float32),
            "W3": init(h2, n_out), "B3": np.zeros(n_out, np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def forward(self, X, train=False, drop=0.0, rng=None):
        p = self.params
        z1 = X @ p["W1"].T + p["B1"]
        a1 = np.maximum(z1, 0)
        if train and drop > 0:
            mask1 = (rng.random(a1.shape) > drop) / (1 - drop)
            a1 = a1 * mask1
        z2 = a1 @ p["W2"].T + p["B2"]
        a2 = np.maximum(z2, 0)
        if train and drop > 0:
            mask2 = (rng.random(a2.shape) > drop) / (1 - drop)
            a2 = a2 * mask2
        logits = a2 @ p["W3"].T + p["B3"]
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=1, keepdims=True)
        cache = (X, z1, a1, z2, a2, probs)
        return probs, cache

    def backward(self, cache, y, wd):
        X, z1, a1, z2, a2, probs = cache
        n = len(X)
        p = self.params
        d3 = probs.copy()
        d3[np.arange(n), y] -= 1
        d3 /= n
        g = {}
        g["W3"] = d3.T @ a2 + wd * p["W3"]
        g["B3"] = d3.sum(0)
        d2 = (d3 @ p["W3"]) * (z2 > 0)
        g["W2"] = d2.T @ a1 + wd * p["W2"]
        g["B2"] = d2.sum(0)
        d1 = (d2 @ p["W2"]) * (z1 > 0)
        g["W1"] = d1.T @ X + wd * p["W1"]
        g["B1"] = d1.sum(0)
        return g

    def adam(self, g, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.params:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.params[k] -= lr * mh / (np.sqrt(vh) + eps)

    def predict(self, X):
        return self.forward(X)[0].argmax(axis=1)

def train(model, X, y, Xv, yv, epochs, lr, batch, rng):
    n = len(X)
    best = (0.0, None)
    for ep in range(epochs):
        idx = rng.permutation(n)
        cur_lr = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)))
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            probs, cache = model.forward(X[b], train=True, drop=0.2, rng=rng)
            g = model.backward(cache, y[b], wd=1e-4)
            model.adam(g, cur_lr)
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            tr_acc = (model.predict(X) == y).mean()
            va_acc = (model.predict(Xv) == yv).mean() if len(Xv) else float("nan")
            print("epoch %4d  train %.3f  val %.3f" % (ep + 1, tr_acc, va_acc))
            if len(Xv) and va_acc >= best[0]:
                best = (va_acc, {k: v.copy() for k, v in model.params.items()})
    if best[1] is not None:
        model.params = best[1]
    return model

def c_array(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        body = ", ".join("%.6ff" % v for v in arr)
        return "const float %s[%d] = {%s};\n" % (name, arr.shape[0], body)
    rows = []
    for r in arr:
        rows.append("  {" + ", ".join("%.6ff" % v for v in r) + "}")
    return "const float %s[%d][%d] = {\n%s\n};\n" % (name, arr.shape[0], arr.shape[1], ",\n".join(rows))

def export_header(path, p, names):
    with open(path, "w") as f:
        f.write("#pragma once\n")
        f.write("#define MODEL_READY 1\n")
        f.write("#define H1 %d\n#define H2 %d\n#define N_CLASSES %d\n" % (H1, H2, len(names)))
        f.write("static const char* const CLASS_NAMES[N_CLASSES] = {%s};\n"
                % ", ".join('"%s"' % n for n in names))
        for k in ("W1", "B1", "W2", "B2", "W3", "B3"):
            f.write(c_array("MODEL_" + k, p[k]))

def c_style_forward(p, x):
    h1 = np.maximum(p["W1"] @ x + p["B1"], 0)
    h2 = np.maximum(p["W2"] @ h1 + p["B2"], 0)
    l = p["W3"] @ h2 + p["B3"]
    e = np.exp(l - l.max())
    return e / e.sum()

def confusion(y_true, y_pred, names):
    k = len(names)
    m = np.zeros((k, k), dtype=int)
    for t, pr in zip(y_true, y_pred):
        m[t, pr] += 1
    head = "true\\pred " + " ".join("%4s" % n for n in names)
    print(head)
    for i, n in enumerate(names):
        print("%9s " % n + " ".join("%4d" % v for v in m[i]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/gestures.csv")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--aug", type=int, default=6, help="augmented copies per sample")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true",
                    help="after evaluating, retrain on all data for the exported model")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y, names = load(args.data)
    X = normalize(X)
    print("Loaded %d samples, classes: %s" % (len(X), " ".join(names)))
    for i, n in enumerate(names):
        print("  %s: %d" % (n, (y == i).sum()))

    tr, va = stratified_split(y, args.val, rng)
    Xtr, ytr = augment(X[tr], y[tr], args.aug, rng)
    Xva, yva = X[va], y[va]
    print("Training on %d (incl. augmented), validating on %d" % (len(Xtr), len(Xva)))

    model = MLP(N_FEAT, H1, H2, len(names), rng)
    model = train(model, Xtr, ytr, Xva, yva, args.epochs, args.lr, 32, rng)

    if len(Xva):
        pred = model.predict(Xva)
        print("\nValidation accuracy: %.1f%%" % (100 * (pred == yva).mean()))
        confusion(yva, pred, names)

    if args.all:
        print("\nRetraining on all data for export...")
        Xall, yall = augment(X, y, args.aug, rng)
        model = MLP(N_FEAT, H1, H2, len(names), rng)
        model = train(model, Xall, yall, Xva[:0], yva[:0], args.epochs, args.lr, 32, rng)

    p = model.params
    export_header("model.h", p, names)
    np.savez("model.npz", names=np.array(names), **p)

    probs, _ = model.forward(X[:8])
    worst = max(np.abs(c_style_forward(p, X[i]) - probs[i]).max() for i in range(min(8, len(X))))
    print("\nExport check: max |C-style - numpy| = %.2e (%s)" % (worst, "ok" if worst < 1e-4 else "MISMATCH"))
    n_params = sum(v.size for v in p.values())
    print("Wrote model.h (%d params, ~%d KB flash) and model.npz" % (n_params, n_params * 4 // 1024))
    print("Now re-upload the sketch in Arduino IDE, then run demo.py.")

if __name__ == "__main__":
    main()
