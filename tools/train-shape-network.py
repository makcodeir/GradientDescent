"""Train the small, fixed-weight classroom demo. NumPy is the only dependency."""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(42)
yy, xx = np.mgrid[:20, :20]

def sample(label):
    cx, cy = rng.uniform(8.4, 10.6, 2)
    radius = rng.uniform(5.5, 8)
    x, y = xx - cx, yy - cy
    if label == 0:
        distance = np.abs(np.maximum(np.abs(x), np.abs(y)) - radius)
    elif label == 1:
        distance = np.abs(np.hypot(x, y) - radius)
    else:
        points = [(cx, cy-radius), (cx+radius, cy+radius), (cx-radius, cy+radius)]
        distances = []
        for a, b in zip(points, points[1:]+points[:1]):
            ax, ay = a; bx, by = b
            t = np.clip(((xx-ax)*(bx-ax)+(yy-ay)*(by-ay))/((bx-ax)**2+(by-ay)**2), 0, 1)
            distances.append(np.hypot(xx-ax-t*(bx-ax), yy-ay-t*(by-ay)))
        distance = np.minimum.reduce(distances)
    return np.clip(rng.uniform(1.1, 1.6)-distance, 0, 1).ravel()

labels = np.arange(3000) % 3
X = np.array([sample(int(y)) for y in labels])
W = rng.normal(0, .08, (400, 8)); b = np.full(8, .1)
V = rng.normal(0, .3, (8, 3)); c = np.zeros(3)
params = [W, b, V, c]
moments = [np.zeros_like(p) for p in params]; variances = [np.zeros_like(p) for p in params]
step = 0
for epoch in range(160):
    for ids in np.array_split(rng.permutation(len(X)), 24):
        x = X[ids]; a = np.maximum(0, x @ W + b)
        logits = a @ V + c; logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
        p[np.arange(len(ids)), labels[ids]] -= 1; p /= len(ids)
        da = (p @ V.T) * (a > 0)
        grads = [x.T @ da + .0002*W, da.sum(axis=0), a.T @ p + .0002*V, p.sum(axis=0)]
        step += 1
        for param, grad, m, v in zip(params, grads, moments, variances):
            m *= .9; m += .1*grad; v *= .999; v += .001*grad*grad
            param -= .003 * (m/(1-.9**step)) / (np.sqrt(v/(1-.999**step))+1e-8)
model = {key: np.round(value, 7).tolist() for key, value in zip(['W','b','V','c'],params)}
Path(__file__).resolve().parents[1].joinpath('shape-network.json').write_text(json.dumps(model,separators=(',',':')))
