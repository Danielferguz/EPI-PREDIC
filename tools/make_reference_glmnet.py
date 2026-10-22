# Referencia de regresión penalizada con scikit-learn (datos estandarizados; λ = 1/(n·C) para la logística).
import json, numpy as np
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
rng = np.random.default_rng(5); n, p = 400, 8
X = rng.normal(size=(n, p)); X[:, 1] += 0.5 * X[:, 0]; X = (X - X.mean(0)) / X.std(0)
eta = -0.5 + X @ np.array([1.0, -0.6, 0.0, 0.4, 0, 0, 0.8, 0]); y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
yc = X @ np.array([1.0, -0.6, 0.0, 0.4, 0, 0, 0.8, 0]) + rng.normal(size=n)
out = {'X': X.tolist(), 'y': y.tolist(), 'yc': yc.tolist(), 'fits': []}
for lam in [0.02, 0.08]:
    C = 1 / (n * lam)
    l1 = LogisticRegression(penalty='l1', C=C, solver='saga', tol=1e-10, max_iter=100000).fit(X, y)
    l2 = LogisticRegression(penalty='l2', C=C, solver='lbfgs', tol=1e-12, max_iter=100000).fit(X, y)
    out['fits'].append({'family': 'binomial', 'alpha': 1, 'lambda': lam, 'intercept': float(l1.intercept_[0]), 'beta': l1.coef_[0].tolist()})
    out['fits'].append({'family': 'binomial', 'alpha': 0, 'lambda': lam, 'intercept': float(l2.intercept_[0]), 'beta': l2.coef_[0].tolist()})
    ls = Lasso(alpha=lam, tol=1e-12, max_iter=100000).fit(X, yc)
    rd = Ridge(alpha=lam * n).fit(X, yc)   # Ridge: ||y-Xw||² + α||w||²  ↔  (1/2n)||·||² + (λ/2)||w||²  ⇒ α = nλ
    out['fits'].append({'family': 'gaussian', 'alpha': 1, 'lambda': lam, 'intercept': float(ls.intercept_), 'beta': ls.coef_.tolist()})
    out['fits'].append({'family': 'gaussian', 'alpha': 0, 'lambda': lam, 'intercept': float(rd.intercept_), 'beta': rd.coef_.tolist()})
json.dump(out, open('tests/reference_glmnet.json', 'w'))
print(len(out['fits']))
