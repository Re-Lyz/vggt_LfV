import numpy as np


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R = R_est @ R_gt.T
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.degrees(np.arccos(trace))
    return float(ang)

def pose_from_extrinsic_w2c(E_w2c: np.ndarray):
    if E_w2c.shape == (4, 4):
        R = E_w2c[:3, :3]
        t = E_w2c[:3, 3:4]
    else:
        R = E_w2c[:3, :3]
        t = E_w2c[:3, 3:4]
    c = -R.T @ t
    return R, t, c.squeeze(-1)

def umeyama_alignment(X: np.ndarray, Y: np.ndarray, with_scale=True):
    assert X.shape[0] == 3 and Y.shape[0] == 3 and X.shape[1] == Y.shape[1]
    n = X.shape[1]
    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)
    Xc = X - mu_x
    Yc = Y - mu_y
    S = (Yc @ Xc.T) / n
    U, D, Vt = np.linalg.svd(S)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    var_y = (Yc**2).sum() / n
    if with_scale:
        s = np.trace(np.diag(D)) / (var_y + 1e-12)
    else:
        s = 1.0
    t = mu_x - s * (R @ mu_y)
    return s, R, t

def depth_metrics(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray):
    m = valid_mask & np.isfinite(gt) & np.isfinite(pred)
    if m.sum() == 0:
        return dict(count=0, si_log_rmse=np.nan, absrel=np.nan, rmse=np.nan, d1=np.nan, d2=np.nan, d3=np.nan)
    p = pred[m].astype(np.float64)
    g = gt[m].astype(np.float64)
    eps = 1e-6
    d = np.log(np.clip(p, eps, None)) - np.log(np.clip(g, eps, None))
    n = d.size
    silog = np.sqrt( (d**2).mean() - (d.mean()**2) )
    absrel = np.mean(np.abs(p - g) / np.maximum(g, eps))
    rmse = np.sqrt(np.mean((p - g) ** 2))
    ratio = np.maximum(p / np.maximum(g, eps), g / np.maximum(p, eps))
    d1 = np.mean(ratio < 1.25)
    d2 = np.mean(ratio < 1.25**2)
    d3 = np.mean(ratio < 1.25**3)
    return dict(count=int(n), si_log_rmse=float(silog), absrel=float(absrel),
                rmse=float(rmse), d1=float(d1), d2=float(d2), d3=float(d3))
