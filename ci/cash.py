"""Cash layer for companies that have aggregate finance extracts instead of invoice-level payment history.

Inputs (canonical, built by an adapter):
  award_m    Series  month 'YYYY-MM' -> value awarded that month (from the pipeline model panel)
  invoice_m  Series  month -> value invoiced (sales extract)
  collect_m  Series  month -> cash collected (collections extract)
  ar         DataFrame, one row per customer from the latest receivables snapshot:
             customer_id, customer, team, terms_days, not_due, ovd_0_30, ovd_30_60, ovd_60_365, ovd_365p, late_days

Outputs feed the stress test: award->invoice lag profile, invoice->cash lag profile (both fitted by non-negative
least squares on monthly totals), backlog invoicing still to come from recent awards, and the expected collection
timing of every open receivable.
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls


def _months(a, b):
    return [str(p) for p in pd.period_range(a, b, freq='M')]


def fit_lag(x, y, k, months):
    """y[m] ~ sum_j w_j * x[m-j], j=0..k-1, w >= 0. Returns weights and fit R^2."""
    X = np.array([[x.get(str(pd.Period(m, 'M') - j), 0.0) for j in range(k)] for m in months])
    Y = np.array([y.get(m, 0.0) for m in months])
    w, _ = nnls(X, Y)
    pred = X @ w
    r2 = 1 - ((Y - pred) ** 2).sum() / max(1e-9, ((Y - Y.mean()) ** 2).sum())
    return w, float(r2)


def fit_geo(x, y, months, K=8):
    """y[m] ~ s * sum_j w_j x[m-j], w geometric with mean lag d months (grid search). Returns w, s, d, R^2."""
    X = np.array([[x.get(str(pd.Period(m, 'M') - j), 0.0) for j in range(K)] for m in months])
    Y = np.array([y.get(m, 0.0) for m in months])
    best = None
    for d in np.arange(0.25, 4.01, 0.25):
        q = 1 / (1 + d)
        w = q * (1 - q) ** np.arange(K)
        w = w / w.sum()
        p = X @ w
        sc = float(p @ Y / max(1e-9, p @ p))
        r2 = 1 - ((Y - sc * p) ** 2).sum() / max(1e-9, ((Y - Y.mean()) ** 2).sum())
        if best is None or r2 > best[3]:
            best = (w, sc, float(d), float(r2))
    return best


def build(award_m, invoice_m, collect_m, ar, as_of, H=6, fit_from=None, invoice_all=None):
    out = {}
    end = as_of
    start = fit_from or str(pd.Period(end, 'M') - 18)
    ms = _months(start, end)
    w_inv, r2i = fit_lag(award_m, invoice_m, 9, ms)
    if r2i < 0.6:   # noisy fit: smooth the profile (3-month moving average, same total)
        tot = w_inv.sum()
        w_inv = np.convolve(np.pad(w_inv, 1, mode='edge'), np.ones(3) / 3, mode='valid')
        w_inv = w_inv * tot / max(1e-9, w_inv.sum())
    inv_cash = invoice_all if invoice_all is not None else invoice_m
    w_col, col_scale, mean_lag, r2c = fit_geo(inv_cash, collect_m, ms)
    src = 'fitted'
    if r2c < 0.2:   # monthly totals too flat to fit: use DSO from receivables and the recent cash/billing ratio
        m6 = _months(str(pd.Period(end, 'M') - 5), end)
        avg_inv = float(np.mean([inv_cash.get(m, 0.0) for m in m6]))
        ar_live = float(ar[['not_due', 'ovd_0_30', 'ovd_30_60', 'ovd_60_365']].sum().sum())
        mean_lag = float(np.clip(ar_live / max(1.0, avg_inv), 0.5, 4.0))
        q = 1 / (1 + mean_lag)
        w_col = q * (1 - q) ** np.arange(8)
        w_col = w_col / w_col.sum()
        col_scale = float(np.mean([collect_m.get(m, 0.0) for m in m6]) / max(1.0, avg_inv))
        src = 'dso'
    other = inv_cash.sub(invoice_m, fill_value=0)              # invoicing outside the modelled product groups
    out['other_inv'] = round(float(other.reindex(_months(str(pd.Period(end, 'M') - 5), end)).fillna(0).mean()))
    out['past_inv'] = [round(float(inv_cash.get(str(pd.Period(end, 'M') - j), 0.0))) for j in range(8)]
    out['col_scale'] = round(col_scale, 4)
    out['inv_lag'] = [round(float(v), 4) for v in w_inv]
    out['col_lag'] = [round(float(v), 4) for v in w_col]
    out['fit'] = dict(invoice_r2=round(r2i, 3), collect_r2=round(r2c, 3), invoice_per_award=round(float(w_inv.sum()), 3),
                      cash_mean_lag_months=round(mean_lag, 2), cash_scale=round(col_scale, 3), cash_method=src)
    # invoicing still to come from awards already made (months end-8 .. end)
    bl = np.zeros(H)
    for j in range(0, 9):
        m = str(pd.Period(end, 'M') - j)
        a = award_m.get(m, 0.0)
        for k in range(1, H + 1):
            lag = j + k
            if lag < len(w_inv):
                bl[k - 1] += a * w_inv[lag]
    out['backlog_inv'] = [round(float(v)) for v in bl]
    # invoicing already raised but not yet collected is in AR, so collections from new invoicing start at month 1
    items = []
    for r in ar.itertuples():
        L = float(np.clip(r.late_days if pd.notna(r.late_days) else 20, 0, 150))
        T = float(r.terms_days or 0)
        for amt, days in ((r.not_due, T / 2 + L), (r.ovd_0_30, 15 + L / 2), (r.ovd_30_60, 30 + L / 2),
                          (r.ovd_60_365, 60 + L)):
            if amt and amt > 0:
                items.append([round(float(amt)), int(max(10, days))])
    out['ar_items'] = items
    out['ar_summary'] = dict(total=float(ar[['not_due', 'ovd_0_30', 'ovd_30_60', 'ovd_60_365', 'ovd_365p']].sum().sum()),
                             doubtful_365p=float(ar.ovd_365p.sum()), customers=int(len(ar)))
    hist = _months(str(pd.Period(end, 'M') - 11), end)
    out['history'] = dict(months=hist, award=[round(award_m.get(m, 0.0)) for m in hist],
                          invoiced=[round(invoice_m.get(m, 0.0)) for m in hist],
                          collected=[round(collect_m.get(m, 0.0)) for m in hist])
    return out
