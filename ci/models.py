"""Predictive layer.

1. Win probability  - P(quotation is awarded within the next 6 months), per open quotation, from its stage, size,
                      age, deal momentum (stage moves, award-date slips) and the out-of-time track record of the
                      customer and the sales engineer. Benchmarked against the CRM stage percentage.
2. Award timing     - empirical distribution of the award month given the months left to the expected award date.
3. Forecast inputs  - per-quotation probabilities + timing, run-rate of business from quotations not yet raised,
                      award->invoice and invoice->cash lags. The Monte Carlo itself runs in the browser (stress
                      test tab) and in `backtest_forecast` here.
4. Churn risk       - P(an active account goes quiet: no award and no quotation in the next 6 months).
5. Next best action - rules on top of 1-4 per sales engineer.

All evaluation is out-of-time: train on older snapshots, test on later ones.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

from .schema import OPEN_STAGES, WON_STAGES

H = 6  # horizon (months)


def _mi(s):
    """'YYYY-MM' (or date) -> integer month index."""
    s = pd.Series(s).astype(str).str[:7]
    return (s.str[:4].astype(int) * 12 + s.str[5:7].astype(int) - 1).values


# ------------------------------------------------------------------------------------------------- features
def build_panel(snap):
    """One row per (open quotation, snapshot) with features and outcome columns."""
    s = snap[['snapshot', 'quote_id', 'customer_id', 'se', 'team', 'region', 'product_group', 'value', 'stage',
              'status', 'valid_from', 'expected_award']].copy()
    s['t'] = _mi(s.snapshot)
    s = s.sort_values(['quote_id', 't'])
    g = s.groupby('quote_id', sort=False)
    s['prev_stage'] = g.stage.shift()
    s['prev_ead'] = g.expected_award.shift()
    s['up'] = (s.stage > s.prev_stage).astype(int)
    s['slip'] = ((s.expected_award.str[:7] > s.prev_ead.str[:7]) & s.prev_ead.notna()).astype(int)
    s['stage_moves'] = g.up.cumsum()
    s['ead_slips'] = g.slip.cumsum()
    s['stage_max'] = g.stage.cummax()
    chg = (s.stage != s.prev_stage).astype(int)
    s['blk'] = chg.groupby(s.quote_id).cumsum()
    s['months_in_stage'] = s.groupby(['quote_id', 'blk']).cumcount()
    s['age_m'] = s.t - _mi(s.valid_from)
    s['to_ead_m'] = _mi(s.expected_award) - s.t
    s['log_value'] = np.log1p(s.value.clip(lower=0))
    s['moy'] = s.t % 12

    # outcome: first month the quotation is won / lost
    res = s[s.status != 'OPEN'].groupby('quote_id').agg(res_t=('t', 'min'), res_status=('status', 'first'))
    s = s.join(res, on='quote_id')
    s['won_t'] = np.where(s.res_status == 'WON', s.res_t, np.nan)

    # out-of-time track record of customer and SE (resolved before this snapshot), smoothed to the overall rate
    r = res.join(s.drop_duplicates('quote_id').set_index('quote_id')[['customer_id', 'se']])
    r['w'] = (r.res_status == 'WON').astype(int)
    T = np.arange(s.t.min(), s.t.max() + 1)
    prior = r.w.mean() if len(r) else 0.5
    for key, k in (('customer_id', 4.0), ('se', 25.0)):
        cnt = r.groupby([key, 'res_t']).w.agg(['sum', 'count']).unstack('res_t', fill_value=0)
        cw = cnt['sum'].reindex(columns=T, fill_value=0).cumsum(axis=1).shift(1, axis=1).fillna(0)
        cn = cnt['count'].reindex(columns=T, fill_value=0).cumsum(axis=1).shift(1, axis=1).fillna(0)
        rate = (cw + k * prior) / (cn + k)
        st = rate.stack().rename(f'{key}_wr')
        nn = cn.stack().rename(f'{key}_n')
        s = s.join(st, on=[key, 't']).join(nn, on=[key, 't'])
        s[f'{key}_wr'] = s[f'{key}_wr'].fillna(prior)
        s[f'{key}_n'] = s[f'{key}_n'].fillna(0)
    return s


FEATS = ['moy', 'stage', 'log_value', 'age_m', 'to_ead_m', 'stage_moves', 'ead_slips', 'stage_max', 'months_in_stage',
         'customer_id_wr', 'customer_id_n', 'se_wr', 'se_n', 'pg', 'tm', 'rg']


def _X(p, cats):
    X = p.copy()
    for c, col in (('pg', 'product_group'), ('tm', 'team'), ('rg', 'region')):
        X[c] = pd.Categorical(X[col], categories=cats[col]).codes
    return X[FEATS].astype(float)


def _label(p):
    return ((p.won_t > p.t) & (p.won_t <= p.t + H)).astype(int)


# ---------------------------------------------------------------------------------------------- win probability
def train_winprob(panel, end_t):
    """Out-of-time evaluation + final model. Returns (model, cats, report, scored open quotes at end_t)."""
    op = panel[panel.stage.isin(OPEN_STAGES)]
    cats = {c: sorted(op[c].astype(str).unique()) for c in ('product_group', 'team', 'region')}
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                min_samples_leaf=80, l2_regularization=1.0,
                                                categorical_features=[FEATS.index(c) for c in ('pg', 'tm', 'rg')],
                                                random_state=0)
    tr = op[op.t <= end_t - 2 * H]
    te = op[(op.t > end_t - 2 * H) & (op.t <= end_t - H)]
    m0 = mk().fit(_X(tr, cats), _label(tr))
    pt = m0.predict_proba(_X(te, cats))[:, 1]
    y = _label(te).values
    base = te.stage.values / 100.0
    # stage-only empirical baseline (stage % recalibrated on training data) - the fairest simple benchmark
    stage_rate = _label(tr).groupby(tr.stage).mean()
    base_cal = te.stage.map(stage_rate).fillna(stage_rate.mean()).values
    w = te.value.values
    rep = dict(
        train_rows=int(len(tr)), test_rows=int(len(te)),
        test_window=[int(te.t.min()), int(te.t.max())] if len(te) else None,
        auc_model=float(roc_auc_score(y, pt)), auc_stage=float(roc_auc_score(y, base)),
        brier_model=float(brier_score_loss(y, pt)), brier_stage=float(brier_score_loss(y, base)),
        brier_stage_cal=float(brier_score_loss(y, base_cal)),
        value_won=float((w * y).sum()), value_exp_model=float((w * pt).sum()),
        value_exp_stage=float((w * base).sum()), value_exp_stage_cal=float((w * base_cal).sum()),
        win_rate=float(y.mean()),
    )
    # calibration table (deciles of model probability)
    q = pd.qcut(pt, 10, duplicates='drop')
    cal = pd.DataFrame({'p': pt, 'y': y, 'b': base}).groupby(q, observed=True).agg(
        pred=('p', 'mean'), actual=('y', 'mean'), stage=('b', 'mean'), n=('y', 'size'))
    rep['calibration'] = cal.round(4).to_dict('records')
    # stage table: CRM % vs actual 6-month win rate in test window
    stab = pd.DataFrame({'stage': te.stage.values, 'y': y, 'p': pt}).groupby('stage').agg(
        n=('y', 'size'), actual=('y', 'mean'), model=('p', 'mean'))
    rep['by_stage'] = [dict(stage=int(k), **{a: float(b) for a, b in v.items()}) for k, v in stab.iterrows()]
    # permutation importance (cheap version on a sample)
    Xs = _X(te.sample(min(len(te), 6000), random_state=0), cats)
    ys = _label(te.loc[Xs.index]).values
    b0 = roc_auc_score(ys, m0.predict_proba(Xs)[:, 1])
    imp = {}
    rng = np.random.default_rng(0)
    for f in FEATS:
        Xp = Xs.copy()
        Xp[f] = rng.permutation(Xp[f].values)
        imp[f] = float(b0 - roc_auc_score(ys, m0.predict_proba(Xp)[:, 1]))
    rep['importance'] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    # recent-conditions recalibration: value won / value expected on the latest resolvable window
    k = float(np.clip((w * y).sum() / max(1.0, (w * pt).sum()), 0.5, 2.0))
    rep['drift_factor'] = k
    # final model on everything resolvable
    full = op[op.t <= end_t - H]
    m = mk().fit(_X(full, cats), _label(full))
    cur = op[op.t == end_t].copy()
    cur['p_raw'] = m.predict_proba(_X(cur, cats))[:, 1]
    cur['p_win'] = cur.p_raw          # drift_factor is reported as a monitoring signal, not applied
    return m, cats, rep, cur, m0


def timing_table(panel, end_t):
    """P(award in month k = 1..H | won within H) by months-to-EAD bucket (-3..9, clipped)."""
    op = panel[panel.stage.isin(OPEN_STAGES) & (panel.t <= end_t - H)]
    w = op[(op.won_t > op.t) & (op.won_t <= op.t + H)]
    b = w.to_ead_m.clip(-3, 9).astype(int)
    k = (w.won_t - w.t).astype(int)
    tab = pd.crosstab(b, k).reindex(index=range(-3, 10), columns=range(1, H + 1), fill_value=0) + 0.5
    tab = tab.div(tab.sum(axis=1), axis=0)
    return {int(i): [round(float(x), 4) for x in r] for i, r in tab.iterrows()}


def new_business_runrate(panel, snap, end_t, lookback=12):
    """Average award in month t+k (k=1..H) coming from quotations created AFTER snapshot t."""
    won = snap[snap.stage.isin(WON_STAGES)].drop_duplicates('quote_id', keep='last')
    born = snap.groupby('quote_id').snapshot.min()
    wt = panel[panel.won_t.notna()].drop_duplicates('quote_id').set_index('quote_id').won_t
    df = pd.DataFrame({'born': pd.Series(_mi(born.values), index=born.index), 'won_t': wt}).dropna()
    df = df.join(won.set_index('quote_id').value)
    out = np.zeros(H)
    starts = range(end_t - lookback - H, end_t - H + 1)
    for t0 in starts:
        x = df[(df.born > t0)]
        for k in range(1, H + 1):
            out[k - 1] += x[x.won_t == t0 + k].value.sum()
    return [float(v / len(starts)) for v in out]


def _fit(tr, cats):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=80,
                                          l2_regularization=1.0,
                                          categorical_features=[FEATS.index(c) for c in ('pg', 'tm', 'rg')],
                                          random_state=0).fit(_X(tr, cats), _label(tr))


SHOCK = 0.15   # sd of the common (market-wide) multiplicative shock per simulated path


def simulate(V, P, T, nb, sims=3000, seed=0, shock=SHOCK):
    """Monte Carlo award by month: independent wins + a common market shock + new-business run-rate."""
    rng = np.random.default_rng(seed)
    res = np.zeros((sims, H))
    for i in range(sims):
        f = rng.lognormal(-shock ** 2 / 2, shock)
        win = rng.random(len(V)) < np.clip(P * f, 0, 0.99)
        mo = (T[win].cumsum(1) < rng.random(win.sum())[:, None]).sum(1)
        res[i] += np.bincount(mo, weights=V[win], minlength=H)[:H]
        res[i] += np.array(nb) * f * rng.lognormal(-0.02, 0.2, H)
    return res


def backtest_forecast(panel, snap, end_t, t0=None, sims=3000, seed=0, drift=False):
    """Forecast as it would have been made at t0 (default end_t-H), using only data visible then, vs actual."""
    t0 = end_t - H if t0 is None else t0
    op = panel[panel.stage.isin(OPEN_STAGES)]
    cats = {c: sorted(op[c].astype(str).unique()) for c in ('product_group', 'team', 'region')}
    ma = _fit(op[op.t <= t0 - 2 * H], cats)
    win_ = op[(op.t > t0 - 2 * H) & (op.t <= t0 - H)]
    pa = ma.predict_proba(_X(win_, cats))[:, 1]
    k = float(np.clip((win_.value * _label(win_)).sum() / max(1.0, (win_.value * pa).sum()), 0.5, 2.0)) if drift else 1.0
    m = _fit(op[op.t <= t0 - H], cats)
    cur = op[op.t == t0].copy()
    cur['p'] = np.clip(m.predict_proba(_X(cur, cats))[:, 1] * k, 0, 0.99)
    vis = panel[panel.t <= t0].copy()
    vis.loc[vis.won_t > t0, 'won_t'] = np.nan
    tim = timing_table(vis, t0)
    nb = new_business_runrate(vis, snap[_mi(snap.snapshot) <= t0], t0)
    V, P = cur.value.values, cur.p.values
    T = np.array([tim[b] for b in cur.to_ead_m.clip(-3, 9).astype(int).values])
    res = simulate(V, P, T, nb, sims, seed)
    naive = np.zeros(H)
    kk = np.clip(cur.to_ead_m.values, 1, H).astype(int) - 1
    np.add.at(naive, kk, cur.value.values * cur.stage.values / 100)
    won = panel[panel.won_t.notna()].drop_duplicates('quote_id')
    act = [float(won[won.won_t == t0 + j].value.sum()) for j in range(1, H + 1)]
    tot = res.sum(1)
    return dict(start_t=int(t0), drift=k, actual=act, p10=np.percentile(res, 10, 0).tolist(),
                p50=np.percentile(res, 50, 0).tolist(), p90=np.percentile(res, 90, 0).tolist(),
                naive_stage=naive.tolist(), new_business=nb,
                total=dict(actual=float(sum(act)), p10=float(np.percentile(tot, 10)),
                           p50=float(np.percentile(tot, 50)), p90=float(np.percentile(tot, 90)),
                           naive=float(naive.sum())))


# ----------------------------------------------------------------------------------------------------- churn
def churn_model(snap, end_t, invoices=None):
    """Account panel: active accounts (award in prior 12m) at month t; label = no quote & no award in t+1..t+6."""
    s = snap[['snapshot', 'quote_id', 'customer_id', 'value', 'status']].copy()
    s['t'] = _mi(s.snapshot)
    born = s.groupby('quote_id').agg(born=('t', 'min'), cid=('customer_id', 'first'), val=('value', 'first'))
    wonq = s[s.status == 'WON'].groupby('quote_id').t.min()
    born['won_t'] = wonq
    T0, T1 = int(s.t.min()) + 12, end_t
    cids = born.cid.unique()
    idx = {c: i for i, c in enumerate(cids)}
    nT = T1 + H + 1
    Q = np.zeros((len(cids), nT))
    A = np.zeros((len(cids), nT))
    for r in born.itertuples():
        if r.born < nT:
            Q[idx[r.cid], int(r.born)] += 1
        if not np.isnan(r.won_t) and r.won_t < nT:
            A[idx[r.cid], int(r.won_t)] += r.val
    late = None
    if invoices is not None and len(invoices):
        iv = invoices[invoices.customer_id.isin(idx)].copy()
        iv['t'] = _mi(iv.invoice_date)
        iv['due'] = pd.to_datetime(iv.due_date)
        iv['paid'] = pd.to_datetime(iv.paid_date, errors='coerce')
        ex = iv.loc[iv.index.repeat(6)].copy()
        ex['tt'] = ex.t + np.tile(np.arange(6), len(iv))
        ex = ex[ex.tt < nT]
        cut = pd.to_datetime([f'{x // 12}-{x % 12 + 1:02d}-01' for x in ex.tt]) + pd.offsets.MonthEnd(0)
        cut = pd.Series(cut, index=ex.index)
        vis = ex.paid.notna() & (ex.paid <= cut)
        ex['lat'] = np.where(vis, (ex.paid - ex.due).dt.days, (cut - ex.due).dt.days)
        g = ex.groupby(['customer_id', 'tt']).lat.mean()
        late = np.full((len(cids), nT), np.nan)
        for (c, t), v in g.items():
            late[idx[c], t] = v
    rows = []
    for t in range(T0, T1 + 1):
        a12 = A[:, t - 11:t + 1].sum(1)
        act = a12 > 0
        for i in np.where(act)[0]:
            qa = Q[i, :t + 1]
            aa = A[i, :t + 1]
            lq = np.nonzero(qa)[0]
            la = np.nonzero(aa)[0]
            fut = Q[i, t + 1:t + 1 + H].sum() + (A[i, t + 1:t + 1 + H] > 0).sum()
            rows.append(dict(cid=cids[i], t=t,
                             m_since_quote=t - lq[-1] if len(lq) else 99, m_since_award=t - la[-1] if len(la) else 99,
                             q3=Q[i, t - 2:t + 1].sum(), q3p=Q[i, t - 5:t - 2].sum(), q12=Q[i, t - 11:t + 1].sum(),
                             a6=A[i, t - 5:t + 1].sum(), a6p=A[i, t - 11:t - 5].sum(), a12=a12[i],
                             late=late[i, t] if late is not None else np.nan,
                             y=int(fut == 0) if t + H <= T1 else -1))
    P = pd.DataFrame(rows)
    P['la12'] = np.log1p(P.a12)
    P['trend_q'] = (P.q3 + 0.5) / (P.q3p + 0.5)
    P['trend_a'] = np.log1p(P.a6) - np.log1p(P.a6p)
    F = ['m_since_quote', 'm_since_award', 'q3', 'q3p', 'q12', 'la12', 'trend_q', 'trend_a', 'late']
    if P.late.isna().all():
        F.remove('late')
    lab = P[P.y >= 0]
    tr, te = lab[lab.t <= T1 - 2 * H], lab[(lab.t > T1 - 2 * H) & (lab.t <= T1 - H)]
    mk = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, min_samples_leaf=40,
                                                random_state=0)
    m0 = mk().fit(tr[F], tr.y)
    pt = m0.predict_proba(te[F])[:, 1]
    base = (te.m_since_quote / 12).clip(0, 1)            # naive rule: longer silence = higher risk
    rep = dict(auc_model=float(roc_auc_score(te.y, pt)), auc_rule=float(roc_auc_score(te.y, base)),
               base_rate=float(te.y.mean()), n_test=int(len(te)))
    top = te.assign(p=pt).sort_values('p', ascending=False)
    k = max(1, int(len(top) * 0.2))
    rep['top20_capture'] = float(top.y.iloc[:k].sum() / max(1, top.y.sum()))
    m = mk().fit(lab[lab.t <= T1 - H][F], lab[lab.t <= T1 - H].y)
    cur = P[P.t == T1].copy()
    cur['p_churn'] = m.predict_proba(cur[F])[:, 1]
    return rep, cur


# -------------------------------------------------------------------------------------------- next best action
def next_best_actions(cur_q, churn_cur, ar_open, owners_now, max_per_se=8):
    """Ranked action list per SE. cur_q: scored open quotations; churn_cur: scored accounts;
    ar_open: overdue AR by customer (customer_id, overdue); owners_now: customer_id -> se."""
    acts = []
    q = cur_q.copy()
    q['exp'] = q.value * q.p_win
    # 1. close: high probability, award date within 2 months
    thr = max(0.2, min(0.5, float(q.p_win.quantile(0.97)))) if len(q) else 0.5
    for r in q[(q.p_win >= thr) & (q.to_ead_m <= 2)].itertuples():
        when = (f'award date {r.expected_award[:7]}' if r.to_ead_m >= 0
                else f'award date {r.expected_award[:7]} has passed - confirm PO timing')
        acts.append((r.se, 'Close', r.exp, f'Close {r.quote_id} ({r.customer}): {r.p_win:.0%} likely, {when}'))
    # 2. re-qualify: CRM stage much more optimistic than the model on large deals
    gap = q[(q.stage >= 50) & (q.stage / 100 - q.p_win >= 0.3) & (q.value >= q.value.quantile(0.75))]
    for r in gap.itertuples():
        acts.append((r.se, 'Re-qualify', r.value * (r.stage / 100 - r.p_win),
                     f'Re-qualify {r.quote_id} ({r.customer}): stage {r.stage}% but model {r.p_win:.0%} '
                     f'({int(r.ead_slips)} date slips)'))
    # 3. retain: valuable active accounts at risk of going quiet
    if churn_cur is not None and len(churn_cur):
        cthr = max(0.5, float(churn_cur.p_churn.quantile(0.8)))
        c = churn_cur[(churn_cur.p_churn >= cthr) & (churn_cur.a12 >= churn_cur.a12.median())].copy()
        c['se'] = c.cid.map(owners_now)
        for r in c.itertuples():
            acts.append((r.se, 'Retain', r.a12 * r.p_churn,
                         f'Visit {r.name_}: {r.p_churn:.0%} risk of going quiet, '
                         f'{int(r.m_since_quote)} months since last quotation'))
    # 4. collect: overdue AR from customers with live pipeline (leverage before the next award)
    if ar_open is not None and len(ar_open):
        live = q.groupby('customer_id').value.sum()
        a = ar_open[ar_open.overdue > 0].copy()
        a['live'] = a.customer_id.map(live).fillna(0)
        for r in a[a.live > 0].itertuples():
            acts.append((owners_now.get(r.customer_id, ''), 'Collect', r.overdue,
                         f'Collect SAR {r.overdue:,.0f} overdue from {r.customer} before next award '
                         f'(SAR {r.live:,.0f} open pipeline)'))
    df = pd.DataFrame(acts, columns=['se', 'type', 'impact', 'text'])
    df = df[df.se.astype(str) != '']
    df = df.sort_values('impact', ascending=False).groupby('se').head(max_per_se)
    return df
