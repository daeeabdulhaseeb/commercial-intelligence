"""Build the dashboard bundle and render site/index.html.

  python -m ci.build                       # synthetic demo  -> docs/index.html (public, GitHub Pages)
  python -m ci.build --private CONFIG.json # company data    -> private/site/index.html (never committed)

CONFIG.json (private): see private.example.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import analytics as A
from . import models as Mo
from . import synth

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
MAX_Q = 2500


def stress_inputs(scored, tim, nb, invoices, inv_open, panel, as_of, cust_delay):
    sc = scored.copy()
    sc['exp'] = sc.value * sc.p_win
    sc = sc.sort_values('exp', ascending=False)
    cust_rank = sc.groupby('customer_id').value.sum().rank(ascending=False, method='first').astype(int)
    keep, rest = sc.head(MAX_Q), sc.iloc[MAX_Q:]
    rest_m = np.zeros(Mo.H)
    for r in rest.itertuples():
        rest_m += r.value * r.p_win * np.array(tim[int(np.clip(r.to_ead_m, -3, 9))])
    q = [[round(float(r.value), 0), round(float(r.p_win), 4), int(np.clip(r.to_ead_m, -3, 9)),
          int(cust_rank[r.customer_id])] for r in keep.itertuples()]
    out = dict(q=q, timing=tim, nb=[round(x) for x in nb], rest=[round(x) for x in rest_m],
               months=[A.ym_add(as_of, i) for i in range(1, Mo.H + 1)])
    # award -> invoice lag (share of awarded value invoiced k months after the award month)
    if invoices is not None and len(invoices):
        wt = panel[panel.won_t.notna()].drop_duplicates('quote_id').set_index('quote_id')[['won_t', 'value']]
        iv = invoices.join(wt, on='quote_id', how='inner')
        iv['lag'] = (Mo._mi(iv.invoice_date) - iv.won_t).clip(0, 8)
        lag = iv.groupby('lag').amount.sum() / wt.loc[wt.index.isin(iv.quote_id)].value.sum()
        out['inv_lag'] = [round(float(lag.get(k, 0)), 4) for k in range(9)]
        end_t = int(Mo._mi([as_of])[0])
        bl = np.zeros(Mo.H)
        recent = wt[(wt.won_t > end_t - 9) & (wt.won_t <= end_t)]
        for wtt, v in zip(recent.won_t, recent.value):
            for k in range(1, Mo.H + 1):
                lg = int(end_t + k - wtt)
                if lg <= 8:
                    bl[k - 1] += v * out['inv_lag'][lg]
        out['backlog_inv'] = [round(x) for x in bl]
        paid = invoices[invoices.paid_date.fillna('').astype(str).str.match(r'\d{4}-')]
        cl = (Mo._mi(paid.paid_date) - Mo._mi(paid.invoice_date)).clip(0, 9)
        cdist = paid.amount.groupby(cl).sum() / invoices.amount.sum()
        out['col_lag'] = [round(float(cdist.get(k, 0)), 4) for k in range(10)]
        cut = pd.Period(as_of, 'M').to_timestamp(how='end').normalize()
        o = inv_open.copy()
        o['exp_days'] = ((o.due + pd.to_timedelta(o.customer_id.map(cust_delay).fillna(15), unit='D')) - cut).dt.days
        items = []
        for a, b in zip(o.amount, o.exp_days):
            if b < 15:      # already late: spread over the next three months instead of assuming instant cash
                items += [[round(a * 0.5), 15], [round(a * 0.3), 45], [round(a * 0.2), 75]]
            else:
                items.append([round(float(a)), int(b)])
        out['ar_items'] = items
    return out


def run(data, out_dir, public=True, label='demo'):
    t0 = time.time()
    snap = data['snapshots']
    as_of = str(snap.snapshot.max())
    end_t = int(Mo._mi([as_of])[0])
    panel = Mo.build_panel(snap)
    print('panel', panel.shape, round(time.time() - t0, 1))
    model, cats, rep, scored, _ = Mo.train_winprob(panel, end_t)
    scored = scored.join(snap[snap.snapshot == as_of].set_index('quote_id')[['customer']], on='quote_id')
    tim = Mo.timing_table(panel, end_t)
    nb = Mo.new_business_runrate(panel, snap, end_t)
    bts = []
    for k in (Mo.H, Mo.H + 2, Mo.H + 4):
        try:
            b = Mo.backtest_forecast(panel, snap, end_t, t0=end_t - k, sims=2000)
            b['start'] = A.ym_add(as_of, -k)
            bts.append(b)
        except Exception as e:                                     # not enough history
            print('backtest skipped', k, e)
    print('models', round(time.time() - t0, 1))
    inv = data.get('invoices')
    churn_rep, churn_cur = Mo.churn_model(snap, end_t, inv)
    targets = data.get('targets')
    ses = data.get('ses')
    if targets is not None and ses is not None and len(targets):
        targets = targets.merge(ses[['se', 'team']], on='se', how='left')
    pipe = A.pipeline(snap, as_of, targets, scored)
    acc, accdf = A.accounts(snap, as_of, data.get('owners'), data.get('customers'), churn_cur)
    se = A.se_scorecard(snap, as_of, targets, inv, ses)
    ar, ar_open = A.receivables(inv, as_of)
    ld = A.leads(data.get('leads'), snap, as_of)
    # next best actions
    owners_now = accdf.owner.to_dict()
    ar_c = None
    cust_delay = {}
    if ar_open is not None:
        ar_c = ar_open[ar_open.od > 0].groupby(['customer_id', 'customer']).amount.sum().rename('overdue').reset_index()
        pdf = inv[inv.paid_date.fillna('').astype(str).str.match(r'\d{4}-')]
        cust_delay = ((pd.to_datetime(pdf.paid_date) - pd.to_datetime(pdf.due_date)).dt.days
                      .groupby(pdf.customer_id).mean().to_dict())
    cc = churn_cur.copy()
    cc['name_'] = cc.cid.map(accdf.name)
    nba = Mo.next_best_actions(scored, cc, ar_c, owners_now)
    stress = stress_inputs(scored, tim, nb, inv, ar_open, panel, as_of, cust_delay)
    remaining_year = max(0.0, pipe['target'] - pipe['award_ytd'])
    bundle = dict(
        meta=dict(label=label, company=data['meta'].get('company', ''), as_of=as_of, public=public,
                  synthetic=bool(data['meta'].get('synthetic', False)),
                  built=time.strftime('%Y-%m-%d %H:%M')),
        pipeline=pipe, leads=ld, accounts=acc, se=se, receivables=ar,
        predictive=dict(win=rep, backtests=bts, churn=churn_rep, horizon=Mo.H,
                        open_scored=int(len(scored)), expected_6m=float((scored.value * scored.p_win).sum()),
                        nba=[dict(se=r.se, type=r.type, impact=round(float(r.impact)), text=r.text)
                             for r in nba.itertuples()]),
        stress=dict(**stress, remaining_year=remaining_year, target=pipe['target'],
                    months_left_in_year=12 - pipe['elapsed']),
    )
    os.makedirs(out_dir, exist_ok=True)
    tpl = open(os.path.join(ROOT, 'ci', 'template.html'), encoding='utf-8').read()
    js = json.dumps(bundle, separators=(',', ':'), default=lambda o: None if (isinstance(o, float) and np.isnan(o))
                    else (o.item() if hasattr(o, 'item') else str(o)))
    js = js.replace('NaN', 'null').replace('</', '<\\/')
    html = tpl.replace('/*__DATA__*/null', js)
    vend = os.path.join(ROOT, 'ci', 'vendor', 'chart.umd.min.js')
    if os.path.exists(vend):                      # inline Chart.js (MIT) so the page works offline
        tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>'
        html = html.replace(tag, '<script>' + open(vend, encoding='utf-8').read().replace('</script', '<\\/script') + '</script>')
    with open(os.path.join(out_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(dict(meta=bundle['meta'], win={k: v for k, v in rep.items() if not isinstance(v, (list, dict))},
                       churn=churn_rep, backtests=[b['total'] | {'start': b['start']} for b in bts]), f, indent=1)
    print('built', os.path.join(out_dir, 'index.html'), round(len(html) / 1e6, 2), 'MB', round(time.time() - t0, 1), 's')
    return bundle


def load_private(cfg):
    """cfg['snapshots']: a file path or a list of {"path": ..., "snapshot": "YYYY-MM" | "mtime" | null}.
    A file without a snapshot column gets the given month (or the month of its modified date for "mtime").
    cfg['colmap'] maps canonical names to the extract's column names; cfg['valid_value'], cfg['groups'] filter."""
    from . import adapters
    srcs = cfg['snapshots'] if isinstance(cfg['snapshots'], list) else [{'path': cfg['snapshots']}]
    parts = []
    cm = cfg.get('colmap', {})
    want = set(cm.values()) if cm else None
    for sdef in srcs:
        p = sdef['path'] if os.path.isabs(sdef['path']) else os.path.join(ROOT, sdef['path'])
        if p.endswith('.parquet'):
            raw = pd.read_parquet(p)
        else:
            raw = pd.read_csv(p, dtype=str, low_memory=False, usecols=(lambda c: c in want) if want else None)
        lab = sdef.get('snapshot')
        if lab == 'mtime':
            lab = time.strftime('%Y-%m', time.localtime(os.path.getmtime(p)))
            if cfg.get('mtime_is_month_end_of_previous', True):
                lab = A.ym_add(lab, -1)
        if lab:
            raw[cfg.get('colmap', {}).get('snapshot', 'snapshot')] = lab
        parts.append(raw)
    raw = pd.concat(parts, ignore_index=True)
    if cfg.get('canonical'):
        snap = raw
        snap['value'] = pd.to_numeric(snap.value)
        snap['stage'] = pd.to_numeric(snap.stage).astype(int)
    else:
        snap = adapters.sap_pipeline_snapshots(raw, colmap=cfg.get('colmap'), valid_value=cfg.get('valid_value'),
                                               groups=cfg.get('groups'))
        if len(srcs) > 1:   # later files win for the same month
            snap = snap.drop_duplicates(['snapshot', 'quote_id'], keep='last')
    d = dict(snapshots=snap, meta=dict(company=cfg.get('company', 'Company'), synthetic=False))
    for k in ('targets', 'ses', 'invoices', 'leads', 'owners'):
        if cfg.get(k):
            d[k] = pd.read_csv(cfg[k] if os.path.isabs(cfg[k]) else os.path.join(ROOT, cfg[k]), dtype={'quote_id': str, 'customer_id': str}, keep_default_na=False)
    if 'targets' in d:
        d['targets']['award_target'] = pd.to_numeric(d['targets'].award_target)
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--private', help='JSON config for company data (output stays in private/)')
    a = ap.parse_args()
    if a.private:
        cfg = json.load(open(a.private))
        out = cfg.get('out', 'private/site')
        run(load_private(cfg), out if os.path.isabs(out) else os.path.join(ROOT, out), public=False, label='private')
    else:
        src = os.path.join(ROOT, 'data', 'synthetic')
        if not os.path.exists(os.path.join(src, 'snapshots.csv.gz')):
            synth.save(synth.build(), src)
        run(synth.load(src), os.path.join(ROOT, 'docs'), public=True, label='demo')
