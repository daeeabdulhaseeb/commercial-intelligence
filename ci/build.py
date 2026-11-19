"""Build the dashboard bundle and render site/index.html.

  python -m ci.build                       # stand-alone synthetic demo of this layer -> docs/model/index.html
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


def stress_inputs(scored, tim, panel, snap, as_of):
    """Per-quotation inputs for the in-browser Monte Carlo, with team / product-group tags for segment levers."""
    end_t = int(Mo._mi([as_of])[0])
    sc = scored.copy()
    sc['exp'] = sc.value * sc.p_win
    sc = sc.sort_values('exp', ascending=False)
    vt = sc.groupby(sc.team.astype(str)).value.sum()
    teams = sorted(t for t in vt.index if t and t not in ('0', '0.0', 'nan') and vt[t] >= 0.002 * vt.sum())
    groups = sorted(g for g in sc.product_group.astype(str).unique() if g)
    ti = {t: i for i, t in enumerate(teams)}
    gi = {g: i for i, g in enumerate(groups)}
    cust_rank = sc.groupby('customer_id').value.sum().rank(ascending=False, method='first').astype(int)
    keep, rest = sc.head(MAX_Q), sc.iloc[MAX_Q:]
    rest_t = {t: np.zeros(Mo.H) for t in teams}
    for r in rest.itertuples():
        if str(r.team) in rest_t:
            rest_t[str(r.team)] += r.value * r.p_win * np.array(tim[int(np.clip(r.to_ead_m, -3, 9))])
    q = [[round(float(r.value), 0), round(float(r.p_win), 4), int(np.clip(r.to_ead_m, -3, 9)),
          int(cust_rank[r.customer_id]), ti.get(str(r.team), -1), gi.get(str(r.product_group), -1)]
         for r in keep.itertuples()]
    nb_t = {}
    for t in teams:
        nb_t[t] = [round(x) for x in Mo.new_business_runrate(panel[panel.team == t], snap[snap.team == t], end_t)]
    return dict(q=q, timing=tim, teams=teams, groups=groups, nb_team=nb_t,
                rest_team={t: [round(x) for x in v] for t, v in rest_t.items()},
                months=[A.ym_add(as_of, i) for i in range(1, Mo.H + 1)])


def award_by_month(panel):
    w = panel[panel.won_t.notna()].drop_duplicates('quote_id')
    am = w.groupby(w.won_t.astype(int)).value.sum()
    am.index = [f'{t // 12}-{t % 12 + 1:02d}' for t in am.index]
    return am


def synthetic_cash(inv, ar_open, cust_delay):
    """Aggregate invoice-level demo data into the same cash inputs a company with aggregate extracts provides."""
    iv = inv.copy()
    invoice_m = iv.groupby(iv.invoice_date.str[:7]).amount.sum()
    paid = iv[iv.paid_date.fillna('').astype(str).str.match(r'\d{4}-')]
    collect_m = paid.groupby(paid.paid_date.str[:7]).amount.sum()
    o = ar_open.copy()
    b = pd.cut(o.od, [-10 ** 6, 0, 30, 60, 365, 10 ** 6], labels=['not_due', 'ovd_0_30', 'ovd_30_60', 'ovd_60_365',
                                                                   'ovd_365p'])
    ar = o.assign(b=b).pivot_table(index=['customer_id', 'customer'], columns='b', values='amount', aggfunc='sum',
                                   fill_value=0, observed=False).reset_index()
    for c in ['not_due', 'ovd_0_30', 'ovd_30_60', 'ovd_60_365', 'ovd_365p']:
        if c not in ar:
            ar[c] = 0.0
    terms = inv.groupby('customer_id').credit_days.first()
    ar['terms_days'] = ar.customer_id.map(terms).fillna(30)
    ar['late_days'] = ar.customer_id.map(cust_delay).fillna(15)
    ar['team'] = ''
    return invoice_m, collect_m, ar


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
    if targets is not None and ses is not None and len(targets) and 'team' not in targets:
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
    gone = set(data.get('exclude_ses') or [])           # SEs who left: kept in team totals, dropped from SE lists
    if gone:
        nba = nba[~nba.se.isin(gone)]
        se['rows'] = [r for r in se['rows'] if r['se'] not in gone]
    stress = stress_inputs(scored, tim, panel, snap, as_of)
    from . import cash as C
    cash_d = None
    if data.get('cash'):                                   # company aggregate extracts (private adapter)
        cd = data['cash']
        cash_d = C.build(award_by_month(panel), cd['invoice_m'], cd['collect_m'], cd['ar'], as_of,
                         invoice_all=cd.get('invoice_all'))
    elif inv is not None and len(inv):
        im, cm, arx = synthetic_cash(inv, ar_open, cust_delay)
        cash_d = C.build(award_by_month(panel), im, cm, arx, as_of)
    if cash_d:
        stress.update(cash_d)
    # target gap on the sales engineers that carry a target (award YTD by expected award date)
    L = snap[snap.snapshot == as_of]
    Y = as_of[:4]
    aw = L[L.stage.isin([90, 100]) & (L.expected_award.str[:7] >= f'{Y}-01') & (L.expected_award.str[:7] <= as_of)]
    tg_rows = []
    remaining_year, tgt_total, aw_t = 0.0, 0.0, 0.0
    if targets is not None and len(targets):
        a_se = aw.groupby('se').value.sum()
        tt = targets.copy()
        tt['award_ytd'] = tt.se.map(a_se).fillna(0)
        tgt_total, aw_t = float(tt.award_target.sum()), float(tt.award_ytd.sum())
        remaining_year = max(0.0, tgt_total - aw_t)
        for team, g in tt.groupby(tt.team.fillna('')):
            tg_rows.append(dict(team=team, target=float(g.award_target.sum()), award_ytd=float(g.award_ytd.sum()),
                                remaining=max(0.0, float(g.award_target.sum() - g.award_ytd.sum())),
                                ses=int(len(g))))
    presets = data.get('presets')
    bundle = dict(
        meta=dict(label=label, company=data['meta'].get('company', ''), as_of=as_of, public=public,
                  synthetic=bool(data['meta'].get('synthetic', False)),
                  built=time.strftime('%Y-%m-%d %H:%M')),
        pipeline=pipe, leads=ld, accounts=acc, se=se, receivables=ar,
        predictive=dict(win=rep, backtests=bts, churn=churn_rep, horizon=Mo.H,
                        open_scored=int(len(scored)), expected_6m=float((scored.value * scored.p_win).sum()),
                        nba=[dict(se=r.se, type=r.type, impact=round(float(r.impact)), text=r.text)
                             for r in nba.itertuples()]),
        stress=dict(**stress, remaining_year=remaining_year, target=tgt_total, award_ytd_targeted=aw_t,
                    targets_by_team=tg_rows, target_ses=int(len(targets)) if targets is not None else 0,
                    all_ses=int(L.se.nunique()), months_left_in_year=12 - pipe['elapsed'], presets=presets),
    )
    os.makedirs(out_dir, exist_ok=True)
    if not public:   # machine-readable scores for the weekly pipeline report and the account platform
        pq = {str(r.quote_id).lstrip('0'): round(float(r.p_win), 4) for r in scored.itertuples()}
        ch = {str(r.cid): dict(p=round(float(r.p_churn), 3), a12=round(float(r.a12)), quiet=int(r.m_since_quote))
              for r in churn_cur.itertuples()}
        cthr = max(0.5, float(churn_cur.p_churn.quantile(0.8))) if len(churn_cur) else 0.5
        with open(os.path.join(out_dir, 'predictive.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(as_of=as_of, built=bundle['meta']['built'], horizon=Mo.H,
                           auc=round(rep['auc_model'], 3), auc_stage=round(rep['auc_stage'], 3),
                           backtests=[dict(start=b['start'], actual=round(b['total']['actual']), p50=round(b['total']['p50']),
                                           stage=round(b['total']['naive'])) for b in bts],
                           churn_threshold=round(cthr, 3), quotes=pq, churn=ch,
                           nb=round(float(sum(nb))), nb_team={t: round(float(sum(v))) for t, v in stress.get('nb_team', {}).items()},
                           nba=[dict(se=r.se, type=r.type, impact=round(float(r.impact)), text=r.text)
                                for r in nba.itertuples()]), f, separators=(',', ':'))
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
    for fi, sdef in enumerate(srcs):
        p = sdef['path'] if os.path.isabs(sdef['path']) else os.path.join(ROOT, sdef['path'])
        lab = sdef.get('snapshot')
        if lab == 'mtime':
            lab = time.strftime('%Y-%m', time.localtime(os.path.getmtime(p)))
            if cfg.get('mtime_is_month_end_of_previous', True):
                lab = A.ym_add(lab, -1)
        scol = cfg.get('colmap', {}).get('snapshot', 'snapshot')
        if p.endswith('.parquet'):
            raw = pd.read_parquet(p)
            if lab:
                raw[scol] = lab
            parts.append(raw)
        elif cfg.get('canonical'):
            raw = pd.read_csv(p, dtype=str, low_memory=False, usecols=(lambda c: c in want) if want else None)
            if lab:
                raw[scol] = lab
            parts.append(raw)
        else:   # large extracts: read in chunks, filter and collapse the item lines chunk by chunk (memory-safe)
            for ch in pd.read_csv(p, dtype=str, usecols=(lambda c: c in want) if want else None, chunksize=300_000):
                if lab:
                    ch[scol] = lab
                parts.append(adapters.sap_pipeline_snapshots(ch, colmap=cfg.get('colmap'), valid_value=cfg.get('valid_value'),
                                                             groups=cfg.get('groups')).assign(_f=fi))
    raw = pd.concat(parts, ignore_index=True)
    if not cfg.get('canonical') and not any(str(x.get('path', '')).endswith('.parquet') for x in srcs):
        # a quotation's lines can straddle two chunks: sum again; later files win for the same month
        val = raw.groupby(['_f', 'snapshot', 'quote_id']).value.sum()
        raw = raw.drop_duplicates(['_f', 'snapshot', 'quote_id'], keep='first').set_index(['_f', 'snapshot', 'quote_id'])
        raw['value'] = val
        raw = raw.reset_index().sort_values('_f').drop_duplicates(['snapshot', 'quote_id'], keep='last')
        snap = raw.drop(columns='_f').reset_index(drop=True)
    elif cfg.get('canonical'):
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
    if cfg.get('cash_adapter'):        # private module returning (invoice_m, invoice_all, collect_m, ar, label)
        import importlib.util
        ca = cfg['cash_adapter']
        spec = importlib.util.spec_from_file_location('cash_adapter', ca['module'] if os.path.isabs(ca['module'])
                                                      else os.path.join(ROOT, ca['module']))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rp = lambda x: x if os.path.isabs(x) else os.path.join(ROOT, x)
        im, ia, cm, ar, lab = mod.load(rp(ca['ar']), rp(ca['collections']), rp(ca['sales']))
        d['cash'] = dict(invoice_m=im, invoice_all=ia, collect_m=cm, ar=ar)
        d['meta']['ar_snapshot'] = lab
    if cfg.get('presets'):
        d['presets'] = cfg['presets']
    if cfg.get('exclude_ses'):
        d['exclude_ses'] = cfg['exclude_ses']
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
        run(synth.load(src), os.path.join(ROOT, 'docs', 'model'), public=True, label='demo')   # docs/index.html is the suite menu
