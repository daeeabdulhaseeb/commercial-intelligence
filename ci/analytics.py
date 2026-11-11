"""Descriptive KPIs for each dashboard tab (pipeline, leads, accounts, SE performance, receivables).

Business rules
  * Award            = quotations at 90% / 100%, dated by expected award date (EAD).
  * Hit ratio        = award value (EAD in window) / value of quotations raised in the window
                       (stages 0,10,30,50,70,90,100 - tender/budget 15% quotations excluded), any status incl. lost.
  * Hot pipeline     = 50% / 70% quotations.
  * Needed quotations per month = annual award target / hit ratio / 12 (SE hit ratio bounded to 0.5-2x company).
  * SE scorecard     = quotation 20% + hot-project-to-award 30% + award 20% + invoicing 30%.
"""
import numpy as np
import pandas as pd

from .schema import OPEN_STAGES, HOT_STAGES, WON_STAGES, HIT_STAGES


def ym_add(ym, k):
    return (pd.Period(ym, 'M') + k).strftime('%Y-%m')


def _r(x, n=0):
    return float(round(float(x), n))


def pipeline(snap, as_of, targets=None, scored=None):
    L = snap[snap.snapshot == as_of].copy()
    L['ead_m'] = L.expected_award.str[:7]
    L['vf_m'] = L.valid_from.str[:7]
    Y = as_of[:4]
    elapsed = int(as_of[5:7])
    won = L[L.stage.isin(WON_STAGES)]
    w12 = [ym_add(as_of, -i) for i in range(12)]
    award_ytd = won[(won.ead_m >= f'{Y}-01') & (won.ead_m <= as_of)]
    quo = L[L.stage.isin(HIT_STAGES)]
    hr12 = won[won.ead_m.isin(w12)].value.sum() / max(1.0, quo[quo.vf_m.isin(w12)].value.sum())
    tgt = float(targets.award_target.sum()) if targets is not None and len(targets) else 0.0
    op = L[L.stage.isin(OPEN_STAGES)]
    nxt3 = [ym_add(as_of, i) for i in range(1, 4)]
    hot = op[op.stage.isin(HOT_STAGES)]
    stale = op[op.ead_m < as_of]
    months_ly = [ym_add(as_of, -i) for i in range(23, -1, -1)]
    by_m_award = won.groupby('ead_m').value.sum()
    by_m_quo = quo.groupby('vf_m').value.sum()
    by_stage = op.groupby('stage').value.agg(['sum', 'count'])
    teams = []
    tg_team = {}
    if targets is not None and 'team' in targets:
        tg_team = targets.groupby('team').award_target.sum().to_dict()
    for tm, g in L.groupby('team'):
        gw = g[g.stage.isin(WON_STAGES)]
        gq = g[g.stage.isin(HIT_STAGES)]
        a = gw[(gw.ead_m >= f'{Y}-01') & (gw.ead_m <= as_of)].value.sum()
        hr = gw[gw.ead_m.isin(w12)].value.sum() / max(1.0, gq[gq.vf_m.isin(w12)].value.sum())
        go = g[g.stage.isin(OPEN_STAGES)]
        teams.append(dict(team=tm, award_ytd=_r(a), target=_r(tg_team.get(tm, 0)), hit_ratio=_r(hr, 4),
                          open=_r(go.value.sum()), hot=_r(go[go.stage.isin(HOT_STAGES)].value.sum()),
                          stale=_r(go[go.ead_m < as_of].value.sum())))
    out = dict(
        as_of=as_of, year=Y, elapsed=elapsed,
        award_ytd=_r(award_ytd.value.sum()), target=_r(tgt), pace=_r(tgt * elapsed / 12),
        quotes_ytd=_r(quo[(quo.vf_m >= f'{Y}-01') & (quo.vf_m <= as_of)].value.sum()),
        quotes_ytd_n=int(((quo.vf_m >= f'{Y}-01') & (quo.vf_m <= as_of)).sum()),
        hit_ratio_12m=_r(hr12, 4),
        needed_quotes_month=_r(tgt / max(hr12, 1e-6) / 12) if tgt else 0.0,
        open_value=_r(op.value.sum()), open_n=int(len(op)),
        hot_value=_r(hot.value.sum()), hot_next3=_r(hot[hot.ead_m.isin(nxt3)].value.sum()),
        stale_value=_r(stale.value.sum()), stale_n=int(len(stale)),
        stage_weighted=_r((op.value * op.stage / 100).sum()),
        by_stage=[dict(stage=int(k), value=_r(v['sum']), n=int(v['count'])) for k, v in by_stage.iterrows()],
        months=months_ly,
        award_by_month=[_r(by_m_award.get(m, 0)) for m in months_ly],
        quotes_by_month=[_r(by_m_quo.get(m, 0)) for m in months_ly],
        teams=sorted(teams, key=lambda x: -x['award_ytd']),
    )
    if scored is not None and len(scored):
        sc = scored.copy()
        sc['exp'] = sc.value * sc.p_win
        top = sc.sort_values('exp', ascending=False).head(25)
        out['model_expected_6m'] = _r(sc.exp.sum())
        out['top_deals'] = [dict(quote=r.quote_id, customer=r.customer, se=r.se, stage=int(r.stage),
                                 value=_r(r.value), p=_r(r.p_win, 3), ead=r.expected_award[:7],
                                 slips=int(r.ead_slips)) for r in top.itertuples()]
    return out


def leads(ld, snap, as_of):
    if ld is None or not len(ld):
        return None
    d = ld.copy()
    d['m'] = d.created.str[:7]
    d['routed'] = d.se.astype(str) != ''
    d['quoted'] = d.quote_id.astype(str) != ''
    d['won'] = d.status == 'Won'
    d['leaked'] = d.status == 'Leaked to competitor'
    last = snap[snap.snapshot == as_of].set_index('quote_id')
    d['won_value'] = np.where(d.won, d.quote_id.map(last.value).fillna(0), 0.0)
    ch = []
    for c, g in d.groupby('channel'):
        ch.append(dict(channel=c, leads=int(len(g)), routed=_r(g.routed.mean(), 3), quoted=int(g.quoted.sum()),
                       won=int(g.won.sum()), leaked=int(g.leaked.sum()), conv_quote=_r(g.quoted.mean(), 3),
                       conv_win=_r(g.won.sum() / max(1, g.quoted.sum()), 3), won_value=_r(g.won_value.sum())))
    temp = d.groupby('temperature').agg(n=('lead_id', 'size'), q=('quoted', 'mean'))
    months = [ym_add(as_of, -i) for i in range(11, -1, -1)]
    trend = {c: [int(((d.channel == c) & (d.m == m)).sum()) for m in months] for c in sorted(d.channel.unique())}
    recent = d[d.m >= ym_add(as_of, -2)]
    return dict(by_channel=ch, months=months, trend=trend,
                by_temp=[dict(temp=k, n=int(v.n), quoted=_r(v.q, 3)) for k, v in temp.iterrows()],
                unrouted_3m=int((~recent.routed).sum()), open=int((d.status == 'Open').sum()),
                total=int(len(d)), quoted=int(d.quoted.sum()), won=int(d.won.sum()),
                leaked=int(d.leaked.sum()))


def accounts(snap, as_of, owners=None, customers=None, churn_cur=None):
    s = snap[['snapshot', 'quote_id', 'customer_id', 'customer', 'value', 'status', 'valid_from',
              'expected_award', 'se']].copy()
    last = s[s.snapshot == as_of]
    first_q = last.groupby('customer_id').valid_from.min().str[:7]
    last_q = last.groupby('customer_id').valid_from.max().str[:7]
    w = last[last.status == 'WON']
    last_a = w.groupby('customer_id').expected_award.max().str[:7]
    a12 = w[w.expected_award.str[:7] > ym_add(as_of, -12)].groupby('customer_id').value.sum()
    a_hist = w.groupby('customer_id').value.sum()
    name = last.groupby('customer_id').customer.first()
    owner = last.sort_values('valid_from').groupby('customer_id').se.last()
    if owners is not None and len(owners):
        cur_own = owners[owners.to_month.astype(str) == ''].set_index('customer_id').se
        owner = cur_own.combine_first(owner)
    df = pd.DataFrame({'name': name, 'owner': owner, 'first': first_q, 'last_q': last_q, 'last_a': last_a,
                       'a12': a12, 'a_hist': a_hist}).fillna({'a12': 0, 'a_hist': 0})
    act = df[['last_q', 'last_a']].max(axis=1)
    six, twelve = ym_add(as_of, -6), ym_add(as_of, -12)
    df['status'] = np.select([df['first'] > six, act > six, act > twelve], ['New', 'Active', 'Slowing'], 'Dormant')
    summ = df.groupby('status').agg(n=('name', 'size'), a12=('a12', 'sum'), hv=('a_hist', 'sum'))
    by_se = df.groupby(['owner', 'status']).size().unstack(fill_value=0)
    winback = df[df.status == 'Dormant'].sort_values('a_hist', ascending=False).head(20)
    trans = 0
    if owners is not None and len(owners):
        trans = int((owners.from_month.astype(str) > six).sum() - 0)
    out = dict(
        summary=[dict(status=k, n=int(v.n), a12=_r(v.a12), hist=_r(v.hv)) for k, v in summ.iterrows()],
        by_se=[dict(se=k, **{c: int(v.get(c, 0)) for c in ('New', 'Active', 'Slowing', 'Dormant')})
               for k, v in by_se.iterrows()],
        winback=[dict(customer=r.name, owner=r.owner, last=max(str(r.last_q), str(r.last_a)),
                      hist=_r(r.a_hist)) for r in winback.itertuples()],
        transfers_6m=trans, total=int(len(df)),
    )
    if churn_cur is not None and len(churn_cur):
        c = churn_cur.sort_values('p_churn', ascending=False)
        thr = max(0.5, float(churn_cur.p_churn.quantile(0.8)))   # riskiest fifth, at least 50%
        c = c[c.p_churn >= thr]
        out['at_risk_n'] = int(len(c))
        out['at_risk_value'] = _r(c.a12.sum())
        out['churn'] = [dict(customer=name.get(r.cid, r.cid), owner=owner.get(r.cid, ''), p=_r(r.p_churn, 3),
                             a12=_r(r.a12), quiet=int(r.m_since_quote)) for r in c.head(25).itertuples()]
    return out, df


def se_scorecard(snap, as_of, targets, invoices=None, ses=None):
    L = snap[snap.snapshot == as_of].copy()
    L['ead_m'] = L.expected_award.str[:7]
    L['vf_m'] = L.valid_from.str[:7]
    Y, el = as_of[:4], int(as_of[5:7])
    w12 = [ym_add(as_of, -i) for i in range(12)]
    won = L[L.stage.isin(WON_STAGES)]
    quo = L[L.stage.isin(HIT_STAGES)]
    co_hr = won[won.ead_m.isin(w12)].value.sum() / max(1.0, quo[quo.vf_m.isin(w12)].value.sum())
    ever_hot = set(snap[snap.stage.isin(HOT_STAGES)].quote_id)
    tg = targets.set_index('se').award_target if targets is not None and len(targets) else pd.Series(dtype=float)
    chan = ses.set_index('se').channel if ses is not None and 'channel' in ses else pd.Series(dtype=str)
    team = ses.set_index('se').team if ses is not None else L.groupby('se').team.first()
    inv = None
    if invoices is not None and len(invoices):
        iv = invoices[(invoices.invoice_date.str[:4] == Y) & (invoices.invoice_date.str[:7] <= as_of)]
        inv = iv.groupby('se').amount.sum()
    rows = []
    for se, g in L.groupby('se'):
        if se == '':
            continue
        t = float(tg.get(se, 0))
        gw = g[g.stage.isin(WON_STAGES)]
        gq = g[g.stage.isin(HIT_STAGES)]
        hr = gw[gw.ead_m.isin(w12)].value.sum() / max(1.0, gq[gq.vf_m.isin(w12)].value.sum())
        hr_b = float(np.clip(hr, 0.5 * co_hr, 2 * co_hr))
        aw = gw[(gw.ead_m >= f'{Y}-01') & (gw.ead_m <= as_of)]
        qy = gq[(gq.vf_m >= f'{Y}-01') & (gq.vf_m <= as_of)].value.sum()
        need_q = t / hr_b / 12 if t else 0
        hot_share = aw[aw.quote_id.isin(ever_hot)].value.sum() / max(1.0, aw.value.sum())
        hot_tgt = 0.5 if chan.get(se, '') == 'dealer' else 0.75
        comp = dict(quotation=qy / max(1.0, need_q * el) if t else np.nan,
                    hot_to_award=hot_share / hot_tgt,
                    award=aw.value.sum() / max(1.0, t * el / 12) if t else np.nan,
                    invoicing=(inv.get(se, 0) / max(1.0, t * 0.9 * el / 12)) if (inv is not None and t) else np.nan)
        wts = dict(quotation=0.2, hot_to_award=0.3, award=0.2, invoicing=0.3)
        avail = {k: min(v, 1.5) for k, v in comp.items() if not np.isnan(v)}
        score = sum(wts[k] * v for k, v in avail.items()) / max(1e-9, sum(wts[k] for k in avail)) * 100
        rows.append(dict(se=se, team=str(team.get(se, '')), target=_r(t), award_ytd=_r(aw.value.sum()),
                         quotes_ytd=_r(qy), hit_ratio=_r(hr, 4), needed_q_month=_r(need_q),
                         hot_share=_r(hot_share, 3), invoiced_ytd=_r(inv.get(se, 0)) if inv is not None else None,
                         **{f'c_{k}': (None if np.isnan(v) else _r(v, 3)) for k, v in comp.items()},
                         score=_r(score, 1)))
    rows.sort(key=lambda r: -r['score'])
    return dict(rows=rows, company_hit_ratio=_r(co_hr, 4), weights=dict(quotation=20, hot_to_award=30, award=20,
                                                                          invoicing=30))


def receivables(inv, as_of):
    if inv is None or not len(inv):
        return None, None
    d = inv.copy()
    d['inv'] = pd.to_datetime(d.invoice_date)
    d['due'] = pd.to_datetime(d.due_date)
    d['paid'] = pd.to_datetime(d.paid_date, errors='coerce')

    def snap_at(cut):
        o = d[(d.inv <= cut) & (d.paid.isna() | (d.paid > cut))].copy()
        o['od'] = (cut - o.due).dt.days
        return o

    cut = pd.Period(as_of, 'M').to_timestamp(how='end').normalize()
    o = snap_at(cut)
    bins = [-10 ** 6, 0, 30, 60, 90, 180, 10 ** 6]
    labels = ['Not due', '1-30', '31-60', '61-90', '91-180', '180+']
    o['bucket'] = pd.cut(o.od, bins=bins, labels=labels)
    ag = o.groupby('bucket', observed=False).amount.sum()
    months = [ym_add(as_of, -i) for i in range(11, -1, -1)]
    trend = []
    for m in months:
        c = pd.Period(m, 'M').to_timestamp(how='end').normalize()
        x = snap_at(c)
        coll = d[(d.paid.dt.strftime('%Y-%m') == m)].amount.sum()
        billed = d[(d.inv.dt.strftime('%Y-%m') == m)].amount.sum()
        trend.append(dict(m=m, ar=_r(x.amount.sum()), overdue=_r(x[x.od > 0].amount.sum()), collected=_r(coll),
                          billed=_r(billed)))
    bill3 = d[(d.inv > cut - pd.Timedelta(days=90)) & (d.inv <= cut)].amount.sum()
    dso = o.amount.sum() / max(1.0, bill3 / 90)
    cust = o[o.od > 0].groupby(['customer_id', 'customer']).agg(overdue=('amount', 'sum'), maxd=('od', 'max'))
    cust = cust.reset_index().sort_values('overdue', ascending=False)
    by_se = o.groupby('se').apply(lambda g: pd.Series(dict(ar=g.amount.sum(), overdue=g[g.od > 0].amount.sum())),
                                  include_groups=False)
    out = dict(total=_r(o.amount.sum()), overdue=_r(o[o.od > 0].amount.sum()),
               over90=_r(o[o.od > 90].amount.sum()), dso=_r(dso, 1),
               buckets=[dict(b=k, v=_r(v)) for k, v in ag.items()], trend=trend,
               top=[dict(customer=r.customer, overdue=_r(r.overdue), days=int(r.maxd)) for r in cust.head(15).itertuples()],
               by_se=[dict(se=k, ar=_r(v.ar), overdue=_r(v.overdue)) for k, v in
                      by_se.sort_values('overdue', ascending=False).iterrows()])
    return out, o
