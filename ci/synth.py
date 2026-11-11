"""Synthetic data generator.

Creates a fictional B2B building-products distributor ("Northwind Climate Co.") with invented customers, sales
engineers, quotations, monthly pipeline snapshots, invoices / payments, leads and account ownership.

The data is not sampled from any real company. It is simulated from a small behavioural model so the analytics
and models have realistic structure to find:
  * every customer has a hidden loyalty score and a payment habit; every sales engineer has a hidden skill score
  * quotations move through stages month by month (Markov chain) with odds driven by those hidden scores,
    quotation size, age and seasonality; expected award dates slip when deals stall
  * some customers fade out (fewer quotations, slower payments) before they go dormant
  * awards are invoiced over a delivery schedule and paid according to credit terms and payment habit
  * leads arrive through three channels and a share convert into quotations

Usage:  python -m ci.synth [--seed 7] [--out data/synthetic]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .schema import SNAP_COLS, INV_COLS, LEAD_COLS, OWN_COLS

START = '2024-07'
END = '2026-10'
COMPANY = 'Northwind Climate Co.'

REGIONS = ['Central', 'East', 'West']
TEAMS = {  # team -> (region, channel)
    'Projects Central': ('Central', 'project'), 'Projects East': ('East', 'project'),
    'Projects West': ('West', 'project'),
    'Dealers Central': ('Central', 'dealer'), 'Dealers East': ('East', 'dealer'),
    'Dealers West': ('West', 'dealer'),
    'Key Accounts': ('Central', 'key'), 'Consultant Channel': ('Central', 'project'),
}
GROUPS = {  # product group -> (share by channel: project, dealer, key), value multiplier
    'Ducted Units': ((0.30, 0.25, 0.20), 1.0),
    'Wall Split Units': ((0.10, 0.35, 0.10), 0.5),
    'Room AC': ((0.05, 0.35, 0.05), 0.25),
    'VRF Systems': ((0.35, 0.05, 0.35), 2.2),
    'Chillers': ((0.20, 0.00, 0.30), 3.5),
}
FIRST = ['Omar', 'Faisal', 'Khalid', 'Nadia', 'Yusuf', 'Rami', 'Sara', 'Tariq', 'Hana', 'Imran', 'Lina', 'Sami',
         'Adel', 'Mona', 'Zaid', 'Rania', 'Bilal', 'Dina', 'Hadi', 'Noor', 'Karim', 'Laila', 'Fadi', 'Rida']
LAST = ['Haddad', 'Farouk', 'Najjar', 'Saleh', 'Kassem', 'Darwish', 'Mansour', 'Aziz', 'Hamdan', 'Rahman',
        'Shams', 'Qureshi', 'Barakat', 'Yassin', 'Taher', 'Hilal', 'Sabri', 'Khoury', 'Nasser', 'Ghanem',
        'Zahid', 'Malik', 'Fawzi', 'Ismail']
CUST_A = ['Falcon', 'Oasis', 'Cedar', 'Summit', 'Harbor', 'Crescent', 'Granite', 'Silver', 'Desert', 'Horizon',
          'Palm', 'Atlas', 'Beacon', 'Coral', 'Delta', 'Emerald', 'Frontier', 'Golden', 'Ivory', 'Jade', 'Keystone',
          'Lumen', 'Meridian', 'Nova', 'Orchid', 'Pioneer', 'Quartz', 'Riverstone', 'Sapphire', 'Topaz', 'Unity',
          'Vista', 'Willow', 'Zenith', 'Amber', 'Bluewave', 'Citadel', 'Dune', 'Everline', 'Fountain']
CUST_B = ['Builders', 'Contracting', 'Developments', 'MEP Services', 'Construction', 'Trading', 'Engineering',
          'Facilities', 'Projects', 'Cooling Solutions', 'Technical Services', 'Real Estate']
PROJ_A = ['Residential Tower', 'Villa Compound', 'Hospital Extension', 'School Campus', 'Mall Fit-out',
          'Hotel Renovation', 'Office Block', 'Warehouse', 'Mosque', 'Clinic', 'Labour Housing', 'Data Hall',
          'Showroom', 'University Annex', 'Apartment Complex', 'Retail Strip']
PROJ_B = ['Phase 1', 'Phase 2', 'Block A', 'Block B', 'North', 'South', 'Plot 14', 'Plot 27', 'Stage 3', '']


def months(a=START, b=END):
    return [str(p) for p in pd.period_range(a, b, freq='M')]


def build(seed=7):
    rng = np.random.default_rng(seed)
    M = months()
    nM = len(M)

    # --- sales engineers -------------------------------------------------------------------------------------
    names = rng.permutation([f'{f} {l}' for f, l in zip(FIRST, rng.permutation(LAST))])
    ses, k = [], 0
    for team, (region, ch) in TEAMS.items():
        for _ in range(3):
            ses.append(dict(se=names[k], team=team, region=region, channel=ch, skill=rng.normal(0, 0.45)))
            k += 1
    se_df = pd.DataFrame(ses)

    # --- customers ---------------------------------------------------------------------------------------------
    n_c = 1200
    used, cust = set(), []
    for i in range(n_c):
        while True:
            nm = f'{rng.choice(CUST_A)} {rng.choice(CUST_A)} {rng.choice(CUST_B)}'
            if nm.split()[0] != nm.split()[1] and nm not in used:
                used.add(nm)
                break
        ch = rng.choice(['project', 'dealer', 'key'], p=[0.55, 0.40, 0.05])
        region = rng.choice(REGIONS, p=[0.45, 0.30, 0.25])
        cands = se_df[(se_df.channel == ch) & ((se_df.region == region) | (ch == 'key'))]
        if ch == 'project' and rng.random() < 0.15:
            cands = se_df[se_df.team == 'Consultant Channel']
        owner = cands.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        lam = {'project': 0.22, 'dealer': 0.75, 'key': 0.55}[ch] * rng.lognormal(0, 0.5)
        churn = int(rng.integers(3, nM)) if rng.random() < 0.28 else 10 ** 6        # month index churn starts
        start = 0 if rng.random() < 0.8 else int(rng.integers(1, nM - 3))             # new customers join later
        cust.append(dict(customer_id=f'C{10000 + i}', customer=nm, channel=ch, region=region, owner=owner.se,
                         loyalty=rng.normal(0, 0.8), pay_delay=max(0, rng.normal(18, 22)),
                         credit=int(rng.choice([0, 30, 60, 90, 120], p=[0.15, 0.25, 0.3, 0.2, 0.1])),
                         lam=lam, churn=churn, start=start))
    cu = pd.DataFrame(cust)

    # --- ownership history (transfers) -------------------------------------------------------------------------
    own = {r.customer_id: [(r.owner, M[0])] for r in cu.itertuples()}

    # --- quotation lifecycle -----------------------------------------------------------------------------------
    team_med = {'project': 380_000, 'dealer': 55_000, 'key': 900_000}
    quotes, snaps, qn = [], [], 20_000_000
    season = {1: 1.0, 2: 1.0, 3: 0.85, 4: 0.9, 5: 1.0, 6: 0.95, 7: 0.75, 8: 0.8, 9: 1.0, 10: 1.05, 11: 1.1, 12: 1.2}
    se_idx = se_df.set_index('se')
    cu_idx = cu.set_index('customer_id')
    for t, m in enumerate(M):
        mo = int(m[5:])
        # transfers
        for cid in cu.customer_id.sample(frac=0.004, random_state=int(rng.integers(1e9))):
            cur = own[cid][-1][0]
            r = se_idx.loc[cur]
            alt = se_df[(se_df.team == r.team) & (se_df.se != cur)]
            if len(alt):
                own[cid].append((alt.sample(1, random_state=int(rng.integers(1e9))).iloc[0].se, m))
        # new quotations
        for c in cu.itertuples():
            if t < c.start:
                continue
            lam = c.lam * season[mo] * (0.62 ** max(0, t - c.churn))
            if t > c.churn + 8 and rng.random() < 0.02:               # occasional win-back
                lam = c.lam * 0.6
            for _ in range(rng.poisson(lam)):
                se = own[c.customer_id][-1][0]
                ch = c.channel
                gshare = np.array([GROUPS[g][0][['project', 'dealer', 'key'].index(ch)] for g in GROUPS])
                g = rng.choice(list(GROUPS), p=gshare / gshare.sum())
                val = round(team_med[ch] * GROUPS[g][1] * rng.lognormal(0, 0.9), -2)
                day = int(rng.integers(1, 28))
                vf = f'{m}-{day:02d}'
                lag = {'project': rng.integers(2, 10), 'dealer': rng.integers(0, 3), 'key': rng.integers(3, 12)}[ch]
                ead = (pd.Period(m, 'M') + int(lag)).strftime('%Y-%m') + f'-{int(rng.integers(1, 28)):02d}'
                stage = 15 if (ch != 'dealer' and rng.random() < 0.30) else 30
                if ch == 'dealer' and rng.random() < 0.15:
                    stage = 50
                qn += int(rng.integers(1, 40))
                q = dict(quote_id=str(qn), customer_id=c.customer_id, customer=c.customer,
                         project=(f'{rng.choice(PROJ_A)} {rng.choice(PROJ_B)}'.strip() if ch != 'dealer'
                                  else f'Dealer order {m}'),
                         se=se, team=se_idx.loc[se].team, region=c.region, product_group=g, value=val,
                         stage=stage, valid_from=vf, expected_award=ead, born=t, won_t=None,
                         z=0.9 * c.loyalty + 0.9 * se_idx.loc[se].skill - 0.35 * np.log(val / team_med[ch])
                           + rng.normal(0, 0.6), ch=ch)
                quotes.append(q)
        # transitions for quotes created before this month
        for q in quotes:
            if q['born'] >= t or q['stage'] in (0, 100):
                continue
            s, z, age = q['stage'], q['z'], t - q['born']
            fast = 1.8 if q['ch'] == 'dealer' else 1.0
            adv, lose = np.exp(0.75 * z) * fast * season[mo], np.exp(-0.6 * z) * (1.6 if age > 9 else 1.0)
            if s == 90:
                if rng.random() < 0.5:
                    q['stage'] = 100
                continue
            P = {15: {30: 0.13 * adv, 0: 0.035 * lose},
                 10: {30: 0.05 * adv, 0: 0.07 * lose},
                 30: {50: 0.12 * adv, 70: 0.03 * adv, 90: 0.035 * adv, 0: 0.06 * lose, 10: 0.02},
                 50: {70: 0.16 * adv, 90: 0.06 * adv, 0: 0.06 * lose, 10: 0.015},
                 70: {90: 0.24 * adv, 0: 0.08 * lose}}[s]
            tot = sum(P.values())
            if tot > 0.9:
                P = {k: v * 0.9 / tot for k, v in P.items()}
            r, acc = rng.random(), 0.0
            for ns, p in P.items():
                acc += p
                if r < acc:
                    q['stage'] = ns
                    if ns == 90:
                        q['won_t'] = t
                        q['expected_award'] = f'{m}-{int(rng.integers(1, 28)):02d}'
                    break
            if q['stage'] in (10, 15, 30, 50, 70) and q['expected_award'][:7] < m and rng.random() < 0.65:
                q['expected_award'] = ((pd.Period(m, 'M') + int(rng.integers(1, 4))).strftime('%Y-%m')
                                       + f'-{int(rng.integers(1, 28)):02d}')
        # snapshot
        for q in quotes:
            st = 'WON' if q['stage'] in (90, 100) else 'LOST' if q['stage'] == 0 else 'OPEN'
            snaps.append((m, q['quote_id'], q['customer_id'], q['customer'], q['project'], q['se'], q['team'],
                          q['region'], q['product_group'], q['value'], q['stage'], st, q['valid_from'],
                          q['expected_award']))
    snap = pd.DataFrame(snaps, columns=SNAP_COLS)

    # --- invoices & payments -------------------------------------------------------------------------------------
    inv, ni = [], 0
    end_date = pd.Timestamp(END) + pd.offsets.MonthEnd(0)
    for q in quotes:
        if q['won_t'] is None:
            continue
        c = cu_idx.loc[q['customer_id']]
        parts = 1 if q['ch'] == 'dealer' else int(rng.integers(2, 6))
        base = pd.Period(M[q['won_t']], 'M') + (0 if q['ch'] == 'dealer' else 1)
        amt = q['value'] * rng.uniform(0.93, 1.0) / parts
        for p in range(parts):
            per = base + p + (int(rng.integers(0, 2)) if p else 0)
            d = per.to_timestamp() + pd.Timedelta(days=int(rng.integers(0, 27)))
            if d > end_date:
                continue
            ni += 1
            due = d + pd.Timedelta(days=int(c.credit))
            churning = q['won_t'] >= c.churn - 2
            delay = max(0.0, rng.normal(c.pay_delay + (25 if churning else 0), 20))
            paid = due + pd.Timedelta(days=int(delay))
            if rng.random() < 0.025:
                paid = pd.NaT                                        # never paid (dispute / bad debt)
            inv.append(dict(invoice_id=f'INV{ni:06d}', customer_id=c.name, customer=c.customer, se=q['se'],
                            region=c.region, quote_id=q['quote_id'], invoice_date=d.date().isoformat(),
                            amount=round(amt, 2), credit_days=int(c.credit), due_date=due.date().isoformat(),
                            paid_date=(None if pd.isna(paid) or paid > end_date else paid.date().isoformat())))
    inv = pd.DataFrame(inv, columns=INV_COLS)

    # --- leads -----------------------------------------------------------------------------------------------------
    leads, nl = [], 0
    qdf = pd.DataFrame(quotes)
    q_by_month_region = {k: list(v.quote_id) for k, v in qdf[qdf.ch != 'dealer'].groupby(
        [qdf.born.map(lambda i: M[i]), 'region'])}
    taken = set()
    ch_rate = {'Consultant spec-in': (55, 0.34), 'Project tracker': (95, 0.20), 'Inbound call centre': (45, 0.28)}
    se_by_region = se_df[se_df.channel == 'project'].groupby('region').se.apply(list).to_dict()
    for t, m in enumerate(M):
        for chn, (n, conv) in ch_rate.items():
            for _ in range(rng.poisson(n)):
                nl += 1
                region = rng.choice(REGIONS, p=[0.45, 0.30, 0.25])
                temp = rng.choice(['Hot', 'Warm', 'Cold'], p=[0.25, 0.45, 0.30])
                p = conv * {'Hot': 1.6, 'Warm': 1.0, 'Cold': 0.45}[temp]
                se = rng.choice(se_by_region[region]) if rng.random() < 0.88 else ''
                if not se:
                    p *= 0.3                                       # unrouted leads rarely convert
                qid, status, cid = '', 'Open', ''
                if t < nM - 1 and rng.random() < p:
                    tm = M[min(nM - 1, t + int(rng.integers(0, 3)))]
                    pool = [x for x in q_by_month_region.get((tm, region), []) if x not in taken]
                    if pool:
                        qid = pool[int(rng.integers(len(pool)))]
                        taken.add(qid)
                        status = 'Quoted'
                if not qid and t < nM - 3:
                    status = rng.choice(['Lost - no response', 'Leaked to competitor', 'Disqualified'],
                                        p=[0.5, 0.3, 0.2]) if rng.random() < 0.8 else 'Open'
                leads.append(dict(lead_id=f'L{nl:06d}', channel=chn,
                                  created=f'{m}-{int(rng.integers(1, 28)):02d}', region=region, se=se,
                                  customer_id=cid, project=f'{rng.choice(PROJ_A)} {rng.choice(PROJ_B)}'.strip(),
                                  est_value=round(380_000 * rng.lognormal(0, 1.0), -3), temperature=temp,
                                  status=status, quote_id=qid))
    leads = pd.DataFrame(leads, columns=LEAD_COLS)
    last = snap[snap.snapshot == END].set_index('quote_id')
    leads = leads.merge(last[['customer_id', 'status']].rename(columns={'status': 'qstatus', 'customer_id': 'qc'}),
                        left_on='quote_id', right_index=True, how='left')
    leads['customer_id'] = leads.qc.fillna('')
    leads.loc[leads.qstatus == 'WON', 'status'] = 'Won'
    leads.loc[leads.qstatus == 'LOST', 'status'] = 'Lost after quote'
    leads = leads[LEAD_COLS]

    # --- ownership table + targets --------------------------------------------------------------------------------
    rows = []
    for cid, hist in own.items():
        for i, (se, frm) in enumerate(hist):
            to = hist[i + 1][1] if i + 1 < len(hist) else ''
            rows.append((cid, se, frm, to))
    owners = pd.DataFrame(rows, columns=OWN_COLS)
    won = snap[(snap.snapshot == END) & snap.stage.isin([90, 100])].copy()
    won['yr'] = won.expected_award.str[:4]
    a25 = won[won.yr == '2025'].groupby('se').value.sum()
    tgt = (se_df.set_index('se').index.to_series().map(lambda s: a25.get(s, 0)) * rng.uniform(1.05, 1.3, len(se_df)))
    targets = pd.DataFrame({'se': tgt.index, 'year': 2026, 'award_target': (tgt.values / 1e5).round() * 1e5})
    customers = cu[['customer_id', 'customer', 'channel', 'region', 'credit']].rename(columns={'credit': 'credit_days'})
    return dict(snapshots=snap, invoices=inv, leads=leads, owners=owners, targets=targets, customers=customers,
                ses=se_df[['se', 'team', 'region', 'channel']],
                meta=dict(company=COMPANY, as_of=END, synthetic=True, seed=seed))


def save(d, out):
    os.makedirs(out, exist_ok=True)
    d['snapshots'].to_csv(os.path.join(out, 'snapshots.csv.gz'), index=False)
    for k in ('invoices', 'leads', 'owners', 'targets', 'customers', 'ses'):
        d[k].to_csv(os.path.join(out, f'{k}.csv'), index=False)
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(d['meta'], f, indent=1)


def load(folder):
    d = dict(snapshots=pd.read_csv(os.path.join(folder, 'snapshots.csv.gz'), dtype={'quote_id': str}))
    for k in ('invoices', 'leads', 'owners', 'targets', 'customers', 'ses'):
        d[k] = pd.read_csv(os.path.join(folder, f'{k}.csv'), dtype=str, keep_default_na=False)
    for k, cols in (('invoices', ['amount', 'credit_days']), ('leads', ['est_value']), ('targets', ['award_target'])):
        for c in cols:
            d[k][c] = pd.to_numeric(d[k][c])
    d['meta'] = json.load(open(os.path.join(folder, 'meta.json')))
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic'))
    a = ap.parse_args()
    data = build(a.seed)
    save(data, a.out)
    s = data['snapshots']
    print({k: len(v) for k, v in data.items() if hasattr(v, '__len__') and k != 'meta'})
    print(s[s.snapshot == END].status.value_counts().to_dict())
