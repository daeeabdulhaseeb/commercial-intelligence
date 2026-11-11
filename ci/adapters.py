"""Adapters: map a company's own extracts to the canonical schema (ci/schema.py).

`sap_pipeline_snapshots` reads monthly pipeline snapshots exported from an ERP / CRM quotation report (one row per
quotation line per month-end) where the probability is text such as '30% - first quotation'. Every column name,
the 'valid document' flag and the product groups come from the private config; nothing company-specific is
hard-coded here.
"""
import re

import numpy as np
import pandas as pd

from .schema import SNAP_COLS

DEFAULT_MAP = {   # canonical name -> column in the source extract; override everything via config
    'snapshot': 'snapshot', 'quote_id': 'quotation_no', 'customer_id': 'customer_no', 'customer': 'customer_name',
    'project': 'project_name', 'se': 'sales_employee', 'team': 'team', 'product_group': 'product_group',
    'value': 'gross_value', 'stage_text': 'probability', 'doc_status': 'document_status',
    'valid_from': 'valid_from', 'expected_award': 'expected_award_date',
}


def _stage(x):
    m = re.match(r'\s*(\d+)\s*%', str(x))
    return int(m.group(1)) if m else -1


def sap_pipeline_snapshots(df, colmap=None, valid_value=None, groups=None,
                           region_of_team=None):
    c = {**DEFAULT_MAP, **(colmap or {})}
    d = df.copy()
    if valid_value is not None and c['doc_status'] in d:
        d = d[d[c['doc_status']].astype(str).str.strip() == valid_value]
    if groups is not None:
        d = d[d[c['product_group']].isin(groups)]
    out = pd.DataFrame({
        'snapshot': d[c['snapshot']].astype(str).str[:7],
        'quote_id': d[c['quote_id']].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.lstrip('0'),
        'customer_id': d[c['customer_id']].astype(str).str.strip().str.replace(r'\.0$', '', regex=True),
        'customer': d[c['customer']].astype(str).str.strip(),
        'project': d[c['project']].astype(str).str.strip(),
        'se': d[c['se']].fillna('').astype(str).str.strip(),
        'team': d[c['team']].fillna('').astype(str).str.strip(),
        'product_group': d[c['product_group']].astype(str).str.strip(),
        'value': pd.to_numeric(d[c['value']], errors='coerce').fillna(0.0),
        'stage': d[c['stage_text']].map(_stage),
        'valid_from': d[c['valid_from']].astype(str).str[:10],
        'expected_award': d[c['expected_award']].astype(str).str[:10],
    })
    rt = region_of_team or (lambda t: next((r for r in ('Central', 'East', 'West') if r in t), 'Other'))
    out['region'] = out.team.map(rt)
    out = out[(out.stage >= 0) & (out.quote_id != '') & out.valid_from.str.match(r'\d{4}-\d{2}')
              & out.expected_award.str.match(r'\d{4}-\d{2}')]
    out['status'] = np.where(out.stage >= 90, 'WON', np.where(out.stage == 0, 'LOST', 'OPEN'))
    # several material / item lines per quotation: sum the value, keep the header fields of the first line
    val = out.groupby(['snapshot', 'quote_id']).value.sum()
    out = out.drop_duplicates(['snapshot', 'quote_id'], keep='first').set_index(['snapshot', 'quote_id'])
    out['value'] = val
    return out.reset_index()[SNAP_COLS]
