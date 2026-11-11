"""Canonical schema shared by the synthetic generator and the private adapters.

Every module downstream reads only these columns, so the same code runs on demo data or on a company's own
CRM / ERP extracts once an adapter maps them here.
"""

# Quotation pipeline, one row per quotation per month-end snapshot
SNAP_COLS = [
    'snapshot',        # 'YYYY-MM' month-end snapshot
    'quote_id',        # str
    'customer_id',     # str
    'customer',        # str
    'project',         # str
    'se',              # sales engineer (owner)
    'team',            # sales team
    'region',          # Central / East / West ...
    'product_group',   # product family
    'value',           # quotation gross value (SAR)
    'stage',           # stage probability as int: 0,10,15,30,50,70,90,100
    'status',          # OPEN / WON / LOST
    'valid_from',      # date the quotation was issued (YYYY-MM-DD)
    'expected_award',  # expected award date in this snapshot (YYYY-MM-DD)
]

STAGES = {
    0: 'Lost',
    10: 'On hold',
    15: 'Tender / budget rate',
    30: '1st quotation',
    50: '2nd quotation',
    70: 'Final quotation',
    90: 'Order received',
    100: 'Order & payment',
}
OPEN_STAGES = [10, 15, 30, 50, 70]
HOT_STAGES = [50, 70]
WON_STAGES = [90, 100]
HIT_STAGES = [0, 10, 30, 50, 70, 90, 100]   # hit-ratio denominator excludes 15% tender/budget quotes

# Invoices (one row per invoice)
INV_COLS = ['invoice_id', 'customer_id', 'customer', 'se', 'region', 'quote_id', 'invoice_date', 'amount',
            'credit_days', 'due_date', 'paid_date']

# Leads (one row per lead)
LEAD_COLS = ['lead_id', 'channel', 'created', 'region', 'se', 'customer_id', 'project', 'est_value',
             'temperature', 'status', 'quote_id']

# Account ownership history
OWN_COLS = ['customer_id', 'se', 'from_month', 'to_month']
