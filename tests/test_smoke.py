"""Smoke test: generate a small synthetic company and run the full build."""
import os
import tempfile

from ci import synth, build


def test_end_to_end():
    d = synth.build(seed=3)
    out = tempfile.mkdtemp()
    b = build.run(d, out, public=True)
    assert os.path.getsize(os.path.join(out, 'index.html')) > 100_000
    assert b['predictive']['win']['auc_model'] > b['predictive']['win']['auc_stage']
