import numpy as np
from expanded_run import summarize

def test_padding_and_incomplete_horizons_are_excluded():
    rows=[dict(tokens=list(range(151)),reset_at=20,score_start=20),
          dict(tokens=list(range(41)),reset_at=30,score_start=30)]
    normal=np.zeros((2,150),dtype=np.float32)
    pert=np.ones((2,150),dtype=np.float32)
    pert[1,40:]=9999
    summary=summarize([(rows,{'normal':normal,'reset_once':pert,'carry_off':pert})])
    assert summary['carry_off']['scored']['signed']['mean_delta_nll']==1
    assert summary['carry_off']['scored']['signed']['documents']==2
    assert summary['reset_once']['lag_0_0']['signed']['documents']==2
    assert summary['reset_once']['lag_32_127']['signed']['documents']==1
    assert summary['reset_once']['lag_32_127']['signed']['mean_delta_nll']==1
