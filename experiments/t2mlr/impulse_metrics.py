"""Statistical units and censored sustained-threshold recovery definitions."""
import numpy as np

def bootstrap(values):
    v=np.asarray(values,dtype=float);assert len(v) and np.isfinite(v).all()
    rng=np.random.default_rng(0);means=v[rng.integers(len(v),size=(2000,len(v)))].mean(1)
    return dict(mean=float(v.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n=len(v))

def recovery_time(curve,fraction):
    curve=np.asarray(curve)
    assert curve.shape==(128,) and np.isfinite(curve).all()
    if curve[0]<=1e-10:return np.nan,False
    for lag in range(1,124):
        if np.all(curve[lag:lag+5]<=fraction*curve[0]):return float(lag),True
    return 124.,False

def recovery_summary(curves,fraction):
    outcomes=[recovery_time(c,fraction) for c in curves]
    valid=[v for v in outcomes if np.isfinite(v[0])]
    if not valid:return dict(defined=0,undefined=len(outcomes))
    times=np.array([v[0] for v in valid]);events=np.array([v[1] for v in valid])
    return dict(defined=len(valid),undefined=len(outcomes)-len(valid),events=int(events.sum()),censored=int((~events).sum()),
                recovery_fraction=bootstrap(events.astype(float)),restricted_mean_to_124=bootstrap(times),
                median_recovery_lag=float(np.sort(times)[int(np.ceil(len(times)/2))-1]) if events.mean()>=.5 else None,
                median_censored=bool(events.mean()<.5))
