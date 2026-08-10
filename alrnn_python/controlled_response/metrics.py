"""Response summaries and scaling records."""

from collections import defaultdict
from statistics import mean, median

import torch

from .geometry import euclidean_norm, rms, safe_cosine
from .types import ResponseMetrics


def response_metrics(r, delta_hat, cosine_eps, include_cosine):
    # Summarize one probability response trajectory
    r_norm = euclidean_norm(r)

    # except for t=0, compute the dynamic RMS of the response trajectory
    r_rms = float(rms(r_norm[1:])) 

    # for own response, check dirction
    r_cosine = safe_cosine(r, delta_hat.expand_as(r), cosine_eps) if include_cosine else None

    return ResponseMetrics(
        response=r,
        norm=r_norm,
        cosine=r_cosine,
        peak=float(r_norm.max()),
        rms=r_rms,
        terminal=float(r_norm[-1]),
        time_to_peak=int(torch.argmax(r_norm)),
    )


def _aggregate(values, statistic):
    # check the overall response behavior of the model across initial conditions
    finite = [value for value in values if torch.isfinite(torch.tensor(value))]
    if not finite:
        return float("nan")
    if statistic == "median":
        return float(median(finite))
    if statistic == "mean":
        return float(mean(finite))
    raise ValueError("statistic must be 'mean' or 'median'")


def scaling_records(runs, statistic="median"):
    """Aggregate own/cross response magnitude at each configured alpha."""
    groups = defaultdict(list)
    for run in runs:
        if run.valid:
            groups[(run.source_player, run.sign, run.alpha)].append(run)
    records = []
    for (source, sign, alpha), selected in sorted(groups.items()):
        records.append({
            "source_player": source,
            "sign": sign,
            "alpha": alpha,
            "own_rms": _aggregate((run.own.rms for run in selected), statistic),
            "cross_rms": _aggregate((run.cross.rms for run in selected), statistic),
        })
    return records
