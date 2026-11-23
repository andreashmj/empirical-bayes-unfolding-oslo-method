import numpy as np


# Approximate gates: Eg1, Eg2, Ex1, Ex2, shape_Ex1, shape_Ex2 (keV).
GATES = (
    (4200., 4700., 6750., 7500., 7000., 7250.),
    (5850., 6400., 7650., 8450., 7900., 8150.),
    (6700., 7400., 8500., 9350., 8750., 9100.),
)


def remove_contamination(values, Ex, Eg):
    """MAMA RD; array order (..., Ex, Eg); no final nonnegative clipping."""
    result = np.array(values, dtype=float, copy=True)
    Ex, Eg = np.asarray(Ex), np.asarray(Eg)
    for eg1, eg2, ex1, ex2, lower, upper in GATES:
        if ex1 >= Ex[-1] or ex2 <= Ex[0] or eg1 >= Eg[-1] or eg2 <= Eg[0]:
            continue
        x1, x2 = [np.abs(Eg - e).argmin() for e in (eg1, eg2)]
        y1, y2 = [np.abs(Ex - e).argmin() for e in (ex1, ex2)]
        l1, l2 = [np.abs(Ex - e).argmin() for e in (lower, upper)]
        fraction = np.arange(1, x2 - x1) / (x2 - x1)
        rows = result[..., l1:l2 + 1, :]
        background = rows[..., x1, None] + fraction * (
            rows[..., x2, None] - rows[..., x1, None]
        )
        shape = (rows[..., x1 + 1:x2] - background).sum(axis=-2)
        area = np.abs(shape.sum(axis=-1, keepdims=True))
        shape = np.divide(shape, area, out=np.zeros_like(shape), where=area != 0)
        edges = result[..., y1:y2 + 1, :][..., [x1, x2]]
        edges = (edges[..., :-2, :] + edges[..., 1:-1, :] + edges[..., 2:, :]) / 3
        background = edges[..., :1] + fraction * (edges[..., 1:] - edges[..., :1])
        region = result[..., y1 + 1:y2, x1 + 1:x2]
        amplitude = np.maximum((region - background).sum(axis=-1), 0)
        region -= amplitude[..., None] * shape[..., None, :]
    return result
