def remove_recursively_empty_fields(d: dict, exclude: list = None) -> dict:
    """
    Recursively remove empty values from a dict.

    Parameters
    ----------
    d:
        The dict to clean.
    exclude:
        Keys to skip entirely (not removed, not recursed into).

    Returns
    -------
    A new dict with empty entries removed.
    """
    if exclude is None:
        exclude = []

    result = {}
    for key, val in d.items():
        if key in exclude:
            result[key] = val
            continue
        if isinstance(val, dict):
            cleaned = remove_recursively_empty_fields(val, exclude)
            if cleaned:  # only keep if non-empty after cleaning
                result[key] = cleaned
        elif not _is_empty(val):
            result[key] = val
    return result


def _is_empty(val) -> bool:
    """Return True if val is considered empty (like MATLAB isempty)."""
    import numpy as np
    import pandas as pd

    if val is None:
        return True
    if isinstance(val, str):
        return len(val) == 0
    if isinstance(val, (list, tuple)):
        return len(val) == 0
    if isinstance(val, np.ndarray):
        return val.size == 0
    if isinstance(val, pd.DataFrame):
        return val.empty
    if isinstance(val, dict):
        return len(val) == 0
    return False
