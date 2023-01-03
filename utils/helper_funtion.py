def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

