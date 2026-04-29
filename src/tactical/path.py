from simap.nlp_colloc.tactical import path as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

del _name, _impl
