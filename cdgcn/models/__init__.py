from .cdgcn import cdgcn

__factory__ = {
    'cdgcn': cdgcn,
}


def build_model(name, *args, **kwargs):
    if name not in __factory__:
        raise KeyError("Unknown model:", name)
    return __factory__[name](*args, **kwargs)
