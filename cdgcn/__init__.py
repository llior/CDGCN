from .test_cdgcn import test_cdgcn
from .train_cdgcn import train_cdgcn

__factory__ = {
    'test_cdgcn': test_cdgcn,
    'train_cdgcn': train_cdgcn,
}


def build_handler(phase):
    key_handler = '{}_cdgcn'.format(phase)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]
