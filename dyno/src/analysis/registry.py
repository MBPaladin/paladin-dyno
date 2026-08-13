"""Processor registry. Adding a processor = adding one file under processors/."""

import importlib
import pkgutil

_REGISTRY = {}


def register(cls):
    """Class decorator. Registers by the processor's `name`."""
    if not getattr(cls, 'name', ''):
        raise ValueError(f'{cls.__name__} must define a non-empty `name`')
    if cls.name in _REGISTRY:
        raise ValueError(f'Duplicate processor name: {cls.name!r}')
    _REGISTRY[cls.name] = cls
    return cls


def _discover():
    """Import every module under processors/ so their @register calls fire."""
    from . import processors
    for mod in pkgutil.iter_modules(processors.__path__):
        importlib.import_module(f'{processors.__name__}.{mod.name}')


def all_processors():
    _discover()
    return [cls() for cls in _REGISTRY.values()]


def get_processor(name):
    _discover()
    if name not in _REGISTRY:
        raise KeyError(f'Unknown processor {name!r}. Known: {sorted(_REGISTRY)}')
    return _REGISTRY[name]()
