"""FTN experiment package; individual scripts also support direct execution."""

def __getattr__(name):
    # Preserve ``from ftn import FTN`` when the renamed directory is imported
    # as a package instead of loading its ftn.py module directly.
    from importlib import import_module
    return getattr(import_module('.ftn', __name__), name)
