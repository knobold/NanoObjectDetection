"""Backward-compatibility module forwarding to nanoobjectdetection."""

import sys
import nanoobjectdetection

# Forward package attributes
__all__ = nanoobjectdetection.__all__
__version__ = nanoobjectdetection.__version__

for name in __all__:
    globals()[name] = getattr(nanoobjectdetection, name)

# Ensure submodules imported via `from nanoobjectdetection_2_0.submodule import ...` work
import importlib

class _CompatibilityFinder:
    """Redirects imports from nanoobjectdetection_2_0 to nanoobjectdetection."""
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname.startswith("nanoobjectdetection_2_0."):
            submodule = fullname[len("nanoobjectdetection_2_0."):]
            try:
                real_module = importlib.import_module(f"nanoobjectdetection.{submodule}")
                sys.modules[fullname] = real_module
                return real_module.__spec__
            except ImportError:
                return None
        return None

sys.meta_path.insert(0, _CompatibilityFinder)
