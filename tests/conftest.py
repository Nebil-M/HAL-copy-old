import os
import sys

# Ensure headless plotting during tests
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception:
    # If matplotlib isn't installed or fails to configure, ignore silently for tests
    pass

import pytest


def _ensure_project_root_on_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_project_root_on_path()


@pytest.fixture(autouse=True)
def _suppress_matplotlib_show_and_cleanup(monkeypatch):
    """Suppress plt.show() and close all figures after each test."""
    try:
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None, raising=False)
        yield
        try:
            plt.close("all")
        except Exception:
            pass
    except Exception:
        # Matplotlib not available or failed to import; nothing to suppress
        yield
