"""Pytest configuration and shared fixtures.

Background: the ARPACK eigensolver (scipy / LAPACK) crashes on its second
invocation when called inside a Starlette/ASGI TestClient on macOS with
Python 3.14 + scipy 1.17.  The crash manifests as SIGBUS/SIGSEGV (exit 139)
after the first FEM solve finishes.

Work-around applied here:
  * A session-scoped ``_warm_bundle_cache`` autouse fixture pre-compiles
    comb_resonator ONCE outside the ASGI layer (plain Python call) so that
    the ``_BUNDLES`` dict is already populated before any test requests it
    through the TestClient.  This means neither ``test_bundle_ok`` nor
    ``test_bundle_cached`` trigger a second ARPACK solve inside the ASGI
    transport, and ``test_compile_endpoint_ok`` is the FIRST (and only)
    ASGI-driven FEM call in the session.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asgi_fem: marks tests that run FEM via the ASGI test client "
        "(may crash on macOS Python 3.14 + scipy if a second FEM solve "
        "is attempted in the same TestClient session)",
    )


@pytest.fixture(scope="session", autouse=True)
def _warm_bundle_cache():
    """Pre-populate _BUNDLES for comb_resonator before any test runs.

    This ensures that TestClient calls to /api/bundle/comb_resonator return
    immediately from cache (no ARPACK eigensolver invoked through ASGI),
    leaving the ASGI FEM slot available for test_compile_endpoint_ok.
    """
    try:
        from webapp.backend import app as backend
        from soidlc import webbundle
        import os

        path = os.path.join(backend.EXAMPLES, "comb_resonator.soidl")
        mtime = os.path.getmtime(path)
        b = webbundle.build_bundle(path)
        backend._BUNDLES["comb_resonator"] = (mtime, b)
    except Exception:
        # If the warm-up fails (e.g. no webapp installed), tests that need
        # it will surface their own failures.
        pass
