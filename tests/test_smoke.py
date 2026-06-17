"""Smoke test: the package imports and pytest is wired up."""

def test_guard_package_imports():
    import guard  # noqa: F401
