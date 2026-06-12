# Root conftest: makes `agenteval` importable when running the test suite
# from a source checkout and registers the plugin + pytester for plugin tests.
pytest_plugins = ["agenteval.pytest_plugin", "pytester"]
