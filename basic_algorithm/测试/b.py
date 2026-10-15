import runpy

# r = runpy.run_path("a.py",init_globals={"a":2})
r = runpy.run_path("a.py")

# import importlib.util
# spec = importlib.util.spec_from_file_location("a", "a.py")
# module = importlib.util.module_from_spec(spec)
# module.a = 3
# spec.loader.exec_module(module)