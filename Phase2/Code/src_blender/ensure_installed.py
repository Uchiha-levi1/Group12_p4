import importlib
import os
import site
import subprocess
import sys


def ensure_installed(package_name: str, import_name: str | None = None) -> None:
    module_name = import_name or package_name

    try:
        importlib.import_module(module_name)
        return
    except ImportError:
        pass

    user_site = site.getusersitepackages()
    os.makedirs(user_site, exist_ok=True)

    if user_site not in sys.path:
        sys.path.append(user_site)

    try:
        import pip  # noqa: F401
    except ImportError:
        import ensurepip
        ensurepip.bootstrap()

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--user",
        package_name,
    ])

    importlib.invalidate_caches()

    if user_site not in sys.path:
        sys.path.append(user_site)

    importlib.import_module(module_name)
