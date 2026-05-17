"""Print all Unitree-DSC environments registered by `unitree_dsc_lab`."""

import importlib
import pathlib
import pkgutil
import sys


def _walk_packages(path=None, prefix="", onerror=None):
    def seen(p, m={}):
        if p in m:
            return True
        m[p] = True

    for info in pkgutil.iter_modules(path, prefix):
        yield info
        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path = getattr(sys.modules[info.name], "__path__", None) or []
                path = [p for p in path if not seen(p)]
                yield from _walk_packages(path, info.name + ".", onerror)


def import_packages():
    sys.path.insert(
        0,
        f"{pathlib.Path(__file__).parent.parent}/source/unitree_dsc_lab/unitree_dsc_lab/tasks/",
    )
    for package in ["stair_climb.robots"]:
        package = importlib.import_module(package)
        for _ in _walk_packages(package.__path__, package.__name__ + "."):
            pass
    sys.path.pop(0)


import_packages()

import gymnasium as gym
from prettytable import PrettyTable


def main():
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Unitree DSC Lab"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            index += 1

    print(table)


if __name__ == "__main__":
    main()
