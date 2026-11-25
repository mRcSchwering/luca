import argparse
import importlib
import pkgutil

from src import cmds


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    for module_info in pkgutil.iter_modules(cmds.__path__):
        module = importlib.import_module(f"src.cmds.{module_info.name}")
        module.register(subparsers.add_parser(module_info.name))

    kwargs = vars(parser.parse_args())
    func = kwargs.pop("func")
    func(**kwargs)


if __name__ == "__main__":
    main()
