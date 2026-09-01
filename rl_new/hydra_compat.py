"""Temporary compatibility for Hydra 1.3.2 on Python 3.14."""

import argparse
import sys
from functools import wraps
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


_WRAPPER_MARKER = "__newcpprl_hydra_py314_compat__"


def install_hydra_py314_compat() -> bool:
    """Install Hydra's upstream Python 3.14 parser workaround when needed."""
    try:
        hydra_version = version("hydra-core")
    except PackageNotFoundError:
        return False

    if sys.version_info < (3, 14) or hydra_version != "1.3.2":
        return False

    hydra_main = import_module("hydra.main")
    current_get_args_parser = hydra_main.get_args_parser
    if getattr(current_get_args_parser, _WRAPPER_MARKER, False):
        return True

    @wraps(current_get_args_parser)
    def get_args_parser():
        original_check_help = argparse.ArgumentParser._check_help
        argparse.ArgumentParser._check_help = lambda _self, _action: None
        try:
            return current_get_args_parser()
        finally:
            argparse.ArgumentParser._check_help = original_check_help

    setattr(get_args_parser, _WRAPPER_MARKER, True)
    hydra_main.get_args_parser = get_args_parser
    return True
