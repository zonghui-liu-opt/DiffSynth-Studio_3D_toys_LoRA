import argparse
import importlib.util
from pathlib import Path


def load_parsers_module():
    module_path = Path("diffsynth/diffusion/parsers.py").resolve()
    spec = importlib.util.spec_from_file_location("diffsynth_diffusion_parsers", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_general_parser_exposes_metrics_path():
    parsers = load_parsers_module()
    parser = parsers.add_general_config(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--dataset_base_path",
            "data",
            "--metrics_path",
            "metrics.jsonl",
        ]
    )

    assert args.metrics_path == "metrics.jsonl"
