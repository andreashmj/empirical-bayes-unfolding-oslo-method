"""
Commands to create the methods figures. 
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    
    commands = parser.add_subparsers(dest="command", required=True)
    
    flow = commands.add_parser("flowchart")
    
    generation = commands.add_parser("data-generation")
    generation.add_argument("--config", dest="config_path", type=Path, required=True)
    generation.add_argument("--ex-values", nargs=2, type=float, default=(4000, 8400))
    
    draws = commands.add_parser("ill-posedness")
    draws.add_argument("nc_path", type=Path)
    draws.add_argument("--ex", type=float)
    draws.add_argument("--max-draws", type=int, default=10)
    draws.add_argument("--seed", type=int, default=123)

    
    diagnostics = commands.add_parser("diagnostics")
    diagnostics.add_argument("nc_path", type=Path)
    diagnostics.add_argument("--ex", type=float)
    diagnostics.add_argument("--var", choices=("x", "eta"), default="eta")
    
    for command in (flow, generation, draws, diagnostics):
        command.add_argument("--out", type=Path)
        command.add_argument("--no-show", dest="show", action="store_false")
        
    return parser


def main(argv: list[str] | None = None) -> None:
    args = vars(build_parser().parse_args(argv))
    command = args.pop("command")
    if command == "flowchart":
        from .flowchart import plot_flowchart as plot
    elif command == "data-generation":
        from .data_generation import plot_data_generation as plot
    elif command == "ill-posedness":
        from .ill_posedness import plot_ill_posedness as plot
    else:
        from .diagnostics import plot_diagnostics as plot
    plot(**args)


if __name__ == "__main__":
    main()
