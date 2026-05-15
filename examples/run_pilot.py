"""Smallest possible programmatic entry point.

    python examples/run_pilot.py
"""
from __future__ import annotations

from pathlib import Path

from attrforge import AttrForge
from attrforge.loop import AttrForgeConfig, configure_logging


def main() -> None:
    configure_logging("INFO")
    cfg_path = Path(__file__).parent / "customer_support" / "config.echo.yaml"
    forge = AttrForge(AttrForgeConfig.from_yaml(cfg_path))
    result = forge.run()
    print(f"\nRun dir: {result.run_dir}")
    print(f"Final prompt v{result.final_prompt_version}:\n")
    print(result.final_prompt)


if __name__ == "__main__":
    main()
