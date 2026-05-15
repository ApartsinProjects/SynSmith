"""AttrForge command line.

Usage::

    attrforge run examples/customer_support/config.yaml --iterations 5
    attrforge inspect runs/<run_id>
    attrforge schema examples/customer_support/schema.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from attrforge.loop import AttrForge, AttrForgeConfig, configure_logging
from attrforge.schema import AttributeSchema

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, help="YAML config path."),
    iterations: int | None = typer.Option(
        None, "--iterations", "-T", help="Override the config's iteration count."
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run the iterative AttrForge loop from a config file."""
    configure_logging(log_level)
    forge = AttrForge(AttrForgeConfig.from_yaml(config))
    result = forge.run(iterations=iterations)
    console.print(
        f"\n[bold green]done.[/bold green] run dir: {result.run_dir}, "
        f"final prompt version: v{result.final_prompt_version}"
    )


@app.command()
def inspect(run_dir: Path = typer.Argument(..., exists=True)) -> None:
    """Pretty-print metric history from a finished run."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"manifest.json not found in {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history = manifest.get("metric_history", [])

    if not history:
        console.print("[yellow]no metric history found[/yellow]")
        return

    table = Table(title=f"Metric history for {run_dir.name}")
    keys = ["iteration"] + sorted({k for row in history for k in row if k != "iteration"})
    for k in keys:
        table.add_column(k, justify="right")
    for row in history:
        table.add_row(*[f"{row.get(k, 0):.3f}" if k != "iteration" else f"{int(row[k])}" for k in keys])
    console.print(table)

    console.rule("[bold]Prompt history")
    for v in manifest.get("prompt_history", []):
        console.print(f"[bold cyan]v{v['version']}[/bold cyan] @ iter {v['iteration']} ({v['motivation']})")
        console.print(v["prompt"])
        console.print()


@app.command()
def schema(path: Path = typer.Argument(..., exists=True)) -> None:
    """Validate an attribute schema YAML and print its shape."""
    s = AttributeSchema.from_yaml(path)
    table = Table(title=f"Schema: {path.name}")
    table.add_column("attribute", style="bold")
    table.add_column("# values", justify="right")
    table.add_column("values")
    for name, values in s.attributes.items():
        table.add_row(name, str(len(values)), ", ".join(values))
    console.print(table)
    console.print(f"label attribute: [bold]{s.label_attribute}[/bold]")
    console.print(f"domain: {s.domain}")
    if s.invalid_combinations:
        console.print(f"invalid combinations: {s.invalid_combinations}")


if __name__ == "__main__":
    app()
