from __future__ import annotations

from pathlib import Path

import click

from inferagni import __version__

output_option = click.option(
    "-o",
    "--output",
    "outdir",
    type=click.Path(exists=True, dir_okay=True, path_type=Path, resolve_path=True),
    help="Path to output folder",
    default=Path.cwd(),
)


@click.group()
@click.version_option(version=__version__)
def cli():
    pass


# ----------------
# 'plot' command
# ----------------


@click.command()
@output_option
@click.argument("zkey", nargs=1)
@click.argument("controls", nargs=-1)
def plot(outdir, zkey, controls):
    """Make mass-radius plot, given some control variables"""
    click.echo(f"Outdir:   {outdir}")
    click.echo(f"Zkey:     {zkey}")
    click.echo(f"Controls: {controls}")

    # convert control to dict
    controls_dict = {}
    for c in controls:
        k, v = c.split("=")
        controls_dict[str(k)] = float(v)

    from inferagni.grid import Grid
    from inferagni.plot import massrad_2d

    massrad_2d(Grid(), zkey, None, controls_dict)


cli.add_command(plot)

# ----------------
# 'infer' command
# ----------------


@click.command()
@output_option
@click.argument("obs", nargs=-1)
def infer(outdir, obs):
    """Infer parameters given some observables"""
    click.echo(f"Observables: {obs}")


cli.add_command(infer)

if __name__ == "__main__":
    cli()
