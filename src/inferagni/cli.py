from __future__ import annotations

from pathlib import Path

import os
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

    click.echo("Plot mass-radius 2D")
    click.echo(f"Outdir:   {outdir}")
    click.echo(f"Zkey:     {zkey}")
    click.echo(f"Controls: {controls}")
    click.echo(" ")

    # convert control to dict
    controls_dict = {}
    for c in controls:
        k, v = c.split("=")
        controls_dict[str(k)] = float(v)

    from inferagni.grid import Grid
    from inferagni.plot import massrad_2d

    massrad_2d(Grid(emits=False, profs=False),
               key1=zkey, key2=None, controls=controls_dict,
               save=os.path.join(outdir,"massrad_2d.pdf"),
               show=True)

cli.add_command(plot)

# ----------------
# 'infer' command
# ----------------


@click.command()
@output_option
@click.argument("planet_name", nargs=1)
@click.argument("quantities", nargs=-1)
@click.option("--steps", type=int)
def infer(outdir:str, planet_name:str, quantities:list, steps:int=None):
    """Infer some quantities for a named planet"""

    click.echo(f"Planet: {planet_name}")
    click.echo(f"Quantities: {quantities}")
    click.echo(" ")

    from inferagni.grid import Grid
    from inferagni.retrieve import run, plot_corner, plot_chain
    from inferagni.planets import get_obs

    # run retrieval
    gr = Grid(emits=False, profs=False)
    keys, samples = run(gr, get_obs(planet_name), extra_keys=quantities, n_steps=steps)

    # make plots
    plot_chain(samples,save=os.path.join(outdir,"retrieve_chain.pdf"),show=False)
    plot_corner(keys, samples,save=os.path.join(outdir,"retrieve_corner.pdf"),show=True)

cli.add_command(infer)

if __name__ == "__main__":
    cli()
