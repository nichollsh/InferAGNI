from __future__ import annotations

import os
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
# 'helper' commands
# ----------------


@click.command()
def listvars():
    """List vars in grid"""

    from inferagni.grid import Grid

    Grid(emits=False, profs=False)  # this will print information about the vars


cli.add_command(listvars)

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

    massrad_2d(
        Grid(emits=False, profs=False),
        key1=zkey,
        key2=None,
        controls=controls_dict,
        save=os.path.join(outdir, "massrad_2d.pdf"),
        show=True,
    )


cli.add_command(plot)

# ----------------
# 'retrieve' command
# ----------------


@click.command()
@output_option
@click.argument("planet_name", nargs=1)
@click.argument("quantities", nargs=-1)
@click.option("--steps", type=int)
@click.option("--walkers", type=int)
@click.option("--procs", type=int)
def retrieve(
    outdir: str, planet_name: str, quantities: list, steps=None, walkers=None, procs=None
):
    """Infer some quantities for a named planet"""

    click.echo(f"Planet: {planet_name}")
    click.echo(f"Quantities: {quantities}")
    click.echo(" ")

    from inferagni.grid import Grid
    from inferagni.planets import get_obs
    from inferagni.retrieve import plot_chain, plot_corner, run
    from inferagni.util import print_sep_min

    obs = get_obs(planet_name)
    if not obs:
        return

    click.echo(print_sep_min)
    click.echo("")

    # run retrieval
    gr = Grid(emits=False, profs=False)
    keys, samples = run(
        gr,
        obs,
        extra_keys=quantities,
        n_steps=steps,
        n_walkers=walkers,
        n_procs=procs,
    )

    # make plots
    plot_chain(samples, save=os.path.join(outdir, "retrieve_chain.pdf"), show=False)
    plot_corner(keys, samples, save=os.path.join(outdir, "retrieve_corner.pdf"), show=True)


cli.add_command(retrieve)

if __name__ == "__main__":
    cli()
