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


@click.command()
@click.argument("planet_name", nargs=1)
def planet(planet_name: str):
    """Get observed parameters for a named planet"""

    click.echo(f"Looking up '{planet_name}'")

    from inferagni.planets import get_obs
    get_obs(planet_name)

cli.add_command(planet)

# ----------------
# 'downloader' commands
# ----------------

@click.command()
@click.argument("gridname", nargs=1, default=None)
@click.option("--force", is_flag=True, help="Force update even if not needed")
def update(gridname: str|None, force: bool):
    """Download required grid data"""

    from inferagni.data import download_grid, check_grid_needs_update

    if not gridname:
        from inferagni.grid import DEFAULT_NAME
        gridname = DEFAULT_NAME

    # check if update is needed
    if not force and not check_grid_needs_update(gridname):
        click.echo("Grid data is up to date. No update needed.")
        return

    # download data
    if download_grid(gridname):
        click.echo("Grid data updated successfully.")
    else:
        click.echo("Failed to update grid data.")

cli.add_command(update)


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

    gr = Grid(emits=False, profs=False)
    if gr.data is None:
        click.echo("Grid data could not be loaded. Please run 'update' command first.")
        return

    massrad_2d(
        gr,
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
@click.option("--gridname", type=str, default=None)
def retrieve(
    outdir: str, planet_name: str, quantities: list,
    steps=None, walkers=None, procs=None, gridname=None
):
    """Infer some quantities for a named planet

    Parameters
    ----------
    planet_name : str
        Name of planet to retrieve for. Must be in the database (see 'planet' command).
    quantities : list of str
        List of quantities to retrieve. Must be in the grid (see 'listvars' command).
    steps : int, optional
        Number of MCMC steps to run.
    walkers : int, optional
        Number of MCMC walkers to use.
    procs : int, optional
        Number of processes to use for parallelisation.
    gridname : str, optional
        Name of grid to use. If not specified, the default grid will be used.
    """

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

    # load grid
    gr = Grid(gridname=gridname, emits=False, profs=False)

    # run retrieval
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
