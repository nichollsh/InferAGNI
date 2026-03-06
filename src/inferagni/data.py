from __future__ import annotations

import hashlib
import os
import shutil
import zipfile

import requests

griddata_dir = os.path.join(os.path.dirname(__file__), "data")

ZENODO_RECORD = "18790976"

# Expected files
EXPECTED_FILES = (
    "base.toml",
    "md5sums.txt",
    "consolidated_emits.csv",
    "consolidated_profs.nc",
    "gridpoints.csv",
    "consolidated_table.csv",
)

# available grid names
AVAILABLE_GRIDS = ("grid_018", "grid_022", "grid_023")

# default grid name
DEFAULT_GRID = "grid_022"


def check_grid_name(gridname: str) -> bool:
    """Check if the provided grid name is valid.

    Parameters
    ----------
    gridname : str, Name of the grid to be checked.

    Returns
    -------
    valid : bool, Is the grid name valid.
    """

    if gridname not in AVAILABLE_GRIDS:
        print(f"Invalid grid name '{gridname}'")
        print(f"    available: {', '.join(AVAILABLE_GRIDS)}")
        return False

    return True


def check_grid_needs_update(gridname: str) -> bool:
    """Check if the grid data needs to be updated.

    Update is required if a file is missing or its md5sum is invalid.

    Parameters
    ----------
    gridname : str, Name of the grid folder to be checked.

    Returns
    -------
    update : bool, Needs updating.
    """

    if not check_grid_name(gridname):
        return False

    print(f"Checking if grid '{gridname}' needs to be updated...")

    # Folder missing
    fpath = os.path.join(griddata_dir, gridname)
    if not os.path.isdir(fpath):
        print(f"    folder '{fpath}' is missing.")
        print("    update needed: True")
        return True

    # Does md5sum.chk exist?
    if not os.path.isfile(os.path.join(fpath, "md5sums.txt")):
        print("    md5sum.chk is missing")
        print("    update needed: True")
        return True

    # Read expected md5sums
    expected_md5sums = {}
    with open(os.path.join(fpath, "md5sums.txt"), "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            md5sum, fname = line.split()
            expected_md5sums[fname] = md5sum

    # Check each file
    for f in EXPECTED_FILES:
        # Don't check md5sums.txt itself
        if f == "md5sums.txt":
            continue

        # Construct file path
        fpath = os.path.join(griddata_dir, gridname, f)

        # Does it exist?
        if not os.path.isfile(fpath):
            print(f"    file '{f}' is missing")
            print("    update needed: True")
            return True

        # If it exists, calculate md5 and compare to expected
        with open(fpath, "rb") as hdl:
            data = hdl.read()
            md5sum = hashlib.md5(data).hexdigest()
            if f not in expected_md5sums:
                print(f"    file '{f}' not listed in md5sums.txt")
                continue

            if md5sum != expected_md5sums[f]:
                print(f"    file '{f}' has invalid md5 checksum")
                print(f"        expected {expected_md5sums[f]}   Got {md5sum}")
                print("    update needed: True")
                return True

    return False


def request_filesize(url) -> int | None:
    """Request the filesize of a file at a URL.

    Parameters
    ----------
    url : str, URL of the file.

    Returns
    -------
    filesize : int, Filesize in MB, or None if not available.
    """

    try:
        response = requests.head(url)
        response.raise_for_status()
        if "Content-Length" in response.headers:
            return int(response.headers["Content-Length"]) / 1e6
        else:
            print(f"Cannot get information about '{url}'")
            return None

    except requests.exceptions.RequestException as e:
        print(e)
        return None


def download_file(url, destination):
    """Download a file from a URL to a local destination."""
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=2048):
                    f.write(chunk)

    except requests.exceptions.RequestException as e:
        print(e)
        return False

    return os.path.isfile(destination)


def download_grid(gridname: str) -> bool:
    """Download the grid data for the specified grid.

    Parameters
    ----------
    gridname : str, Name of the grid folder to be downloaded.

    Returns
    -------
    success : bool, Download successful.
    """

    if not check_grid_name(gridname):
        return False

    print(f"Obtaining data for '{gridname}' from Zenodo...")

    # Remove old folder, make new
    shutil.rmtree(os.path.join(griddata_dir, gridname), ignore_errors=True)
    os.makedirs(os.path.join(griddata_dir, gridname), exist_ok=True)

    # Construct destination file path
    f = f"{gridname}.zip"
    fpath = os.path.join(griddata_dir, gridname, f)

    # Construct url for zip on Zenodo
    url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{f}?download=1"

    # Check file size
    filesize = request_filesize(url)
    if filesize is not None:
        print(f"    expected file size: {filesize:.1f} MB")

    # Download file
    if not download_file(url, fpath):
        print(f"        failed to download '{f}' from '{url}'")
        return False

    # Unzip the file
    with zipfile.ZipFile(fpath, "r") as zip_ref:
        zip_ref.extractall(os.path.join(griddata_dir, gridname))

    # Remove the zip file
    os.remove(fpath)

    # Validate grid folder
    if check_grid_needs_update(gridname):
        print(f"    download finished, but data for '{gridname}' is invalid")
        return False

    print("    done")
    return True
