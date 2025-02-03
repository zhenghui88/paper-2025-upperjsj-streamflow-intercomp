#!/usr/bin/env python3
# coding: utf-8

from array import array
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Sequence

import netCDF4 as nc
import numpy as np
from numpy.typing import NDArray

DATETIME_EPOCH = datetime(1970, 1, 1)


def glob_files(srcroot: Path, start: datetime, end: datetime):
    ret: list[Path] = []
    for filepath in srcroot.glob("**/*.nc"):
        if filepath.stem.isnumeric():
            fdate = datetime.strptime(filepath.stem, "%Y%m%d")
            if start <= fdate <= end:
                ret.append(filepath)
    return sorted(ret)


def define_output_file(desfile: Path, lat: array[float], lon: array[float]):
    with nc.Dataset(desfile, "w") as f:
        f.createDimension("time", None)
        f.createDimension(
            "latitude",
            len(lat),
        )
        f.createDimension("nbnds", 2)
        f.createDimension("longitude", len(lon))
        f.createVariable(
            "time",
            np.int64,
            ("time",),
            compression="zlib",
        )
        f.variables["time"].setncattr(
            "units",
            f"seconds since {DATETIME_EPOCH.isoformat(sep='T', timespec='seconds')}Z",
        )
        f.variables["time"].setncattr("time_bnds", "time_bnds")
        f.createVariable("time_bnds", np.int64, ("time", "nbnds"), compression="zlib")
        f.variables["time_bnds"].setncattr(
            "units",
            f"seconds since {DATETIME_EPOCH.isoformat(sep='T', timespec='seconds')}Z",
        )
        f.createVariable(
            "latitude",
            np.float64,
            ("latitude",),
            compression="zlib",
        )
        f.variables["latitude"].setncattr("units", "degrees_north")
        f.variables["latitude"].setncattr("standard_name", "latitude")
        f.createVariable(
            "longitude",
            np.float64,
            ("longitude",),
            compression="zlib",
        )
        f.variables["longitude"].setncattr("units", "degrees_east")
        f.variables["longitude"].setncattr("standard_name", "longitude")
        f.createVariable(
            "mrro",
            np.float32,
            ("time", "latitude", "longitude"),
            fill_value=np.float32(np.nan),
            compression="zlib",
            chunksizes=(1, len(lat), len(lon)),
        )
        f.variables["mrro"].setncattr("units", "kg m-2 s-1")
        f.variables["mrro"].setncattr("standard_name", "runoff_flux")
        f.variables["mrro"].setncattr("cell_methods", "time:mean")
        f.variables["latitude"][:] = lat
        f.variables["longitude"][:] = lon


def read_lat_lon(filepath: Path):
    with nc.Dataset(filepath, "r") as f:
        lat = array("d", f.variables["latitude"][:])
        lon = array("d", f.variables["longitude"][:])
    return lat, lon


type RunoffRecord = tuple[datetime, NDArray[np.float32]]


def read_runoff_pair(
    files: Sequence[Path],
) -> Iterator[tuple[RunoffRecord, RunoffRecord]]:
    RHO_WATER = 1000.0
    ptime = None
    ctime = None
    pdata = None
    cdata = None
    for ff in files:
        with nc.Dataset(ff, "r") as f:
            if cdata is None:
                cdata = np.full(f.variables["ro"].shape[1:], np.nan, np.float32)
                pdata = np.full(f.variables["ro"].shape[1:], np.nan, np.float32)
            datetime_ref = datetime.strptime(
                f.variables["valid_time"].units,
                "seconds since %Y-%m-%d",
            )
            for index in range(len(f.variables["valid_time"])):  # type: ignore
                ctime = datetime_ref + timedelta(
                    seconds=int(f.variables["valid_time"][index])
                )
                assert cdata is not None
                assert pdata is not None
                cdata[:] = f.variables["ro"][index, ...] * RHO_WATER
                if ptime is not None:
                    yield (ptime, pdata), (ctime, cdata)
                ptime = ctime
                pdata, cdata = cdata, pdata


def main(srcroot: Path, start: datetime, end: datetime, desfile: Path):
    ncfiles = glob_files(srcroot, start, end)
    if not desfile.exists():
        lat, lon = read_lat_lon(ncfiles[0])
        define_output_file(desfile, lat, lon)
    with nc.Dataset(desfile, "a") as f:
        for (ptime, pdata), (ctime, cdata) in read_runoff_pair(ncfiles):
            if ctime <= start:
                continue
            if ctime > end:
                break
            print(ptime, ctime)
            index = len(f.variables["time"])
            f.variables["time"][index] = (ctime - DATETIME_EPOCH).total_seconds()
            f.variables["time_bnds"][index, :] = (
                (ptime - DATETIME_EPOCH).total_seconds(),
                (ctime - DATETIME_EPOCH).total_seconds(),
            )
            if ctime.hour == 1:
                f.variables["mrro"][index, ...] = (
                    cdata / (ctime - ptime).total_seconds()
                )
            else:
                f.variables["mrro"][index, :, :] = (cdata - pdata) / (
                    ctime - ptime
                ).total_seconds()
            f.sync()


if __name__ == "__main__":
    DATETIME_START = datetime(2000, 1, 1)
    DATETIME_END = datetime(2020, 1, 1)
    SRCDATA_ROOT = Path("data", "era5l")
    DESDATA_FILE = Path("data", "runoff", "era5l.nc")
    DESDATA_FILE.unlink(missing_ok=True)
    main(SRCDATA_ROOT, DATETIME_START, DATETIME_END, DESDATA_FILE)
