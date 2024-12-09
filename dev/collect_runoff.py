import argparse
import itertools
from array import array
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Sequence, cast

import h5netcdf
import numpy as np
from numpy.typing import NDArray

DATETIME_EPOCH = datetime(1970, 1, 1)


def define_headers(ncfile: Path, lat: array[float], lon: array[float]):
    with h5netcdf.File(ncfile, "w") as f:
        f.dimensions.update({"time": None, "lat": len(lat), "lon": len(lon)})
        f.create_variable(
            "time", ("time",), dtype=np.int64, compression="gzip"
        ).attrs.update(
            {
                "units": np.bytes_(
                    f"seconds since {DATETIME_EPOCH.isoformat(sep='T')}Z", "ascii"
                ),
                "calendar": "standard",
            }
        )
        f.create_variable(
            "lat", ("lat",), dtype=np.float64, data=lat, compression="gzip"
        ).attrs.update(
            {
                "units": np.bytes_("degrees_north", "ascii"),
                "standard_name": np.bytes_("latitude", "ascii"),
            }
        )
        f.create_variable(
            "lon", ("lon",), dtype=np.float64, data=lon, compression="gzip"
        ).attrs.update(
            {
                "units": np.bytes_("degrees_east", "ascii"),
                "standard_name": np.bytes_("longitude", "ascii"),
            }
        )
        f.create_variable(
            "mrros",
            ("time", "lat", "lon"),
            dtype=np.float32,
            fillvalue=np.float32(np.nan),
            compression="gzip",
        ).attrs.update(
            {
                "units": np.bytes_("kg m-2 s-1", "ascii"),
                "standard_name": np.bytes_("surface_runoff_flux", "ascii"),
            }
        )
        f.create_variable(
            "mrrob",
            ("time", "lat", "lon"),
            dtype=np.float32,
            fillvalue=np.float32(np.nan),
            compression="gzip",
        ).attrs.update(
            {
                "units": np.bytes_("kg m-2 s-1", "ascii"),
                "standard_name": np.bytes_("subsurface_runoff_flux", "ascii"),
            }
        )


def read_latlon(ncfile: Path):
    with h5netcdf.File(ncfile, "r") as f:
        lat = array("d", f.variables["south_north"][:])
        lon = array("d", f.variables["west_east"][:])
    return lat, lon


def read_runoff(ncfiles: Sequence[Path]) -> Iterator[tuple[datetime, NDArray, NDArray]]:
    for ncfile in ncfiles:
        dt = datetime.strptime(ncfile.stem.split(".")[-1], "%Y%m%dT%H%M%S")
        with h5netcdf.File(ncfile, "r") as f:
            rsfc = cast(NDArray, f.variables["SFCRNOFF"][:]).astype(np.float32)
            rsub = cast(NDArray, f.variables["UGDRNOFF"][:]).astype(np.float32)
            yield dt, rsfc, rsub


def main(setupfile: Path, inputroot: Path, outputfile: Path):
    lat, lon = read_latlon(setupfile)
    define_headers(outputfile, lat, lon)
    ncfiles = sorted(inputroot.glob("ldasout*.nc"))
    with h5netcdf.File(outputfile, "a") as f:
        for (dt1, rsfc1, rsub1), (dt2, rsfc2, rsub2) in itertools.pairwise(
            read_runoff(ncfiles)
        ):
            assert (dt2 - dt1) == timedelta(
                seconds=3600
            ), f"Time interval between {dt2} and {dt1} is not 1 hour"
            print(dt2.isoformat(sep="T"), flush=True)
            f.resize_dimension("time", f.dimensions["time"].size + 1)
            f.variables["time"][-1] = (dt2 - DATETIME_EPOCH).total_seconds()
            np.where(
                np.logical_and(np.isfinite(rsfc2), rsfc2 > rsfc1), rsfc2 - rsfc1, 0.0
            )
            f.variables["mrros"][-1, :, :] = (
                np.where(
                    np.logical_and(np.isfinite(rsfc2), rsfc2 > rsfc1),
                    rsfc2 - rsfc1,
                    0.0,
                )
                / (dt2 - dt1).total_seconds()
            )
            f.variables["mrrob"][-1, :, :] = (
                np.where(
                    np.logical_and(np.isfinite(rsub2), rsub2 > rsub1),
                    rsub2 - rsub1,
                    0.0,
                )
                / (dt2 - dt1).total_seconds()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect runoff data from NoahMP output"
    )
    parser.add_argument("setupfile", type=Path, help="Setup file")
    parser.add_argument(
        "inputroot", type=Path, help="Root directory of the NoahMP output"
    )
    parser.add_argument("outputfile", type=Path, help="Output file")
    args = parser.parse_args()

    main(args.setupfile, args.inputroot, args.outputfile)
