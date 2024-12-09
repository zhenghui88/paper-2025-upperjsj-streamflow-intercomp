import argparse
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
            "mrro",
            ("time", "lat", "lon"),
            dtype=np.float32,
            fillvalue=np.float32(np.nan),
            compression="gzip",
        ).attrs.update(
            {
                "units": np.bytes_("kg m-2 s-1", "ascii"),
                "standard_name": np.bytes_("runoff_flux", "ascii"),
            }
        )


def read_latlon(ncfile: Path):
    with h5netcdf.File(ncfile, "r") as f:
        lat = array("d", f.variables["lat"][:])
        lon = array("d", f.variables["lon"][:])
    return lat, lon


def read_runoff(ncfiles: Sequence[Path]) -> Iterator[tuple[datetime, NDArray]]:
    for ncfile in sorted(ncfiles):
        with h5netcdf.File(ncfile, "r") as f:
            time_units = datetime.strptime(
                f.variables["time"].attrs["units"],
                "seconds since %Y-%m-%dT%H:%M:%SZ",
            )
            time = array("d", f.variables["time"][:])
            for ii in range(len(time)):
                yield (
                    time_units + timedelta(seconds=time[ii]),
                    cast(
                        NDArray,
                        np.maximum(f.variables["mrros"][ii, :, :], 0.0)
                        + np.maximum(f.variables["mrrob"][ii, :, :], 0.0),
                    ).astype(np.float32),
                )


def main(inputfile: Path, outputfile: Path):
    lat, lon = read_latlon(inputfile)
    define_headers(outputfile, lat, lon)
    with h5netcdf.File(outputfile, "a") as f:
        for dt, ro in read_runoff([inputfile]):
            print(dt, flush=True)
            f.resize_dimension("time", f.dimensions["time"].size + 1)
            f.variables["time"][-1] = (dt - DATETIME_EPOCH).total_seconds()
            f.variables["mrro"][-1, ...] = ro


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert NoahMP accumulated data into flux data"
    )
    parser.add_argument("input", type=Path, help="Input NoahMP data file")
    parser.add_argument("output", type=Path, help="Output NoahMP data file")
    args = parser.parse_args()

    main(args.input, args.output)
