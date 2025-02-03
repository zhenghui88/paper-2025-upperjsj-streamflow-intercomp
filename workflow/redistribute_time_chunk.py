import argparse
import itertools
from array import array
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, cast

import h5netcdf
import numpy as np
from numpy.typing import NDArray

EPSILON = np.float32(1e-8)

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


def read_fineweight(
    fineweightfile: Path,
) -> Iterator[tuple[datetime, NDArray[np.float32]]]:
    with h5netcdf.File(fineweightfile, "r") as f:
        time_units = datetime.strptime(
            f.variables["time"].attrs["units"], "seconds since %Y-%m-%dT%H:%M:%SZ"
        )
        for itime, timenum in enumerate(f.variables["time"][:]):
            yield (
                time_units + timedelta(seconds=int(timenum)),
                cast(NDArray, f.variables["mrro"][itime, :, :]).astype(np.float32),
            )


def read_coarse_grfr(
    coarsefile: Path,
) -> Iterator[tuple[datetime, NDArray[np.float32]]]:
    with h5netcdf.File(coarsefile, "r") as f:
        time_units = datetime.strptime(
            f.variables["time"].attrs["units"], "minutes since %Y-%m-%d %H:%M"
        )
        for itime, timenum in enumerate(f.variables["time"][:]):
            data = cast(NDArray, f.variables["ro"][itime, ...])
            data[data == f.variables["ro"].attrs["_FillValue"]] = np.nan
            data /= 3 * 3600
            yield (
                time_units + timedelta(minutes=int(timenum)),
                data.astype(np.float32),
            )


def read_coarse_glofas(
    coarsefile: Path,
) -> Iterator[tuple[datetime, NDArray[np.float32]]]:
    with h5netcdf.File(coarsefile, "r") as f:
        time_units = datetime.strptime(
            f.variables["valid_time"].attrs["units"], "seconds since %Y-%m-%d"
        )
        for itime, timenum in enumerate(f.variables["valid_time"][:]):
            data = cast(NDArray, f.variables["rowe"][itime, ...])
            data /= 24 * 3600
            yield (
                time_units + timedelta(seconds=int(timenum)),
                data.astype(np.float32),
            )


def match(
    fineiter: Iterable[tuple[datetime, NDArray[np.floating]]],
    coarseiter: Iterable[tuple[datetime, NDArray[np.floating]]],
    chunk: int = 3,
) -> Iterator[
    tuple[
        tuple[datetime, NDArray[np.floating]],
        tuple[tuple[datetime, NDArray[np.floating]], ...],
    ]
]:
    finechunkiter = itertools.batched(fineiter, chunk)
    while True:
        try:
            finedata = next(finechunkiter)
            coarsedata: tuple[datetime, NDArray[np.floating]] = next(coarseiter)
            while finedata[-1][0] < coarsedata[0]:
                finedata = next(finechunkiter)
            while finedata[-1][0] > coarsedata[0]:
                coarsedata = next(coarseiter)
            yield (coarsedata, finedata)
        except StopIteration:
            break


def redistribute(
    coarse: tuple[datetime, NDArray[np.floating]],
    fineweight: tuple[tuple[datetime, NDArray[np.floating]], ...],
):
    sum = np.zeros_like(fineweight[0][1])
    for _, data in fineweight:
        sum += data + EPSILON
    return ((dt, (data / sum) * len(fineweight) * coarse[1]) for dt, data in fineweight)


def main(
    inputfile: Path, dataname: str, weightfile: Path, chunk: int, outputfile: Path
):
    lat, lon = read_latlon(weightfile)
    define_headers(outputfile, lat, lon)
    if dataname == "grfr":
        coarsereader = read_coarse_grfr(inputfile)
    elif dataname == "glofas":
        coarsereader = read_coarse_glofas(inputfile)
    else:
        raise ValueError(f"Unknown data name: {dataname}")
    with h5netcdf.File(outputfile, "a") as f:
        for coarsedata, era5ldata in match(
            read_fineweight(weightfile), coarsereader, chunk=chunk
        ):
            for dt, data in redistribute(coarsedata, era5ldata):
                print(dt, coarsedata[0], flush=True)
                f.resize_dimension("time", f.dimensions["time"].size + 1)
                f.variables["time"][-1] = (dt - DATETIME_EPOCH).total_seconds()
                f.variables["mrro"][-1, :, :] = data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redistribute coarse-scale data")
    parser.add_argument("input", type=Path, help="Input coarse-scale data file")
    parser.add_argument("dataname", choices=["grfr", "glofas"], help="Data name")
    parser.add_argument("fineweight", type=Path, help="Fine-scale weighting data file")
    parser.add_argument(
        "chunk",
        type=int,
        help="Number of fine-scale data compared to one coarse-scale data",
    )
    parser.add_argument("output", type=Path, help="Output fine-scale data file")
    args = parser.parse_args()

    main(args.input, args.dataname, args.fineweight, args.chunk, args.output)
