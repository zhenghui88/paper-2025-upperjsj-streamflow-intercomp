import argparse
import json
import tomllib
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import h5netcdf
import numpy as np
from numpy.typing import NDArray


def read_runoff(
    inputfile: Path, dtstart: datetime, dtstop: datetime
) -> Iterator[tuple[datetime, NDArray[np.float32]]]:
    with h5netcdf.File(inputfile, "r") as f:
        time_units = datetime.strptime(
            f.variables["time"].attrs["units"],
            "seconds since %Y-%m-%dT%H:%M:%SZ",
        )
        for itime, ctimenum in enumerate(f.variables["time"][:]):
            ctime = time_units + timedelta(seconds=int(ctimenum))
            if ctime > dtstop:
                return
            elif ctime >= dtstart:
                yield (
                    ctime,
                    np.array(
                        np.maximum(f.variables["mrro"][itime, :, :], 0.0),
                        np.float32,
                    ),
                )


def accumulate(
    mrro: NDArray[np.float32],
    points: Sequence[Sequence[tuple[int, int]]],
    areas: Sequence[Sequence[float]],
):
    result: list[float] = []
    for ps, ws in zip(points, areas):
        cumsum = 0.0
        cumarea = 0.0
        for (i, j), w in zip(ps, ws):
            if np.isfinite(mrro[i, j]):
                cumsum += float(mrro[i, j]) * w
                cumarea += w
        result.append(cumsum if cumarea > 0 else float("nan"))
    return result


def main(
    rivnetfile: Path,
    remapfile: Path,
    inputroot: Path,
    outputroot: Path,
    dtstart: datetime,
    dtstop: datetime,
):
    riverid: list[UUID] = []
    with open(rivnetfile, "rt") as f:
        for ll in f:
            data = json.loads(ll)
            riverid.append(UUID(data["properties"]["uuid"]))
    with open(remapfile, "rb") as f:
        remap = {UUID(k): v for k, v in tomllib.load(f).items()}
    points = [remap[x]["latlon_index"] for x in riverid]
    areas = [remap[x]["area"] for x in riverid]

    for cdt, mrro in read_runoff(inputroot, dtstart, dtstop):
        print(cdt, np.nanmax(mrro), np.nanmin(mrro), flush=True)
        result = accumulate(mrro, points, areas)

        outputfile = outputroot.joinpath(f"{cdt:%Y%m%dT%H%M%S}.csv")
        with open(outputfile, "wt") as f:
            f.write("riverid,qlat(m3 s-1)\n")
            for rid, x in zip(riverid, result):
                f.write(f"{rid.urn},{(x / RHO_WATER):.6e}\n")


if __name__ == "__main__":
    RHO_WATER = 1000.0

    parser = argparse.ArgumentParser()
    parser.add_argument("rivnetfile", type=Path)
    parser.add_argument("remapfile", type=Path, default=Path("weight.toml"))
    parser.add_argument("landoutfile", type=Path, default=Path("land"))
    parser.add_argument("outputroot", type=Path)
    parser.add_argument(
        "start",
        type=lambda x: datetime.strptime(x, "%Y%m%dT%H%M%S"),
        help='start time in "YYYYMMDDTHHMMSS" format, exclusive',
    )
    parser.add_argument(
        "stop",
        type=lambda x: datetime.strptime(x, "%Y%m%dT%H%M%S"),
        help='stop time in "YYYYMMDDTHHMMSS" format, inclusive',
    )
    args = parser.parse_args()
    rivnetfile = Path(args.rivnetfile)
    remapfile = Path(args.remapfile)
    inputfile = Path(args.landoutfile)
    outputroot = Path(args.outputroot)

    dtstart = cast(datetime, args.start)
    dtstop = cast(datetime, args.stop)

    main(rivnetfile, remapfile, inputfile, outputroot, dtstart, dtstop)
