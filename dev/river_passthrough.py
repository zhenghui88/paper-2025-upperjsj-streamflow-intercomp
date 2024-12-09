import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence, cast
from uuid import UUID

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit
def routing_step(
    q: NDArray[np.floating],
    qin: NDArray[np.floating],
    qlat: NDArray[np.floating],
    to: NDArray[np.integer],
):
    qin[:] = 0.0
    for ii, it in enumerate(to):
        q[ii] = qin[ii] + qlat[ii]
        if it >= 0:
            qin[it] = qin[it] + q[ii]


def streamflow_exists(root: Path, dt: datetime) -> bool:
    return root.joinpath(f"{dt:%Y%m%dT%H%M%S}.csv").exists()


def streamflow_db_read(root: Path, dt: datetime) -> dict[UUID, float]:
    filepath = root.joinpath(f"{dt:%Y%m%dT%H%M%S}.csv")
    data: dict[UUID, float] = {}
    with open(filepath, "rt") as f:
        next(f)
        for line in f:
            rid, q = line.strip().split(",")
            data[UUID(rid)] = float(q)
    return data


def streamflow_read(root: Path, dt: datetime, riverid: Sequence[UUID]) -> list[float]:
    data = streamflow_db_read(root, dt)
    return [data[rid] if not math.isnan(data[rid]) else 0.0 for rid in riverid]


def streamflow_save(
    root: Path,
    dt: datetime,
    riverid: Sequence[UUID],
    streamflow: Sequence[float],
    stage: Sequence[float],
):
    filepath = root.joinpath(f"{dt:%Y%m%dT%H%M%S}.csv")
    with open(filepath, "wt") as f:
        f.write("code,q(m3 s-1),stage(m)\n")
        for rid, q, h in zip(riverid, streamflow, stage):
            f.write(f"{rid.urn},{q:.6e},{h:.6e}\n")


def main(
    riverid: Sequence[UUID],
    to: NDArray[np.integer],
    inputroot: Path,
    outputroot: Path,
    tmstart: datetime,
    tmstop: datetime,
    dtio: timedelta,
):
    h = np.zeros(len(riverid))
    q = np.zeros(len(to_list))
    qin = np.zeros(len(to_list))
    qlat = np.zeros(len(to_list))

    # Integration
    cdtio = tmstart + dtio
    while cdtio <= tmstop:
        print(cdtio)
        qlat[:] = streamflow_read(inputroot, cdtio, riverid)
        routing_step(
            q,
            qin,
            qlat,
            to,
        )
        streamflow_save(outputroot, cdtio, riverid, list(q), list(h))
        cdtio += dtio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="River passthrough model")
    parser.add_argument("rivnetfile", type=Path)
    parser.add_argument("inputroot", type=Path, default=Path("land"))
    parser.add_argument("outputroot", type=Path)
    parser.add_argument(
        "start",
        type=lambda x: datetime.strptime(x, "%Y%m%dT%H%M%S"),
        help='start time in "YYYYMMDDTHHMMSS" format',
    )
    parser.add_argument(
        "stop",
        type=lambda x: datetime.strptime(x, "%Y%m%dT%H%M%S"),
        help='stop time in "YYYYMMDDTHHMMSS" format',
    )
    parser.add_argument(
        "iostep",
        type=int,
        help="input/output time step in seconds",
    )
    args = parser.parse_args()
    inputroot = Path(args.inputroot)
    outputroot = Path(args.outputroot)
    tmstart = cast(datetime, args.start)
    tmstop = cast(datetime, args.stop)
    dtio = timedelta(seconds=args.iostep)

    # Read river network
    riverid: list[UUID] = []
    to_list: list[int] = []
    with open(args.rivnetfile, "rt") as f:
        for ll in f:
            data = json.loads(ll)
            to_value = data["properties"]["to"]
            if to_value is None:
                to_list.append(-1)
            else:
                to_list.append(int(to_value))
            riverid.append(UUID(data["properties"]["uuid"]))
    to = np.array(to_list, np.int32)

    main(
        riverid,
        to,
        inputroot,
        outputroot,
        tmstart,
        tmstop,
        dtio,
    )
