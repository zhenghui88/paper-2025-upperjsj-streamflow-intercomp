import argparse
import json
import math
import tomllib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence, cast
from uuid import UUID

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit
def muskingum_cunge_cost(
    dt: float,
    h: float,
    qp: float,
    qinc: float,
    qinp: float,
    qlat: float,
    bounded_roughness: float,
    bounded_slope: float,
    length: float,
    bounded_side_slope: float,
    bounded_bottom_width: float,
) -> tuple[float, float]:
    hp = max(h, 0.0)
    width = bounded_bottom_width + 2.0 * bounded_side_slope * hp
    area = hp * (bounded_bottom_width + bounded_side_slope * hp)
    perimeter = bounded_bottom_width + 2.0 * hp * math.sqrt(
        1.0 + bounded_side_slope * bounded_side_slope
    )
    radius = area / perimeter

    velocity = (
        math.sqrt(bounded_slope) / bounded_roughness * math.pow(radius, 2.0 / 3.0)
    )

    celerity_multiplier = (
        5.0
        - 4.0
        * radius
        * math.sqrt(1.0 + bounded_side_slope * bounded_side_slope)
        / (bounded_bottom_width + 2.0 * bounded_side_slope * hp)
    ) / 3.0
    celerity = velocity * celerity_multiplier

    x = 0.5 - 0.5 * max(
        0.0, min(1.0, area / (width * bounded_slope * celerity_multiplier * length))
    )

    e = celerity * dt / length
    a = 1.0 - x + 0.5 * e
    b = x - 0.5 * e
    c = 1.0 - x - 0.5 * e
    d = x + 0.5 * e

    q = velocity * area
    return q, a * q + b * qinc - c * qp - d * qinp - e * qlat


@numba.njit
def muskingum_cunge_routing(
    qinc: float,
    qp: float,
    qinp: float,
    qlat: float,
    dt: float,
    length: float,
    roughness: float,
    slope: float,
    side_slope: float,
    bottom_width: float,
) -> tuple[float, float]:
    EPSILON = 1e-4
    EPSILON_H = 1e-4
    MAX_LOOP = 1000
    MAX_H = 1e3
    MIN_ROUGHNESS = 0.01
    MIN_SLOPE = 2e-5

    bounded_roughness = max(roughness, MIN_ROUGHNESS)
    bounded_slope = max(slope, MIN_SLOPE)
    bounded_bottom_width = max(bottom_width, 0.1)
    bounded_side_slope = max(side_slope, 0.0)

    hl = 0.0
    hh = 1.0
    while math.isfinite(hh):
        q, qf = muskingum_cunge_cost(
            dt,
            hh,
            qp,
            qinc,
            qinp,
            qlat,
            bounded_roughness,
            bounded_slope,
            length,
            bounded_side_slope,
            bounded_bottom_width,
        )
        if abs(qf) < EPSILON:
            return q, hh
        elif qf > 0.0:
            break
        if hh > MAX_H:
            print(f"Warning: The stage is too high ({hh} > {MAX_H}).")
            print(
                f"qp = {qp}, qinc = {qinc}, qinp = {qinp}, qlat = {qlat}, roughness = {bounded_roughness}, slope = {bounded_slope}, length = {length}, side_slope = {bounded_side_slope}, bottom_width = {bounded_bottom_width}"
            )
        hl = hh
        hh *= 2.0

    qret = 0.0
    hret = hh
    for iloop in range(MAX_LOOP):
        hc = 0.5 * (hl + hh)
        q, qf = muskingum_cunge_cost(
            dt,
            hc,
            qp,
            qinc,
            qinp,
            qlat,
            bounded_roughness,
            bounded_slope,
            length,
            bounded_side_slope,
            bounded_bottom_width,
        )
        qret = q
        hret = hc
        if abs(qf) < EPSILON or (hh - hl) < EPSILON_H:
            break
        elif qf > 0.0:
            hh = hc
        else:
            hl = hc
        if iloop == MAX_LOOP - 1:
            print("Error: Maximum loop reached.")
            print(
                f"qp = {qp}, qinc = {qinc}, qinp = {qinp}, qlat = {qlat}, roughness = {bounded_roughness}, slope = {bounded_slope}, length = {length}, side_slope = {bounded_side_slope}, bottom_width = {bounded_bottom_width}"
            )
    return qret, hret


@numba.njit
def routing_step(
    h: NDArray[np.floating],
    q: NDArray[np.floating],
    qin: NDArray[np.floating],
    qp: NDArray[np.floating],
    qinp: NDArray[np.floating],
    qlat: NDArray[np.floating],
    dt: float,
    to: NDArray[np.integer],
    length: NDArray[np.floating],
    roughness: NDArray[np.floating],
    slope: NDArray[np.floating],
    side_slope: NDArray[np.floating],
    bottom_width: NDArray[np.floating],
):
    qin[:] = 0.0
    for ii, it in enumerate(to):
        q[ii], h[ii] = muskingum_cunge_routing(
            qin[ii].item(),
            qp[ii].item(),
            qinp[ii].item(),
            qlat[ii].item(),
            dt,
            length[ii].item(),
            roughness[ii].item(),
            slope[ii].item(),
            side_slope[ii].item(),
            bottom_width[ii].item(),
        )
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
    length: NDArray[np.floating],
    slope: NDArray[np.floating],
    roughness: NDArray[np.floating],
    bottom_width: NDArray[np.floating],
    side_slope: NDArray[np.floating],
    inputroot: Path,
    outputroot: Path,
    start: datetime,
    stop: datetime,
    dt: timedelta,
    dtio: timedelta,
):
    h = np.zeros(len(riverid))
    q = np.zeros(len(to_list))
    qin = np.zeros(len(to_list))
    qp = np.zeros(len(to_list))
    qinp = np.zeros(len(to_list))
    qlat = np.zeros(len(to_list))

    # Initial condition
    pdtio = start
    if streamflow_exists(outputroot, pdtio):
        print(f"Initial condition exists under {outputroot}")
        qp[:] = streamflow_read(outputroot, pdtio, riverid)
        for ii, it in enumerate(to):
            if it >= 0:
                qinp[it] = qinp[it] + qp[ii]
    else:
        print(f"Initial condition does not exist under {outputroot}")
        qp[:] = 0.0

    # Integration
    cdtio = pdtio + dtio
    while cdtio <= stop:
        qlat[:] = streamflow_read(inputroot, cdtio, riverid)
        for istep in range(int(dtio // dt)):
            routing_step(
                h,
                q,
                qin,
                qp,
                qinp,
                qlat,
                dt.total_seconds(),
                to,
                length,
                roughness,
                slope,
                side_slope,
                bottom_width,
            )
            q, qp = qp, q
            qin, qinp = qinp, qin
        streamflow_save(outputroot, cdtio, riverid, list(q), list(h))
        pdtio = cdtio
        cdtio += dtio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="River routing using Muskingum-Cunge")
    parser.add_argument("rivnetfile", type=Path)
    parser.add_argument("parameterfile", type=Path, default=Path("config.toml"))
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
        "step",
        type=int,
        help="time step in seconds",
    )
    parser.add_argument(
        "iostep",
        type=int,
        help="input/output time step in seconds",
    )
    args = parser.parse_args()
    inputroot = Path(args.inputroot)
    outputroot = Path(args.outputroot)
    start = cast(datetime, args.start)
    stop = cast(datetime, args.stop)
    dtio = timedelta(seconds=args.iostep)
    dt = timedelta(seconds=args.step)
    assert dtio % dt == timedelta(
        0
    ), "input/output timestep must be a multiple of the model timestep."

    # Read river network
    riverid: list[UUID] = []
    order: list[int] = []
    to_list: list[int] = []
    length_list: list[float] = []
    slope_list: list[float] = []
    with open(args.rivnetfile, "rt") as f:
        for ll in f:
            data = json.loads(ll)
            to_value = data["properties"]["to"]
            if to_value is None:
                to_list.append(-1)
            else:
                to_list.append(int(to_value))
            riverid.append(UUID(data["properties"]["uuid"]))
            order.append(int(data["properties"]["strahler_order"]))
            length_list.append(float(data["properties"]["length"]))
            slope_list.append(float(data["properties"]["slope"]))
    to = np.array(to_list, np.int32)
    length = np.array(length_list, np.float32)
    slope = np.array(slope_list, np.float32)

    # Read parameters
    with open(args.parameterfile, "rb") as f:
        config = tomllib.load(f)
        roughness_table = [float(x) for x in config["river"]["parameter"]["roughness"]]
        bottom_width_table = [
            float(x) for x in config["river"]["parameter"]["bottom_width"]
        ]
        side_slope_table = [
            float(x) for x in config["river"]["parameter"]["side_slope"]
        ]
    roughness = np.array([roughness_table[i - 1] for i in order], np.float32)
    bottom_width = np.array([bottom_width_table[i - 1] for i in order], np.float32)
    side_slope = np.array([side_slope_table[i - 1] for i in order], np.float32)

    main(
        riverid,
        to,
        length,
        slope,
        roughness,
        bottom_width,
        side_slope,
        inputroot,
        outputroot,
        start,
        stop,
        dt,
        dtio,
    )
