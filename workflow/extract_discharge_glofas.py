#!/usr/bin/env python3
# coding: utf-8

import argparse
from array import array
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, cast
from uuid import UUID

import h5netcdf
import polars as pl


def read_stations(filepath: Path) -> dict[UUID, tuple[float, float]]:
    stations: dict[str, tuple[float, float]] = {}
    with open(filepath, "rt") as f:
        f.readline()
        for line in f:
            sid, _name, _merit, _rid, lat, lon = line.strip().split(",")
            stations[UUID(sid)] = (float(lat), float(lon))
    return stations


def find_grid_index(
    lats: array[float],
    lons: array[float],
    stations: dict[UUID, tuple[float, float]],
) -> dict[UUID, tuple[int, int]]:
    ret: dict[UUID, tuple[int, int]] = {}
    for sid, (lat, lon) in stations.items():
        latdiffmin = float("inf")
        latidx = 0
        for ii, ll in enumerate(lats):
            latdiff = abs(ll - lat)
            if latdiff < latdiffmin:
                latdiffmin = latdiff
                latidx = ii
        londiffmin = float("inf")
        lonidx = 0
        for ii, ll in enumerate(lons):
            londiff = abs(ll - lon)
            if londiff < londiffmin:
                londiffmin = londiff
                lonidx = ii
        ret[sid] = (latidx, lonidx)
    return ret


def read_discharge(
    filepaths: Iterable[Path],
    stations: dict[UUID, tuple[float, float]],
    dtbegin: datetime,
    dtstop: datetime,
) -> Iterator[tuple[datetime, dict[UUID, float]]]:
    for filepath in filepaths:
        with h5netcdf.File(filepath, "r") as f:
            tm_units = datetime.strptime(
                f.variables["valid_time"].attrs["units"], "seconds since %Y-%m-%d"
            )
            tm = [
                tm_units + timedelta(seconds=int(x))
                for x in f.variables["valid_time"][:]
            ]
            if tm[0] > dtstop or tm[-1] < dtbegin:
                continue
            lats = array("d", f.variables["latitude"][:])
            lons = array("d", f.variables["longitude"][:])
            gridindex = find_grid_index(lats, lons, stations)
            for ii, tt in enumerate(tm):
                if tt < dtbegin or tt > dtstop:
                    continue
                yield (
                    tt,
                    {
                        sid: f.variables["dis24"][ii, loc[0], loc[1]]
                        for sid, loc in gridindex.items()
                    },
                )


def main(
    srcroot: Path, stationfile: Path, desfile: Path, dtbeing: datetime, dtstop: datetime
):
    stations = read_stations(stationfile)
    srcfiles = sorted(srcroot.glob("*.nc"))
    tm_list: list[float] = []
    data_list: dict[UUID, list[float]] = defaultdict(list)
    for tm, data in read_discharge(srcfiles, stations, dtbegin, dtstop):
        print(tm)
        tm_list.append(tm)
        for sid, value in data.items():
            data_list[sid].append(value)
    qglofas = pl.DataFrame(
        [
            pl.Series("datetime", tm_list, pl.Datetime("ms", "UTC")),
        ]
        + [pl.Series(sid.urn, data, pl.Float32) for sid, data in data_list.items()]
    )
    qglofas.write_parquet(desfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("srcroot", type=Path)
    parser.add_argument("stationfile", type=Path)
    parser.add_argument("desfile", type=Path)
    parser.add_argument("-b", "--begin", type=datetime.fromisoformat)
    parser.add_argument("-e", "--end", type=datetime.fromisoformat)
    args = parser.parse_args()

    dtbegin = cast(datetime, args.begin) or datetime(1979, 1, 1)
    dtstop = cast(datetime, args.end) or datetime(2024, 1, 1)
    srcroot = cast(Path, args.srcroot)
    stationfile = cast(Path, args.stationfile)
    desfile = cast(Path, args.desfile)

    main(srcroot, stationfile, desfile, dtbegin, dtstop)
