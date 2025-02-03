# %%
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, cast
from uuid import UUID

import h5netcdf
import polars as pl

DATETIME_START = datetime(1979, 1, 1)
DATETIME_STOP = datetime(2020, 1, 1)


def read_stations(filepath: Path) -> dict[int, UUID]:
    stations: dict[UUID, UUID] = {}
    with open(filepath, "rt") as f:
        f.readline()
        for line in f:
            sid, _name, merit, _rid, _lat, _lon = line.strip().split(",")
            stations[int(merit)] = UUID(sid)
    return stations


def find_river_index(riverids: list[int], stations: dict[int, UUID]) -> dict[int, UUID]:
    ret = {}
    for merit, station in stations.items():
        idx = riverids.index(merit)
        ret[idx] = station
    return ret


def read_discharge(
    filepaths: Iterable[Path],
    stations: dict[int, UUID],
    dtbegin: datetime,
    dtstop: datetime,
) -> Iterator[tuple[datetime, dict[UUID, float]]]:
    for filepath in filepaths:
        with h5netcdf.File(filepath, "r") as f:
            tm = [
                datetime.strptime(
                    f.variables["time"].attrs["units"], "seconds since %Y-%m-%d %H:%M"
                )
                + timedelta(seconds=int(x))
                for x in f.variables["time"][:]
            ]
            if tm[0] > dtstop or tm[-1] < dtbegin:
                continue
            riverindex = find_river_index(list(f.variables["rivid"][:]), stations)
            for i, t in enumerate(tm):
                if t < dtbegin or t > dtstop:
                    continue
                yield (
                    t,
                    {
                        station: f.variables["Qout"][i, idx]
                        for idx, station in riverindex.items()
                    },
                )


def main(
    srcroot: Path, stationfile: Path, desfile: Path, dtbegin: datetime, dtstop: datetime
):
    stations = read_stations(stationfile)
    srcfiles = sorted(srcroot.glob("output_pfaf_04_*.nc"))
    tm_list: list[float] = []
    data_list: dict[UUID, list[float]] = defaultdict(list)
    for tm, data in read_discharge(srcfiles, stations, dtbegin, dtstop):
        print(tm)
        tm_list.append(tm)
        for station, value in data.items():
            data_list[station].append(value)
    qgrfr = pl.DataFrame(
        [
            pl.Series("datetime", tm_list, pl.Datetime("ms", "UTC")),
        ]
        + [
            pl.Series(station.urn, data, pl.Float32)
            for station, data in data_list.items()
        ]
    )
    qgrfr.write_parquet(desfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("srcroot", type=Path)
    parser.add_argument("stationfile", type=Path)
    parser.add_argument("desfile", type=Path)
    parser.add_argument("-b", "--begin", type=datetime.fromisoformat)
    parser.add_argument("-e", "--end", type=datetime.fromisoformat)
    args = parser.parse_args()

    dtbegin = cast(datetime, args.begin) or DATETIME_START
    dtstop = cast(datetime, args.end) or DATETIME_STOP
    stationfile = Path(args.stationfile)
    srcroot = Path(args.srcroot)
    desfile = Path(args.desfile)

    main(srcroot, stationfile, desfile, dtbegin, dtstop)
