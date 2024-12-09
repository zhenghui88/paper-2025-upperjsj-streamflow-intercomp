import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, cast
from uuid import UUID
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

EPSILON = np.float32(1e-8)


def read_fineweight(
    fineweightfile: Path,
) -> Iterator[tuple[datetime, dict[UUID, float]]]:
    df = pl.read_parquet(fineweightfile)
    for row in df.sort(by="datetime").iter_rows(named=True):
        yield (
            row["datetime"].astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
            {UUID(sid): value for sid, value in row.items() if sid != "datetime"},
        )


def read_coarse(
    coarsefile: Path,
) -> Iterator[tuple[datetime, dict[UUID, float]]]:
    df = pl.read_parquet(coarsefile)
    for row in df.sort(by="datetime").iter_rows(named=True):
        yield (
            row["datetime"].astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
            {UUID(sid): value for sid, value in row.items() if sid != "datetime"},
        )


def match(
    coarseiter: Iterable[tuple[datetime, dict[UUID, float]]],
    fineiter: Iterable[tuple[datetime, dict[UUID, float]]],
    hour: int,
) -> Iterator[
    tuple[
        tuple[datetime, dict[UUID, float]],
        tuple[tuple[datetime, dict[UUID, float]], ...],
    ]
]:
    type DataType = tuple[datetime, dict[UUID, float]]
    finedata = cast(DataType, next(fineiter))
    coarsedata = cast(DataType, next(coarseiter))
    while True:
        try:
            finedata_batch: list[DataType] = []
            while coarsedata[0] < finedata[0]:
                coarsedata = cast(DataType, next(coarseiter))
            dtbeg = cast(datetime, coarsedata[0]) - timedelta(hours=hour)
            while finedata[0] <= coarsedata[0]:
                if finedata[0] > dtbeg:
                    finedata_batch.append(finedata)
                finedata = cast(DataType, next(fineiter))
            yield (coarsedata, tuple(finedata_batch))
        except StopIteration:
            break


def redistribute(
    coarse: tuple[datetime, dict[UUID, float]],
    fineweight: tuple[tuple[datetime, dict[UUID, float]], ...],
) -> tuple[tuple[datetime, dict[UUID, float]], ...]:
    sum = {x: 0.0 for x in coarse[1].keys()}
    assert coarse[1].keys() == fineweight[0][1].keys()
    for _, data in fineweight:
        for k in data.keys():
            sum[k] += data[k] + EPSILON
    ret = []
    for dt, data in fineweight:
        ret.append(
            (
                dt,
                {
                    k: (data[k] / sum[k]) * len(fineweight) * coarse[1][k]
                    for k in data.keys()
                },
            )
        )
    return tuple(ret)


def main(inputfile: Path, weightfile: Path, hour: int, outputfile: Path):
    coarsereader = read_coarse(inputfile)
    finereader = read_fineweight(weightfile)
    tm_list: list[datetime] = []
    data_list: dict[UUID, list[float]] = defaultdict(list)
    for coarsedata, finedata in match(coarsereader, finereader, hour=hour):
        for tm, data in redistribute(coarsedata, finedata):
            print(tm, coarsedata[0])
            tm_list.append(tm)
            for sid, value in data.items():
                data_list[sid].append(value)
    q = pl.DataFrame(
        [
            pl.Series("datetime", tm_list, pl.Datetime("ms", "UTC")),
        ]
        + [pl.Series(sid.urn, data, pl.Float32) for sid, data in data_list.items()]
    )
    q.write_parquet(outputfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redistribute coarse-scale data")
    parser.add_argument("input", type=Path, help="Input coarse-scale data file")
    parser.add_argument("fineweight", type=Path, help="Fine-scale weighting data file")
    parser.add_argument(
        "hour",
        type=int,
        help="the time step of the coarse-scale data in hours",
    )
    parser.add_argument("output", type=Path, help="Output fine-scale data file")
    args = parser.parse_args()

    main(args.input, args.fineweight, args.hour, args.output)
