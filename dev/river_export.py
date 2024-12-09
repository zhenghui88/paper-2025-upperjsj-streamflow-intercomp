import argparse
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import polars as pl


def read_streamflow(file: Path) -> tuple[datetime, dict[UUID, float]]:
    dt = datetime.strptime(file.stem, "%Y%m%dT%H%M%S")
    streamflow: dict[UUID, float] = {}
    with open(file, "rt") as f:
        f.readline()  # skip header
        for ll in f:
            data = ll.strip().split(",")
            streamflow[UUID(data[0])] = float(data[1])
    print(dt, flush=True)
    return dt, streamflow


def main(
    inputroot: Path,
    outputfile: Path,
    begin: datetime | None = None,
    end: datetime | None = None,
) -> None:
    dts = []
    streamflow = {}
    for ff in sorted(inputroot.glob("**/*.csv")):
        dt = datetime.strptime(ff.stem, "%Y%m%dT%H%M%S")
        if (begin is not None and dt <= begin) or (end is not None and dt > end):
            continue
        dt, sf = read_streamflow(ff)
        dts.append(dt)
        for k, v in sf.items():
            streamflow.setdefault(k.urn, []).append(v)

    df = pl.DataFrame(
        [pl.Series("datetime", dts, pl.Datetime("ms", "UTC"))]
        + [pl.Series(k, v, pl.Float32) for k, v in streamflow.items()]
    )
    df = df.sort("datetime")
    df.write_parquet(outputfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputroot", type=Path)
    parser.add_argument("outputfile", type=Path)
    parser.add_argument(
        "-b",
        "--begin",
        type=lambda x: datetime.strptime(x, "%Y%m%dT%H%M%S"),
        help='begining time in "YYYYMMDDTHHMMSS" format, exclusive',
    )
    parser.add_argument(
        "-e",
        "--end",
        type=lambda x: datetime.strptime(x, "%Y%m%dT%H%M%S"),
        help='ending time in "YYYYMMDDTHHMMSS" format, inclusive',
    )

    args = parser.parse_args()
    inputroot = Path(args.inputroot)
    outputfile = Path(args.outputfile)
    begin = cast(datetime, args.begin) if args.begin else None
    end = cast(datetime, args.end) if args.end else None

    main(inputroot, outputfile, begin, end)
