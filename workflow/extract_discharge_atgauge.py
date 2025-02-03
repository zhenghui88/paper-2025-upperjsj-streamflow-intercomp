import argparse
from pathlib import Path
from uuid import UUID

import polars as pl


def read_station(filepath: Path) -> dict[UUID, tuple[UUID, tuple[float, float]]]:
    data = {}
    with open(filepath, "rt") as f:
        f.readline()
        for line in f:
            sid, _name, _merit, rid, lat, lon = line.strip().split(",")
            data[UUID(rid)] = (UUID(sid), (float(lat), float(lon)))
    return data


def main(inputfile: Path, stationfile: Path, outputfile: Path) -> None:
    stations = read_station(stationfile)
    data = pl.read_parquet(inputfile)
    data = data.select(
        pl.col("datetime"),
        *[pl.col(k.urn).alias(v[0].urn) for k, v in stations.items()],
    )
    data.write_parquet(outputfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", type=Path)
    parser.add_argument("stationfile", type=Path)
    parser.add_argument("outputfile", type=Path)
    args = parser.parse_args()

    inputfile = Path(args.inputfile)
    stationfile = Path(args.stationfile)
    outputfile = Path(args.outputfile)

    main(inputfile, stationfile, outputfile)
