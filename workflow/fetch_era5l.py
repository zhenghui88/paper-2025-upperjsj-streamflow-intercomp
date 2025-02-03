# %%
import argparse
import tempfile
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import cdsapi
import dateutil.parser
import urllib3

urllib3.disable_warnings()


# %%
variables = [
    # "10m_u_component_of_wind",
    # "10m_v_component_of_wind",
    # "2m_dewpoint_temperature",
    # "2m_temperature",
    # "skin_temperature",
    # "potential_evaporation",
    # "surface_pressure",
    # "downward_uv_radiation_at_the_surface",
    # "surface_solar_radiation_downwards",
    # "surface_thermal_radiation_downwards",
    # "total_precipitation",
    # "snowfall",
    # "cloud_base_height",
    # "total_cloud_cover",
    # "high_cloud_cover",
    # "low_cloud_cover",
    # "medium_cloud_cover",
    "runoff",
    # "surface_runoff",
    # "sub_surface_runoff",
    # "snow_depth",
    # "snow_depth_water_equivalent",
    # "soil_temperature_level_1",
    # "soil_temperature_level_2",
    # "soil_temperature_level_3",
    # "soil_temperature_level_4",
    # "volumetric_soil_water_layer_1",
    # "volumetric_soil_water_layer_2",
    # "volumetric_soil_water_layer_3",
    # "volumetric_soil_water_layer_4",
]


days = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
]

times = [
    "00:00",
    "01:00",
    "02:00",
    "03:00",
    "04:00",
    "05:00",
    "06:00",
    "07:00",
    "08:00",
    "09:00",
    "10:00",
    "11:00",
    "12:00",
    "13:00",
    "14:00",
    "15:00",
    "16:00",
    "17:00",
    "18:00",
    "19:00",
    "20:00",
    "21:00",
    "22:00",
    "23:00",
]


def download(cdt: datetime, outroot: Path, tmproot: Path):
    filename = f"{cdt:%Y%m}.nc"
    tfile = tmproot.joinpath(filename)
    ofile = outroot.joinpath(filename)
    print(ofile, cdt)
    c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api")
    try:
        tfile.unlink(missing_ok=True)
        ofile.parent.mkdir(parents=True, exist_ok=True)
        ofile.unlink(missing_ok=True)
        res = c.retrieve(
            "reanalysis-era5-land",
            {
                "data_format": "netcdf",
                "download_format": "unarchived",
                "variable": variables,
                "year": f"{cdt:%Y}",
                "month": f"{cdt:%m}",
                "day": days,
                "time": times,
            },
        )
        res.download(tfile)
        tfile.rename(ofile)
    except Exception as e:
        print(e)


def main(
    outroot: Path,
    dt_beg: datetime,
    dt_end: datetime,
    tmproot: Path,
    verbose=False,
):
    dts: list[datetime] = []
    ctime = dt_beg.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while ctime < dt_end:
        filename = f"{ctime:%Y%m}.nc"
        tfile = tmproot.joinpath(filename)
        tfile.unlink(missing_ok=True)
        ofile = outroot.joinpath(filename)
        if not ofile.is_file():
            dts.append(ctime)
        ctime = (ctime + timedelta(days=31)).replace(day=1)

    if verbose:
        print(dts)
    with Pool(12) as pool:
        pool.starmap(download, [(x, outroot, tmproot) for x in dts])


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ERA5-Land data")
    parser.add_argument(
        "saving_root", type=Path, help="Root directory to save the output"
    )
    parser.add_argument(
        "datetime_start",
        type=str,
        help="Start datetime in UTC (inclusive)",
    )
    parser.add_argument(
        "datetime_stop",
        type=str,
        help="Stop datetime in UTC (exclusive)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print the datetime list",
    )
    args = parser.parse_args()
    dt_start = dateutil.parser.parse(args.datetime_start)
    dt_stop = dateutil.parser.parse(args.datetime_stop)

    try:
        tmproot = Path(
            tempfile.mkdtemp(
                prefix="tmp-fetch_era5l-",
                dir=args.saving_root.parent.absolute().as_posix(),
            )
        )
        main(
            args.saving_root,
            dt_start,
            dt_stop,
            tmproot,
            verbose=args.verbose,
        )
    finally:
        for child in tmproot.iterdir():
            if child.is_file():
                child.unlink()
            else:
                child.rmdir()
        tmproot.rmdir()
