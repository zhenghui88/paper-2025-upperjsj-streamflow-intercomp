from datetime import datetime
from pathlib import Path

import xarray as xr

DATETIME_START = datetime(2009, 1, 1)
DATETIME_STOP = datetime(2017, 1, 1)

DATAROOT = Path("../data/runoff")
datanames = ["era5l", "era5lnmp", "glofas", "grfr", "cnrd", "cldasnmp"]

YRMEAN_FILE = Path("../data/runoff_yrmean.nc")

SEAMEAN_FILE = Path("../data/runoff_seamean.nc")


yrmeandata: dict[str, xr.DataArray] = {}

for dsn in datanames:
    print(f"Processing {dsn}")
    data = xr.open_dataset(DATAROOT / f"{dsn}.nc")
    data = data["mrro"].sel(time=slice(DATETIME_START, DATETIME_STOP))
    yrmeandata[dsn] = data.mean(dim="time")
    data.close()
runoff_yrmean = xr.Dataset(
    data_vars={dsn: (["lat", "lon"], yrmeandata[dsn].data) for dsn in datanames},
    coords={
        "lat": list(yrmeandata.values())[0]["lat"].data,
        "lon": list(yrmeandata.values())[0]["lon"].data,
    },
)

runoff_yrmean.to_netcdf(YRMEAN_FILE)


seameandata: dict[str, dict[str, xr.DataArray]] = {}
for dsn in datanames:
    print(f"Processing {dsn}")
    data = xr.open_dataset(DATAROOT / f"{dsn}.nc")
    data = data["mrro"].sel(time=slice(DATETIME_START, DATETIME_STOP))
    seameandata[dsn] = data.groupby("time.season").mean("time")
    data.close()

runoff_seamean = xr.Dataset(
    data_vars={
        dsn: (["season", "lat", "lon"], seameandata[dsn].data) for dsn in datanames
    },
    coords={
        "lat": list(seameandata.values())[0]["lat"].data,
        "lon": list(seameandata.values())[0]["lon"].data,
        "season": [str(x) for x in list(seameandata.values())[0]["season"].data],
    },
)

runoff_seamean.to_netcdf(SEAMEAN_FILE)
