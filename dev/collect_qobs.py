# %%
from datetime import datetime
from uuid import UUID
from zoneinfo import ZoneInfo

import polars as pl

# %%
DATETIME_START = datetime(2008, 1, 1, tzinfo=ZoneInfo("UTC"))
DATETIME_STOP = datetime(2017, 1, 1, tzinfo=ZoneInfo("UTC"))

# %%
dfzmd = pl.read_parquet("../datasource/直门达(通天河).parquet").select(
    pl.col("tm").alias("datetime"),
    pl.col("q").alias(UUID("urn:uuid:E4FF26A2-3871-481A-9A83-D43847596BB0").urn),
)

# %%
dfgt = pl.read_parquet("../datasource/岗拖(三).parquet").select(
    pl.col("tm").alias("datetime"),
    pl.col("q").alias(UUID("urn:uuid:2947AFEF-51B6-4B02-82F7-995D12185682").urn),
)


# %%
dfbzl = pl.read_parquet("../datasource/奔子栏(三).parquet").select(
    pl.col("tm").alias("datetime"),
    pl.col("q").alias(UUID("urn:uuid:50CC5DFB-734F-4E67-A8F5-4095D0D72E85").urn),
)


# %%
dfbt = pl.read_parquet("../datasource/巴塘(五).parquet").select(
    pl.col("tm").alias("datetime"),
    pl.col("q").alias(UUID("urn:uuid:AC5B1300-C6EE-4395-A285-D77C6B3A1C95").urn),
)


# %%
df = (
    (
        dfzmd.join(dfgt, on="datetime", how="full", coalesce=True)
        .join(dfbt, on="datetime", how="full", coalesce=True)
        .join(dfbzl, on="datetime", how="full", coalesce=True)
    )
    .sort(by="datetime")
    .filter(pl.col("datetime") >= DATETIME_START)
    .filter(pl.col("datetime") <= DATETIME_STOP)
)


# %%
df.write_parquet("../data/discharge_observation.parquet")
