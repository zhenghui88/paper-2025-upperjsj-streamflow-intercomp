import argparse
from pathlib import Path

import h5netcdf
import numpy as np


def read_inbasin(inbasin: Path):
    with h5netcdf.File(inbasin, "r") as f:
        return np.array(f.variables["inbasin"][:, :], np.bool)


def copy_ncfile_headers(inputfile: h5netcdf.File, outputfile: h5netcdf.File):
    outputfile.dimensions.update(
        {
            k: None if v.isunlimited() else len(v)
            for k, v in inputfile.dimensions.items()
        }
    )
    for varname, var in inputfile.variables.items():
        attrs = {str(k): str(v) for k, v in var.attrs.items()}
        fillvalue = attrs.pop("_FillValue", None)
        dims = var.dimensions
        if any(outputfile.dimensions[d].isunlimited() for d in dims):
            data = None
        else:
            data = var[:]
        outputfile.create_variable(
            varname,
            var.dimensions,
            data=data,
            dtype=var.dtype,
            fillvalue=fillvalue,
            compression="gzip",
        ).attrs.update(attrs)


def main(inputfile: Path, basinmaskfile: Path, outputfile: Path):
    basinmask = read_inbasin(basinmaskfile)
    with h5netcdf.File(inputfile, "r") as fi, h5netcdf.File(outputfile, "w") as fo:
        copy_ncfile_headers(fi, fo)
        for itime, timenum in enumerate(fi.variables["time"][:]):
            print(itime)
            fo.resize_dimension("time", itime + 1)
            fo.variables["time"][itime] = timenum
            data = fi.variables["mrro"][itime, :, :]
            data[~basinmask] = np.nan
            fo.variables["mrro"][itime, :, :] = data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="The runoff data needed to be masked")
    parser.add_argument("basinmask", type=Path, help="The basin mask")
    parser.add_argument("output", type=Path, help="The output file")
    args = parser.parse_args()

    main(args.input, args.basinmask, args.output)
