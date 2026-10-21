import sys 
import argparse
import os
import numpy as np
import xarray as xr
import healpy as hp
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

sys.path.insert(0, "../")
from data_processing.remap.healpix import HEALPixRemap
from data_processing.remap.cubesphere import to_chunked_dataset

PARAMS = [
    # list of dictionaries. if you want to remap several files you can include them as another dictionary in this list
    {
        # name of the file to be remapped. make sure to include full path
        'file_name' : '/path/to/file_in_lat-lon.nc',
        # variable name in the dataset above 
        'target_variable_name' : 'var', # this is used to extract the DataArray from the Dataset
        # this is the variable name that will be appended to output HPX file 
        'file_variable_name' : 'var', 
        # this is the "prefix" for the output file. Should include the desired path to the output file
        'prefix' : '/path/for/output/data_HPX64_',
        # healpix dimension
        'nside' : 64,
        # this is for debugging, creates image of healpix data and exits 
        'visualize' : False, 
        # this is a weird parameter Matthias used to remap... don't mess with this 
        'resolution_factor' : 1.0,
    },
]

def main(args):

    nside = args.nside
    prefix = args.prefix

    assert f"HPX{nside}" in prefix, (f"'HPX{nside}' could not be found in the prefix '{prefix}'. Please make sure "
                                     f"that the nside={nside} parameter and the prefix provided to this function "
                                     f"match.")

    # Load .nc file in latlon format to extract latlon information and to initialize the remapper module
    ds_ll = xr.open_dataset(args.file_name).rename({"time": "sample"}).squeeze() #,"pressure_level":"level"}).squeeze()

    latitudes, longitudes = ds_ll.dims["latitude"], ds_ll.dims["longitude"]
    mapper = HEALPixRemap(
        latitudes=latitudes,
        longitudes=longitudes,
        nside=nside,
        resolution_factor=args.resolution_factor,
        order="biquadratic"
        )

    # Determine whether a "constant" or "variable" is processed
    const = False if "sample" in list(ds_ll.dims.keys()) else True
    vname = list(ds_ll.keys())[0]

    # Set up coordinates and chunksizes for the HEALPix dataset
    coords = {}
    if not const:
        coords["sample"] = ds_ll.coords["sample"]
        coords["varlev"] = xr.DataArray(data=np.array([f'{ds_ll.level.values}']) if "level" in ds_ll.coords.keys() else np.array([f'{args.target_variable_name}/0']),
                                        dims=['varlev'],
                                        coords = {'varlev':(('varlev'),np.array([f'{args.target_variable_name}/0']))},
                                        name='level')
    coords["face"] = np.array(range(12), dtype=np.int64)
    coords["height"] = np.array(range(nside), dtype=np.int64)
    coords["width"] = np.array(range(nside), dtype=np.int64)

    # Map the "constant" or "variable" to HEALPix
    if const:
        data_hpx = mapper.ll2hpx(ds_ll.variables[vname].values, visualize=args.visualize)
        ds_mean = ds_ll.variables[vname].mean()
        ds_std = ds_ll.variables[vname].std()
    else:
        dims = []
        for coord in coords: 
            dims.append(len(coords[coord]) if coord!='varlev' else 1)

        # Sequential sample mapping via for-loop
        # Allocate a (huge) array to store all samples (time steps) of the projected data
        data_hpx = np.zeros(dims, dtype=ds_ll.variables[vname])
        # Iterate over all samples and levels, project them to HEALPix and store them in the predictors array
        pbar = tqdm(ds_ll.coords["sample"])
        for s_idx, sample in enumerate(pbar):
            pbar.set_description("Remapping time steps")
            #print("empty array:", data_hpx.shape)
            data_hpx[s_idx, 0] = mapper.ll2hpx(data=ds_ll.variables[vname][s_idx].values,
                                               visualize=args.visualize)
    print('made it out of the loop')
    # Determine latitude and longitude values for the HEALPix faces
    hpxlon, hpxlat = hp.pix2ang(mapper.nside, range(mapper.npix), nest=True, lonlat=True)
    data_lat = mapper.hpx1d2hpx3d(hpx1d=hpxlat, dtype=np.float32)
    data_lon = mapper.hpx1d2hpx3d(hpx1d=hpxlon, dtype=np.float32)

    # Build HEALPix dataset and write it to file
    ds_hpx = xr.Dataset(
        coords=coords,
        data_vars={
            "lat": (["face", "height", "width"], data_lat),
            "lon": (["face", "height", "width"], data_lon),
            "predictors": (list(coords.keys()), data_hpx),
            },
        attrs=ds_ll.attrs
        )
    # define chunksizes for dataarray 
    chunksizes = {}
    for coord in coords: 
        chunksizes[coord] = len(coords[coord]) if coord in ['face','height','width'] else 1
    ds_hpx = to_chunked_dataset(ds=ds_hpx, chunking=chunksizes)
    print(ds_hpx)
    print(f"Dataset sucessfully built. Writing data to file {prefix + args.file_variable_name + '.nc'}...")
    ds_hpx.to_netcdf(prefix + args.file_variable_name + ".nc")
    ds_reload = xr.open_dataset(prefix + args.file_variable_name + ".nc")
    print(ds_reload)

if __name__ == "__main__":

    print('checking parameters...')
    for param in tqdm(PARAMS): 
        if not os.path.isfile(param['file_name']):
            print(f'source file ~{param["file_name"]}~ not found. Aborting.')
            exit()
        print(param['prefix']+param['file_variable_name']+'.nc')
        if os.path.isfile(param['prefix']+param['file_variable_name']+'.nc'):
            print(f'target file ~{param["prefix"]+param["file_variable_name"]+".nc"}~ already exists. Aborting.')
            exit()
    for param in PARAMS: 
        run_args = argparse.Namespace(**param)
        print(f'beginning remap with parameters:')
        print(f'{run_args}')
        main(args=run_args)
        print("Done.")
