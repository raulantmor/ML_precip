import xarray as xr
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('..'))
from data_processing.remap.healpix import HEALPixRemap
remapper = HEALPixRemap(181,360,64)
# Need example ll file
path_to_ll = '/path/to/example/1-deg_data.nc'

path_ed = '/path/to/model/output/HPX64_data.nc'
fc_ll = remapper.inverse_remap(path_ed,path_to_ll,vname='tp3')
fc_ll.to_netcdf('/path/to/remapped/ll_data.nc')