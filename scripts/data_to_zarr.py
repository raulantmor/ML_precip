from omegaconf import OmegaConf

from data_processing.utils import write_zarr
import pandas as pd

# NOTE: All files should be in HPX64 format.
zarr_params={
    "dst_directory":"/path/to/destination/",
    "dataset_name":"hpx64_zarr",
    "inputs":{
        "z500":"/path/to/z500.nc",
        "tau300-700":"/path/to/tau300-700.nc",
        "z1000":"/path/to/z1000.nc",
        "t2m0":"/path/to/t2m.nc",
        "tcwv0":"/path/to/tcwv.nc",
        "t850":"/path/to/t850.nc",
        "z250":"/path/to/z250.nc",
        "ws10":"/path/to/ws10.nc",
        "olr":"/path/to/ttr.nc",
        "sst":"/path/to/sst.nc",
        "u850":"/path/to/u850.nc",
        "v850":"/path/to/v850.nc",
        "q850":"/path/to/q850.nc"
    },
    "outputs":{"tp3":"/path/to/tp3.nc"},
    "constants":{
        "lsm":"/path/to/lsm.nc",
        "z":"/path/to/topography.nc"
    },
    "batch_size":32,
    "time_subset":pd.date_range(start="1980-01-01T00", end="2024-12-31T00", freq='3H'),
    "scaling":OmegaConf.load("/ML_precip/training/configs/data/scaling/ML_imerg_data-config.yaml")
    
    
}

write_zarr.create_prebuilt_zarr(
    zarr_params["dst_directory"],
    zarr_params["dataset_name"],
    zarr_params["inputs"],
    zarr_params["outputs"],
    constants=zarr_params["constants"],
    time_subset=zarr_params["time_subset"],
    scaling=zarr_params["scaling"]
)

print('finished zarr.')