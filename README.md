# A Deep Learning Precipitation Diagnostic Model

Code, model configurations, and trained checkpoints accompanying:

Moreno and Durran, "Do Climate Models Need Microphysical and Convective Parameterizations to Generate Accurate Precipitation Fields?," Artificial Intelligence for the Earth Systems (2026). DOI:

## Overview

To run ML$_{\rm IMERG}$ or ML$_{\rm ERA5}$, you must first set up the environment, clone the repository, and obtain all necessary input data.
    
1.  Environment:

    Conda is recommended for environment management (and was used in this work). A yaml file is provided to replicate the necessary environment. Simply run:
    ```sh
    conda env create -f /path/to/ML_precip/environments/ml_precip.yaml
    ```
    then activate the environment: 
    ```sh
    conda activate ml_precip
    ```
    
2.  Clone the repository:

    ```sh
    git clone https://github.com/raulantmor/ML_precip.git
    ```
    
3.  Compile the data:

    All data used in this work is publicly available through the Climate Copernicus Data Store for ERA5, NASA’s EarthData for IMERG, and through NOAA for Stage IV. Data should then be remapped to the HEALPix64 grid used for training and running the diagnostic models. An example script to use for remapping data can be found at:
    ```sh
    /scripts/map2hpx.py
    ```
    
## Running ML$_{\rm IMERG}$ and ML$_{\rm ERA5}$

The analysis presented in this work spanned the period between 2022 and the end of 2024. To reporduce these simulations, follow these steps:

1.  Activate the environment:
    ```sh
    conda activate ml_precip
    ```
    
2.  Run the diagnosis script, giving specifications for which model to run, for what time period, and path to save the output. An example is shown below:

    ```sh
    python scripts/diagnose.py -m 'models/ML_imerg' -c 'last.ckpt' -l 0 -s '2022-01-01T00' -f '6h' -e '2024-12-31T21' -o 'outputs/' --output-filename 'EI_diag_2022-2024'
    ```
    
3.  The resulting output file will be in the HPX64 grid format. To map it to 1-degree lat/lon, use the remapping script in data_processing. An example implemnetation can be found in

    ```sh
    /scripts/hpx64_to_1deg.py
    ```
    
And you're done! The 1-degree lat/lon data can now be evaluated using the scripts found in /evaluation/utils

    
