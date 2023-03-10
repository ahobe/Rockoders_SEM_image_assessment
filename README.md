<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

# AssesSEM
Cleaned version of the second place winning code for the SPE Europe Energy GeoHackathon focused on predicting Geothermal reservoir quality from SEM images.
https://www.spehackathon-eu.com/

## What this project do
This project was part of the code submmited to the 2022 SPE Geothermal Hackathon, the project is used for automatically characterize and quantify Geothermal Fields. The model clasiffieds minerals and pore spaces in thin sections using the input of backscatter electrons (BSE), cathodoluminescence (CL) microscopy and mineral-map images

### Dataset location:
https://dataverse.geus.dk/dataset.xhtml?persistentId=doi:10.22008/FK2/5TWAZK

This dataset needs to be added into the existing folders.
Inside each folder is a dummy tiff file, which git requires to show the folders.
Delete these files before proceeding.

### Downloading the required ML models:
The models can be found [here](https://drive.google.com/drive/folders/1-hV6jhjIMoUTvTnvGypGklVqeicIiAEe?usp=sharing).
These models need to be added to *src/assesSEM/models/* for the methods to work. 

### Creating the conda environment: 
```bash
conda env create -f assesSEM.yml
```

### Activate conda environment:
```bash
conda activate assesSEM
```

### Install package from its root folder 
```bash
pip install -e .
```

### Test if the installation worked
```bash
pytest
```

### Usage in the bottom of the dataset folder
```bash
python run.py
```

You will then be prompted about the desired nr of images per dataset.
