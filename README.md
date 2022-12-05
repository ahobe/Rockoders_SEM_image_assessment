Original submission to the SPE Europe Energy GeoHackathon focused on predicting Geothermal reservoir quality from SEM images.
https://www.spehackathon-eu.com/

### Dataset location:
https://dataverse.geus.dk/dataset.xhtml?persistentId=doi:10.22008/FK2/5TWAZK

This dataset needs to be added into the existing folders.
Inside each folder is a dummy tiff file, which git requires to show the folders.
Delete these files before proceeding.

### Creating the conda environment: 
```bash
conda env create -f Rockoders.yml
```

### Activate conda environment:
```bash
conda activate Rockoders
```

### Usage of original submission
```bash
python run.py
```

You will then be prompted about the desired nr of images per dataset.
