<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

# AssesSEM
Original submission to the SPE Europe Energy GeoHackathon focused on predicting Geothermal reservoir quality from SEM images.
https://www.spehackathon-eu.com/

### Dataset location:
https://dataverse.geus.dk/dataset.xhtml?persistentId=doi:10.22008/FK2/5TWAZK

This dataset needs to be added into the existing folders.
Inside each folder is a dummy tiff file, which git requires to show the folders.
Delete these files before proceeding.

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
