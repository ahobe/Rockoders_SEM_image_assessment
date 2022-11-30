# Rockoders_SEM_image_assessment
SEM image assessment tool produced during the 2022 SPE geothermal hackathon

### We recommend using GPU acceleration (easiest in google colab).
- ~30 s per sample using GPU acceleration.
- ~300 s per sample using CPU only.

### Creating the conda environment: 
```bash
conda env create -f assesSEM.yml
```

### Activate conda environment:
```bash
conda activate assesSEM
```

### install package
```bash
pip install -e .
```


###TODO:
- add results to test if the methods are working correctly

Check out https://git-lfs.github.com/ for large files.


conda env create -f Rockoders.yml
conda env remove -n Rockoders
