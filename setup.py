from setuptools import setup, find_packages

setup(
    name='assesSEM',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        'assesSEM/models': ['*.h5'],
    }
)
