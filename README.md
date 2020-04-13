# AutoCNet

[![Gitter Chat](https://badges.gitter.im/USGS-Astrogeology/autocnet.svg)](https://gitter.im/USGS-Astrogeology/autocnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![Travis-CI](https://travis-ci.org/USGS-Astrogeology/autocnet.svg?branch=dev)](https://travis-ci.org/USGS-Astrogeology/autocnet)

[![Coveralls](https://coveralls.io/repos/USGS-Astrogeology/autocnet/badge.svg?branch=dev&service=github)](https://coveralls.io/github/USGS-Astrogeology/autocnet?branch=dev)

[![Docs](https://img.shields.io/badge/Docs-latest-green.svg)](https://usgs-astrogeology.github.io/autocnet/)

Automated sparse control network generation to support photogrammetric control of planetary image data.

## Documentation
Is available at: https://usgs-astrogeology.github.io/autocnet/

## Installation Instructions
We suggest using Anaconda Python to install Autocnet within a virtual environment.  These steps will walk you through the process.

1. [Download](https://www.continuum.io/downloads) and install the Python 3.x Miniconda installer.  Respond ``Yes`` when prompted to add conda to your BASH profile.  
2. Install the autocnet environment using the supplied environment.yml file: `conda env create -n autocnet -f environment.yml` 
3. Activate your environment: `conda activate autocnet`
4. If you are doing to develop autocnet or would like to use the bleeding edge version: `python setup.py develop`. Otherwise, `conda install -c usgs-astrogeology` autocnet.

## How to run the test suite locally

1. Install Docker
2. Get the Postgresql with Postgis container and run it `docker run --name testdb -e POSTGRES_PASSOWRD='NotTheDefault' -e POSTGRES_USER='postgres' -p 5432:5432 -d mdillon/postgis`
3. create database template_postgis: `docker exec testdb psql -c 'create database template_postgis;' -U postgres`
4. Run the test suite: `pytest autocnet`
