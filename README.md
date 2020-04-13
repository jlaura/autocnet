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
2. Get the Postgresql with Postgis container and run it `docker run --name testdb -e POSTGRES_PASSOWRD='' -e POSTGRES_USER='postgres' -p 5432:5432 -d mdillon/postgis`
3. create database template_postgis: `docker exec testdb psql -c 'create database template_postgis;' -U postgres`
4. Run the test suite: `autocnet_config=config/test_config.yml pytest autocnet`

## Simple Network Examples:


### Setup a project
This first example imports the object that is used to orchestrate jobs, manage
a database session, and generally work with the images, points, and measures in
a control network.

```
from autocnet.graph.network import NetworkCandidateGraph

# Make an empty NCG
ncg = NetworkCandidateGraph()
# Load the configuration file
ncg.config_from_file('config/demo.yml')

# Populate the nodes/edges from the DB
ncg.from_database()
```

Line by line, this code first imports the network candidate graph, a collection
of nodes and edges that represents a potential control network.
`from autocnet.graph.network import NetworkCandidateGraph`.

Next, a network candidate graph is instantiated. `ncg =
NetworkCandidateGraph()`.

The network candidate graph (or NCG) is assocaited with a collection of
PostGreSQL database tables. We have to initiate the database connection via a
configuration file. The configuration file. An example configuration file is
provided in the config directory. `ncg.config_from_file('config/demo.yml')`

Once configured, the images need to be loaded and the graph of potential
overlapping images generated. We do this with `ncg.from_database()`.

At this point, you have a fully functioning autocnet project using an NCG.

### Import images from a data store containing image footprints
Autocnet does not assume where your image footprints are coming from for
initial setup. We do assume that you have a prepopulated database of image
footprints with a `geom` column. Otherwise, you could use any software to
create image footprints and populate the footprint database.

To initialize a project from a data store of image footprints we can do the
following:

```
# These lines are pulled from the example above
from autocnet.graph.network import NetworkCandidateGraph

ncg = NetworkCandidateGraph()
ncg.config_from_file('config/demo.yml')

# Create the connection 
source_db_config = {'username':'jay',
        'password':'abcde',
        'host':'localhost',
        'pgbouncer_port':5432,
        'name':'someothertable'}

# Subset the data store using a spatial query.
geom = 'LINESTRING(145 10, 145 10.25, 145.25 10.25, 145.25 10, 145 10)'
srid = 949900
outpath = '/scratch/some/path/for/data'
query = f"SELECT * FROM ctx WHERE ST_INTERSECTS(geom, ST_Polygon(ST_GeomFromText('{geom}'), {srid})) = TRUE"
ncg.add_from_remote_database(source_db_config, outpath, query_string=query)
```

Here we create an NCG as above. Then we define a new database connection with
the name of the database from which data will be extracted. `source_db_config =
{'username':'jay', 'password':'abcde', 'host':'localhost',
'pgbouncer_port':5432, 'name':'someothertable'}.

In this example, we want to use a spatial query to subset the data. We could
also use an attribute query or some combination. The only restriction is that
the quert string be valid SQL. `geom = 'LINESTRING(145 10, 145 10.25, 145.25
10.25, 145.25 10, 145 10)'`

The PostGIS query requires a valid SRID for the input geometry, so we
explicitly define that here. This is the SRID that the footprints are being
stored in inside of the data store. `srid = 949900`.

The `add_from_remote_database` call copies the image files in the source
database into a new directory. Here we define that directory. `outpath =
'/scratch/some/path/for/data'`. 

The query string is then constructed: `query = f"SELECT * FROM ctx WHERE
ST_INTERSECTS(geom, ST_Polygon(ST_GeomFromText('{geom}'), {srid})) = TRUE"` 

Finally, our database associated with the NCG is populated and the image data
are copied. `ncg.add_from_remote_database(source_db_config, outpath,
query_string=query)`

### Operations on the NCG
After we have an NCG, we want to perform operations on the graph or on database
rows associated with the graph (e.g., the Points, Measures, or Image Overlaps).
We use a functional approach where an arbitray function can be applied to an
iterable associated with the graph. Here is a concrete example to help
illustrate what this looks like in practice.

```
from autocnet.graph.network import NetworkCandidateGraph

ncg = NetworkCandidateGraph()
ncg.config_from_file('/home/jlaura/autocnet_projects/demo.yml')
ncg.from_database()


# Define a function to govern the distribution of points in the North/South direction
def ns(x):
    from math import ceil
    return ceil(round(x,1)*8)

# Define a function to govern the distribution of points in the East/West direction
def ew(x):
    from math import ceil
    return ceil(round(x,1)*1)

# Pack a set of kwargs into a keyword that the called function is expecting
distribute_points_kwargs = {'nspts_func':ns, 'ewpts_func':ew, 'method':'new'}

# Apply a function on an iterable
njobs = ncg.apply('spatial.overlap.place_points_in_overlap', 
                  on='overlaps', 
                  cam_type='isis',
                  distribute_points_kwargs=distribute_points_kwargs)
```

Most of the above is either familiar boiler plate or a pair of helper functions
that we want to pass in. The interesting stuff is happening in:

```
njobs = ncg.apply('spatial.overlap.place_points_in_overlap',
                  on='overlaps',
                  cam_type='isis',
                  distribute_points_kwargs=distribute_points_kwargs)
```

Here, we are applying the `spatial.overlap.place_points_in_overlap` function on
an iterable (`overlaps`) with two keyword arguments (that the function is
expecting). The syntax for the function is module.submodule.function_name'.
Where the submodule can be repeated, e.g.,
module.submodule.subsubmodule.function_name.

It is possible to use a similar block to, for example, apply some subpixel
registration algorithm: 

`njobs = ncg.apply('matcher.subpixel.subpixel_register_point', on='points')`

or to apply a second pass subpixel alignment on only measures meeting some
criteria:

```
filters = {'ignore' : True}  # A database filter in the form column name : equality
njobs = ncg.apply('matcher.subpixel.subpixel_register_measure',
                  on='measures',
                  filters=filters)
```

