import numpy as np
import pandas as pd
import geopandas as gpd

import scipy
from scipy.misc import imresize

from plio.io.io_gdal import GeoDataset

from geoalchemy2 import functions, shape
from shapely import wkt
from shapely.geometry import Point

from autocnet import config, engine, Session
from autocnet.io.db.model import Images, Points, Measures, ThemisImages
from autocnet.graph.network import NetworkCandidateGraph
from autocnet.matcher.subpixel import iterative_phase
from autocnet.cg.cg import distribute_points_in_geom
from autocnet.io.db.connection import new_connection
from autocnet.io.db.model import Images
from autocnet.spatial import isis

import warnings

def generate_ground_points(ground_database, image_subquery=None, nspts_func=lambda x: int(round(x,1)*1), ewpts_func=lambda x: int(round(x,1)*4)):
    """
    Generate some number of candidate ground points inside of an already controlled data set using
    a set of images in the current working data set. For example, if the working data set is CTX, 
    this method can be used to place points into THEMIS (or HRSC) images. The algorithm operates
    by selecting some set of images from the current active database, performing a spatial union
    on the footprints of those images, placing points into the resultant unioned footprint, selecting
    all intersecting images from the existing controlled data set, and finally identifying those
    points which intersect a given already controlled image.

    TODO: Break this func apart for better testability.
    
    Parameters
    ----------
    ground_database : dict
                      in the form {'username':'jay',
                                   'password':'abcde',
                                   'host':'autocnet.wr.usgs.gov', 
                                   'pgbouncer_port':'6543', 
                                   'name':'themis'}
    
    image_subquery : obj
                     A valid sqlalchemy or geomalchemy filter object. For example:
                     `Images.id < 5` or `Images.footprint_latlon.intersects(p.wkt)`,
                     where `p` is a shapely geometry
                     
    nspts_func : obj
                 A function taking one argument controlling the number of north-south
                 points to be placed into a footprint
                 
    ewpts_func : obj
                 A function taking one argument controlling the number of east-west
                 points to be placed into a footprint
                 
    Returns
    -------
    ground_cnet : GeoDataframe
                  A geopandas GeoDataframe containing one row for every point-image
                  combination including 'line', 'sample', 'resolution', and 'path'
                  columns.
                 
    """
    # Get the union of some number of images in the database being worked on
    images_poly = Images.union(subquery=image_subquery)
    
    # Distribute points within the unioned geometry as candidate ground points.
    coords = distribute_points_in_geom(images_poly, nspts_func=nspts_func, ewpts_func=ewpts_func)
    coords = np.asarray(coords)
    coords = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=coords[:,0], y=coords[:,1]))

    # Get the ground dataset that the working images are to be tied to
    ground_Session, _ = new_connection(ground_database)
    ground_session = ground_Session()
    # Get the query object, pass to pandas for the SQL read, remap wkb to shapely geoms, and get a geodataframe
    ground_query = ground_session.query(ThemisImages).filter(ThemisImages.geom.intersects(images_poly.wkt))
    df = pd.read_sql(ground_query.statement, ground_query.session.bind)
    df['geom'] = df['geom'].apply(lambda g: shape.to_shape(g))
    themis_images = gpd.GeoDataFrame(df, geometry='geom')
    ground_session.close()

    # Add mock keys to support a cartesian product merge of the points and the images, mask all the
    # points that do not fall in a given image, and prepare to group and intersect
    coords['key'] = 0
    coords['pointid'] = np.arange(len(coords))
    themis_images['key'] = 0
    merged = themis_images.merge(coords, how='outer')
    mask = merged.apply(lambda r:r.geom.contains(r.geometry), axis=1)
    ground_cnet = merged[mask]

    
    # group by images so campt can do multiple at a time
    for _, group in ground_cnet.groupby('path'):
        
        row = group.iloc[0] # Get the image path for all the points
        lons = [p.x for p in group['geometry']]
        lats = [p.y for p in group['geometry']]

        # Project to ground
        point_list = isis.point_info(row['path'], lons, lats, 'ground')
        lines = []
        samples = []
        resolutions = []
        
        for _, res in enumerate(point_list):
            if res[1].get('Error') is not None:
                print('Bad intersection')
                lines.append(None)
                samples.append(None)
                resolutions.append(None)
            else:
                lines.append(res[1].get('Line'))
                samples.append(res[1].get('Sample'))
                resolutions.append(res[1].get('LineResolution').value)
        # TODO: These lines throw a series set warning
        ground_cnet.loc[group.index, 'line'] = lines
        ground_cnet.loc[group.index, 'sample'] = samples
        ground_cnet.loc[group.index, 'resolution'] = resolutions
    return ground_cnet

def propagate_control_network(base_cnet):
    """

    """
    dest_images = gpd.GeoDataFrame.from_postgis("select * from images", engine, geom_col="footprint_latlon")
    spatial_index = dest_images.sindex
    groups = base_cnet.groupby('pointid').groups
    # append to list if images, mostly used for working with the network in python
    # after this step, is this uncecceary outside of debugging? Maybe actually should return
    # more info of where everything was sourced in the original DataFrames?
    images = []

    # append CNET info into structured Python list
    constrained_net = []
    dbpoints = []
    dbmeasures = []

    # easily parrallelized on the cpoint level, dummy serial for now
    for cpoint, indices in groups.items():
        measures = base_cnet.loc[indices]
        measure = measures.iloc[0]

        p = measure.point
        # get image in he destination that overlap
        matches = dest_images[dest_images.intersects(p)]

        # lazily iterate for now
        for i,row in matches.iterrows():
            res = isis.point_info(row["path"], p.x, p.y, point_type="ground", allow_outside=False)
            dest_line, dest_sample = res["GroundPoint"]["Line"], res["GroundPoint"]["Sample"]

            try:
                dest_resolution = res["GroundPoint"]["LineResolution"].value
            except:
                warnings.warn(f'Failed to generate ground point info on image {row["path"]} at lat={p.y} lon={p.x}')
                continue

            dest_data = GeoDataset(row["path"])
            dest_arr = dest_data.read_array()

            # dynamically set scale based on point resolution
            dest_to_base_scale = dest_resolution/measure["resolution"]

            scaled_dest_line = (dest_arr.shape[0]-dest_line)*dest_to_base_scale
            scaled_dest_sample = dest_sample*dest_to_base_scale

            dest_arr = imresize(dest_arr, dest_to_base_scale)[::-1]

            # list of matching results in the format:
            # [measure_index, x_offset, y_offset, offset_magnitude]
            match_results = []
            for k,m in measures.iterrows():
                base_arr = GeoDataset(m["path"]).read_array()

                sx, sy = m["sample"], m["line"]
                dx, dy = scaled_dest_sample, scaled_dest_line
                try:
                    # not sure what the best parameters are here
                    ret = iterative_phase(sx, sy, dx, dy, base_arr, dest_arr, size=10, reduction=1, max_dist=1, convergence_threshold=1)
                except Exception as ex:
                    match_results.append(ex)
                    continue

                if ret is not None and None not in ret:
                    x,y,metrics = ret
                else:
                    match_results.append("Failed to Converge")
                    continue

                dist = np.linalg.norm([x-dx, -1*(y-dy)])
                match_results.append([k, x-dx, -1*(y-dy), dist])

            # get best offsets, if possible we need better metric for what a
            # good match looks like
            match_results = np.asarray([res for res in match_results if isinstance(res, list)])
            if match_results.shape[0] == 0:
                # no matches
                continue
            match_results = match_results[np.argwhere(match_results[:,3] == match_results[:,3].min())][0][0]

            if match_results[3] > 2:
                # best match diverged too much
                continue

            measure = measures.loc[match_results[0]]

            # apply offsets
            sample = (match_results[1]/dest_to_base_scale) + dest_sample
            line = (match_results[2]/dest_to_base_scale) + dest_line

            pointpvl = isis.point_info(row["path"], sample, line, point_type="image")
            groundx, groundy, groundz = pointpvl["GroundPoint"]["BodyFixedCoordinate"].value
            groundx, groundy, groundz = groundx*1000, groundy*1000, groundz*1000

            images.append(row["path"])
            constrained_net.append({
                    'pointid' : cpoint,
                    'imageid' : row['id'],
                    'serial' : row.serial,
                    'line' : line,
                    'sample' : sample,
                    'point_latlon' : p,
                    'point_ground' : Point(groundx, groundy, groundz)
                })

    ground = gpd.GeoDataFrame.from_dict(constrained_net).set_geometry('point_latlon')
    groundpoints = ground.groupby('pointid').groups

    points = []

    # upload new points
    for p,indices in groundpoints.items():
        point = ground.loc[indices].iloc[0]
        p = Points()
        p.pointtype = 3
        p.apriori = point['point_ground']
        p.adjusted = point['point_ground']

        for i in indices:
            m = ground.loc[i]
            p.measures.append(Measures(line=float(m['line']),
                                       sample = float(m['sample']),
                                       imageid = int(m['imageid']),
                                       serial = m['serial'],
                                       measuretype=3))
        points.append(p)

    session = Session()
    session.add_all(points)
    session.commit()

    return ground

