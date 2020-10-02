import warnings
import json

from redis import StrictRedis
import numpy as np
import pyproj
import shapely
import sqlalchemy
from plio.io.io_gdal import GeoDataset
from pysis.exceptions import ProcessError

from autocnet.cg import cg as compgeom
from autocnet.graph.node import NetworkNode
from autocnet.io.db.model import Images, Measures, Overlay, Points, JsonEncoder
from autocnet.spatial import isis
from autocnet.matcher.cpu_extractor import extract_most_interesting
from autocnet.matcher.subpixel import geom_match
from autocnet.transformation.spatial import reproject, og2oc, oc2og
from autocnet.transformation import roi

from plurmy import Slurm
import csmapi

# SQL query to decompose pairwise overlaps
compute_overlaps_sql = """
WITH intersectiongeom AS
(SELECT geom AS geom FROM ST_Dump((
   SELECT ST_Polygonize(the_geom) AS the_geom FROM (
     SELECT ST_Union(the_geom) AS the_geom FROM (
     SELECT ST_ExteriorRing((ST_DUMP(geom)).geom) AS the_geom
       FROM images WHERE images.geom IS NOT NULL) AS lines
  ) AS noded_lines))),
iid AS (
 SELECT images.id, intersectiongeom.geom AS geom
    FROM images, intersectiongeom
    WHERE images.geom is NOT NULL AND
    ST_INTERSECTS(intersectiongeom.geom, images.geom) AND
    ST_AREA(ST_INTERSECTION(intersectiongeom.geom, images.geom)) > 0.000001
)
INSERT INTO overlay(intersections, geom) SELECT row.intersections, row.geom FROM
(SELECT iid.geom, array_agg(iid.id) AS intersections
  FROM iid GROUP BY iid.geom) AS row WHERE array_length(intersections, 1) > 1;
"""

def add_measures_to_nodes(nodes, gnd, cam_type='csm', measuretype=2, choosername='place_points_in_overlap'):
        """
        Given a list of nodes and a ground coordinate (lon, lat), attempt to add measures to 
        the node at the given coordinate using a sensor model.

        Parameters
        ----------
        nodes : iterable
                of node objects

        gnd : tuple
              in the form (x,y,z) if cam_type='csm' or (lon, lat) if cam_type='isis'

            
        """
        measures = []
        for node in nodes:
            if cam_type == "csm":
                gnd = csmapi.EcefCoord(*gnd)
                image_coord = node.camera.groundToImage(gnd)
                sample, line = image_coord.samp, image_coord.line
            if cam_type == "isis":
                try:
                    lon, lat = gnd
                    line, sample = isis.ground_to_image(node["image_path"], lon, lat)
                except ProcessError as e:
                    if 'Requested position does not project in camera model' in e.stderr:
                        print(f'interesting point ({lon},{lat}) does not project to image {node["image_path"]}')
                        continue
            measure = Measures(sample=sample,
                                line=line,
                                apriorisample=sample,
                                aprioriline=line,
                                imageid=node['node_id'],
                                serial=node.isis_serial,
                                measuretype=measuretype,
                                choosername=choosername)
            measures.append(measure)
        return measures

def place_points_in_overlaps(size_threshold=0.0007,
                             distribute_points_kwargs={},
                             cam_type='csm',
                             ncg=None):
    """
    Place points in all of the overlap geometries by back-projecing using
    sensor models.

    Parameters
    ----------
    nodes : dict-link
            A dict like object with a shared key with the intersection
            field of the database Overlay table and a cg node object
            as the value. This could be a NetworkCandidateGraph or some
            other dict-like object.

    Session : obj
              The session object from the NetworkCandidateGraph

    size_threshold : float
                     overlaps with area <= this threshold are ignored
    """
    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    for overlap in Overlay.overlapping_larger_than(size_threshold, Session):
        if overlap.intersections == None:
            continue
        place_points_in_overlap(overlap,
                                cam_type=cam_type,
                                distribute_points_kwargs=distribute_points_kwargs,
                                ncg=ncg)

def place_points_in_overlap(overlap,
                            cam_type="csm",
                            size=71,
                            distribute_points_kwargs={},
                            ncg=None,
                            **kwargs):
    """
    Place points into an overlap geometry by back-projecing using sensor models.
    The DEM specified in the config file will be used to calculate point elevations.

    Parameters
    ----------
    overlap : obj
              An autocnet.io.db.model Overlay model instance

    cam_type : str
               options: {"csm", "isis"}
               Pick what kind of camera model implementation to use

    size : int
           The size of the window used to extractor features to find an
           interesting feature to which the point is shifted.

    Returns
    -------
    points : list of Points
        The list of points seeded in the overlap
    """
    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    # Determine what sensor type to use
    avail_cams = {"isis", "csm"}
    cam_type = cam_type.lower()
    if cam_type not in cam_type:
        raise Exception(f'{cam_type} is not one of valid camera: {avail_cams}')

    points = []
    semi_major = ncg.config['spatial']['semimajor_rad']
    semi_minor = ncg.config['spatial']['semiminor_rad']

    # Determine the point distribution in the overlap geom
    geom = overlap.geom
    valid = compgeom.distribute_points_in_geom(geom, **distribute_points_kwargs, **kwargs)
    if not valid:
        warnings.warn('Failed to distribute points in overlap')
        return []

    # Setup the node objects that are covered by the geom
    nodes = []
    with ncg.session_scope() as session:
        for id in overlap.intersections:
            res = session.query(Images).filter(Images.id == id).one()
            nn = NetworkNode(node_id=id, image_path=res.path)
            nn.parent = ncg
            nodes.append(nn)

    print(f'Have {len(valid)} potential points to place.')
    for v in valid:
        lon = v[0]
        lat = v[1]

        # Calculate the height, the distance (in meters) above or
        # below the aeroid (meters above or below the BCBF spheroid).
        px, py = ncg.dem.latlon_to_pixel(lat, lon)
        height = ncg.dem.read_array(1, [px, py, 1, 1])[0][0]

        # Need to get the first node and then convert from lat/lon to image space
        node = nodes[0]
        if cam_type == "isis":
            try:
                line, sample = isis.ground_to_image(node["image_path"], lon, lat)
            except ProcessError as e:
                if 'Requested position does not project in camera model' in e.stderr:
                    print(f'point ({geocent_lon}, {geocent_lat}) does not project to reference image {node["image_path"]}')
                    continue
        if cam_type == "csm":
            lon_og, lat_og = oc2og(lon, lat, semi_major, semi_minor)
            x, y, z = reproject([lon_og, lat_og, height],
                                semi_major, semi_minor,
                                'latlon', 'geocent')
            # The CSM conversion makes the LLA/ECEF conversion explicit
            gnd = csmapi.EcefCoord(x, y, z)
            image_coord = node.camera.groundToImage(gnd)
            sample, line = image_coord.samp, image_coord.line

        # Extract ORB features in a sub-image around the desired point
        image_roi = roi.Roi(node.geodata, sample, line, size_x=size, size_y=size)
        image = image_roi.clip()
        try:
            interesting = extract_most_interesting(image)
        except:
            continue

        # kps are in the image space with upper left origin and the roi
        # could be the requested size or smaller if near an image boundary.
        # So use the roi upper left_x and top_y for the actual origin.
        left_x, _, top_y, _ = image_roi.image_extent
        newsample = left_x + interesting.x
        newline = top_y + interesting.y

        # Get the updated lat/lon from the feature in the node
        if cam_type == "isis":
            try:
                p = isis.point_info(node["image_path"], newsample, newline, point_type="image")
            except ProcessError as e:
                if 'Requested position does not project in camera model' in e.stderr:
                    print(node["image_path"])
                    print(f'interesting point ({newsample}, {newline}) does not project back to ground')
                    continue
            try:
                x, y, z = p["BodyFixedCoordinate"].value
            except:
                x, y, z = ["BodyFixedCoordinate"]

            if getattr(p["BodyFixedCoordinate"], "units", "None").lower() == "km":
                x = x * 1000
                y = y * 1000
                z = z * 1000
        elif cam_type == "csm":
            image_coord = csmapi.ImageCoord(newline, newsample)
            pcoord = node.camera.imageToGround(image_coord)
            # Get the BCEF coordinate from the lon, lat
            updated_lon_og, updated_lat_og, _ = reproject([pcoord.x, pcoord.y, pcoord.z],
                                                           semi_major, semi_minor, 'geocent', 'latlon')
            updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

            px, py = ncg.dem.latlon_to_pixel(updated_lat, updated_lon)
            updated_height = ncg.dem.read_array(1, [px, py, 1, 1])[0][0]


            # Get the BCEF coordinate from the lon, lat
            x, y, z = reproject([updated_lon_og, updated_lat_og, updated_height],
                                semi_major, semi_minor, 'latlon', 'geocent')

        # If the updated point is outside of the overlap, then revert back to the
        # original point and hope the matcher can handle it when sub-pixel registering
        updated_lon_og, updated_lat_og, updated_height = reproject([x, y, z], semi_major, semi_minor,
                                                             'geocent', 'latlon')
        updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

        if not geom.contains(shapely.geometry.Point(updated_lon, updated_lat)):
            lon_og, lat_og = oc2og(lon, lat, semi_major, semi_minor)
            x, y, z = reproject([lon_og, lat_og, height],
                                semi_major, semi_minor, 'latlon', 'geocent')
            updated_lon_og, updated_lat_og, updated_height = reproject([x, y, z], semi_major, semi_minor,
                                                                 'geocent', 'latlon')
            updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

        point_geom = shapely.geometry.Point(x, y, z)
        point = Points(overlapid=overlap.id,
                       apriori=point_geom,
                       adjusted=point_geom,
                       pointtype=2, # Would be 3 or 4 for ground
                       cam_type=cam_type)

        # Compute ground point to back project into measurtes
        gnd = csmapi.EcefCoord(x, y, z)

        for node in nodes:
            if cam_type == "csm":
                image_coord = node.camera.groundToImage(gnd)
                sample, line = image_coord.samp, image_coord.line
            if cam_type == "isis":
                try:
                    line, sample = isis.ground_to_image(node["image_path"], updated_lon, updated_lat)
                except ProcessError as e:
                    if 'Requested position does not project in camera model' in e.stderr:
                        print(f'interesting point ({geocent_lon},{geocent_lat}) does not project to image {node["image_path"]}')
                        continue

            point.measures.append(Measures(sample=sample,
                                           line=line,
                                           apriorisample=sample,
                                           aprioriline=line,
                                           imageid=node['node_id'],
                                           serial=node.isis_serial,
                                           measuretype=3,
                                           choosername='place_points_in_overlap'))

        if len(point.measures) >= 2:
            points.append(point)
    print(f'Able to place {len(points)} points.')
    Points.bulkadd(points, ncg.Session)
    return points

def place_points_in_image(image,
                          cam_type="csm",
                          size=71,
                          distribute_points_kwargs={},
                          ncg=None,
                          **kwargs):
    """
    Place points into an image and then attempt to place measures
    into all overlapping images. This function is funcitonally identical
    to place_point_in_overlap except it works on individual images.

    Parameters
    ----------
    image : obj
            An autocnet Images model object

    cam_type : str
               options: {"csm", "isis"}
               Pick what kind of camera model implementation to use

    size : int
           The size of the window used to extractor features to find an
           interesting feature to which the point is shifted.

    distirbute_points_kwargs : dict
                               Of optional arguments for distirbute_points_in_geom
    """
    # Arg checking
    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    # Determine what sensor type to use
    avail_cams = {"isis", "csm"}
    cam_type = cam_type.lower()
    if cam_type not in cam_type:
        raise Exception(f'{cam_type} is not one of valid camera: {avail_cams}')

    if not isinstance(image, Images):
        raise TypeError('image argument must be of type Images')

    # Logic
    geom = image.geom
    # Put down a grid of points over the image; the density is intentionally high
    valid = compgeom.distribute_points_in_geom(geom, **distribute_points_kwargs)
    
    points = []
    print(f'Have {len(valid)} potential points to place.')
    for v in valid:
        # Get the network nodes that intersect the point
        nodes = ncg.intersecting_nodes(v[1], v[0])  # lat, lon
                
        # Need to get the first node and then convert from lat/lon to image space
        node = nodes[0]
        if cam_type == "isis":
            try:
                line, sample = isis.ground_to_image(node["image_path"], lon, lat)
            except ProcessError as e:
                if 'Requested position does not project in camera model' in e.stderr:
                    print(f'point ({lon}, {lat}) does not project to reference image {node["image_path"]}')
                    continue
        if cam_type == "csm":
            lon_og, lat_og = oc2og(lon, lat, semi_major, semi_minor)
            x, y, z = reproject([lon_og, lat_og, height],
                                semi_major, semi_minor,
                                'latlon', 'geocent')
            # The CSM conversion makes the LLA/ECEF conversion explicit
            gnd = csmapi.EcefCoord(x, y, z)
            image_coord = node.camera.groundToImage(gnd)
            sample, line = image_coord.samp, image_coord.line

        # Extract ORB features in a sub-image around the desired point
        image_roi = roi.Roi(node.geodata, sample, line, size_x=size, size_y=size)
        image = image_roi.clip()
        try:
            interesting = extract_most_interesting(image)
        except:
            continue

        # kps are in the image space with upper left origin and the roi
        # could be the requested size or smaller if near an image boundary.
        # So use the roi upper left_x and top_y for the actual origin.
        left_x, _, top_y, _ = image_roi.image_extent
        newsample = left_x + interesting.x
        newline = top_y + interesting.y

        # Get the updated lat/lon from the feature in the node
        if cam_type == "isis":
            try:
                p = isis.point_info(node["image_path"], newsample, newline, point_type="image")
            except ProcessError as e:
                if 'Requested position does not project in camera model' in e.stderr:
                    print(node["image_path"])
                    print(f'interesting point ({newsample}, {newline}) does not project back to ground')
                    continue
            try:
                x, y, z = p["BodyFixedCoordinate"].value
            except:
                x, y, z = ["BodyFixedCoordinate"]

            if getattr(p["BodyFixedCoordinate"], "units", "None").lower() == "km":
                x = x * 1000
                y = y * 1000
                z = z * 1000
        elif cam_type == "csm":
            image_coord = csmapi.ImageCoord(newline, newsample)
            pcoord = node.camera.imageToGround(image_coord)
            # Get the BCEF coordinate from the lon, lat
            updated_lon_og, updated_lat_og, _ = reproject([pcoord.x, pcoord.y, pcoord.z],
                                                           semi_major, semi_minor, 'geocent', 'latlon')
            updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

            px, py = ncg.dem.latlon_to_pixel(updated_lat, updated_lon)
            updated_height = ncg.dem.read_array(1, [px, py, 1, 1])[0][0]


            # Get the BCEF coordinate from the lon, lat
            x, y, z = reproject([updated_lon_og, updated_lat_og, updated_height],
                                semi_major, semi_minor, 'latlon', 'geocent')

        # If the updated point is outside of the overlap, then revert back to the
        # original point and hope the matcher can handle it when sub-pixel registering
        updated_lon_og, updated_lat_og, updated_height = reproject([x, y, z], semi_major, semi_minor,
                                                             'geocent', 'latlon')
        updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

        if not geom.contains(shapely.geometry.Point(updated_lon, updated_lat)):
            lon_og, lat_og = oc2og(lon, lat, semi_major, semi_minor)
            x, y, z = reproject([lon_og, lat_og, height],
                                semi_major, semi_minor, 'latlon', 'geocent')
            updated_lon_og, updated_lat_og, updated_height = reproject([x, y, z], semi_major, semi_minor,
                                                                 'geocent', 'latlon')
            updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

        point_geom = shapely.geometry.Point(x, y, z)
        
        # Insert a spatial query to find which overlap this is in.
        with ncg.session_scope() as session:
            oid = session.query(Overlay.id).filter(Overlay.geom.ST_Contains(point_geometry)).one()[0]

        point = Points(overlapid=oid,
                       apriori=point_geom,
                       adjusted=point_geom,
                       pointtype=2, # Would be 3 or 4 for ground
                       cam_type=cam_type)
        
        for node in nodes:
            insert = True
            if cam_type == "csm":
                image_coord = node.camera.groundToImage(gnd)
                sample, line = image_coord.samp, image_coord.line
            if cam_type == "isis":
                try:
                    line, sample = isis.ground_to_image(node["image_path"], updated_lon, updated_lat)
                except ProcessError as e:
                    if 'Requested position does not project in camera model' in e.stderr:
                        print(f'interesting point ({lon},{lat}) does not project to image {node["image_path"]}')
                        insert = False
            
            point.measures.append(Measures(sample=sample,
                                           line=line,
                                           apriorisample=sample,
                                           aprioriline=line,
                                           imageid=node['node_id'],
                                           serial=node.isis_serial,
                                           measuretype=3,
                                           choosername='place_points_in_image'))

        if len(point.measures) >= 2:
            points.append(point)
    print(f'Able to place {len(points)} points.')
    Points.bulkadd(points, ncg.Session)

def place_ground_control_point(coordinate,
                               ground_mosaic, 
                               size=(100,100),
                               cam_type='isis', 
                               template_kwargs={"image_size":(121,121), "template_size":(61, 61)},
                               cost_func=lambda x,y:y,
                               threshold=0.6,
                               ncg=None, 
                               Session=None, 
                               **kwargs):

    """

    """

    def find_reference_point(lon, lat):
        linessamples = isis.point_info(ground_mosaic.file_name, lon, lat, 'ground')
        if linessamples is None:
            print('unable to find point in ground image')
            return [None] * 4
        line = linessamples[0].get('Line')
        sample = linessamples[0].get('Sample')

        image = roi.Roi(ground_mosaic, sample, line, size_x=size[0], size_y=size[1])
        image_roi = image.clip(dtype="uint64")

        # Get the most interesting feature in the ROI and recenter
        interesting = extract_most_interesting(image_roi,  extractor_parameters={'nfeatures':30})

        # kps are in the image space with upper left origin, so convert to
        # center origin and then convert back into full image space
        left_x, _, top_y, _ = image.image_extent
        newsample = left_x + interesting.x
        newline = top_y + interesting.y

        # Get the lon, lat from the new iamge coordinate
        newpoint = isis.point_info(ground_mosaic.file_name, newsample, newline, 'image')
        
        lon = newpoint[0].get('PositiveEast360Longitude')
        lat = newpoint[0].get('PlanetocentricLatitude')

        # Removing because these points sbould be, by definition, inside the poly.
        #if not (compgeom.xy_in_polygon(lon, lat, fp_poly)):
        #        print('Interesting point not in mosaic area, ignore')
        #        return

        return lon, lat, newsample, newline

    if isinstance(ground_mosaic, str):
        ground_mosaic = GeoDataset(ground_mosaic)

    lon, lat = coordinate

    print(f'Attempting to place a ground point near {lon}, {lat}')
    lon, lat, base_x, base_y = find_reference_point(lon, lat)
    
    if not lon:
        return

    px, py = ncg.dem.latlon_to_pixel(lat, lon)
    height = ncg.dem.read_array(1, [px, py, 1, 1])[0][0]

    semi_major = ncg.config['spatial']['semimajor_rad']
    semi_minor = ncg.config['spatial']['semiminor_rad']

    # reprojection takes ographic lat
    lon_og, lat_og = oc2og(lon, lat, semi_major, semi_minor)
    x, y, z = reproject([lon_og, lat_og, height],
                        semi_major, semi_minor,
                        'latlon', 'geocent')
    
    # Create the point
    point_geom = shapely.geometry.Point(x, y, z)
    point = Points(apriori=point_geom,
                    adjusted=point_geom,
                    pointtype=3,
                    cam_type=cam_type,
                    identifier='PlaceGroundPoints')

    
    # Find all nodes (images) that intersect the point and place an apriori measure
    nodes = ncg.intersecting_nodes(lat, lon)
    if not nodes:
        print('No intersecting nodes at these coordinates.')

    best = -np.inf
    reference_node = None
    reference_x = None
    reference_y = None

    for node in nodes:
        newx, newy, dist, metric, _ = geom_match(ground_mosaic, node.geodata, base_x, base_y, template_kwargs=template_kwargs, **kwargs)
        if newx is None:
            continue

        cost = cost_func(dist, metric)
        if cost >= threshold:
            best = cost
            reference_node = node
            reference_x = newx
            reference_y = newy

    # 1+ matches to the base were good. The reference node is the best
    if reference_node:
        # Then match the best CTX image to all other CTX images that overalap

        # Add the best measure first so that it is used as the reference

        ref_measure = Measures(sample=reference_x,
                                    line=reference_y,
                                    apriorisample=reference_x,
                                    aprioriline=reference_y,
                                    imageid=node['node_id'],
                                    serial=node.isis_serial,
                                    measuretype=3,
                                    choosername='place_ground_control_points')
        point.measures.append(ref_measure)
        # Now use the matched reference node to get a lat/lon 
        pointinfo = isis.point_info(reference_node["image_path"], reference_x, reference_y, point_type="image")

        # This is ocentric because we within the ISIS pipeline the whole time
        lon = pointinfo['PositiveEast360Longitude'].value
        lat = pointinfo['PlanetocentricLatitude'].value
    else:
        print('Failed to place a reference point.')
        return
    # Remove the reference node from the node list so we do not double add a measure
    nodes.remove(reference_node)

    # Attempt to add measures to the other nodes
    measures = add_measures_to_nodes(nodes, 
                                    (lon, lat), 
                                    measuretype=3, # ground
                                    cam_type=cam_type, 
                                    choosername='place_ground_control_points')

    # Add the new measures to the point object
    point.measures += measures

    with ncg.session_scope() as session:
        session.add(point)
        session.commit()
        print(f'Sucessfully added point {point.id} to the database with {len(point.measures)} measures.')

    return
