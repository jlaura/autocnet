from itertools import combinations
from shapely.ops import cascaded_union
import sqlalchemy
from sqlalchemy.orm import subqueryload
import geoalchemy2

from autocnet.io.db.model import Points


def insert_points_notin_geom(destination_network, networks, srid=949900):
    """
    Given a destination network and a list of networks, compute the intersection of
    of the image boundry footprints. Any points inside of these areas are from overlapping
    networks. This function finds all points outside the intersection and then
    adds them to the database. Since these points do not already exist in the DB they can
    be added without any need to merge.

    Parameters
    ----------
    destination_network : object
                          NetworkCandidateGraph object (can be empty)

    networks : list
               of NetworkCandidateGraphs objects from which images are merged

    srid : int
           The SRID of the geometry that is computed from the intersection of 
           the network footprints.
    Returns
    -------
    None
    """
    all_networks = networks + [destination_network]
    intersection = cascaded_union([a.intersection(b) for a, b in combinations([i.footprint for i in all_networks], 2)])
    # Convert the geom to a geoalchemy2 type for DB queries
    geom = geoalchemy2.shape.from_shape(intersection, srid=srid)
    
    for network in networks:
        with network.session_scope() as s_session:
            # Spatial filter for all the points not in the intersection and then eager load the associated measures
            to_be_added = s_session.query(Points).filter(sqlalchemy.not_(geoalchemy2.functions.ST_Within(Points.geom, geom))).options(subqueryload(Points.measures)).all()
            
            # Convert the rows to dicts
            add_as_dicts = [obj.to_dict() for obj in to_be_added]
            
            # Update the primary keys for the Points, Measures since these are autoincrementing 
            for point in add_as_dicts:
                point.pop('id')
                for measure in point['measures']:
                    measure.pop('pointid')
                    measure.pop('id')
            
            with destination_network.session_scope() as d_session:
                d_session.bulk_insert_mappings(Points, add_as_dicts)
    
    return
networks = [ncgA, ncgB]
intersection = cascaded_union([a.intersection(b) for a, b in combinations([i.footprint for i in networks], 2)])

def merge_images(destination_network, networks):
    """
    Given a destination network (database) and a list of network candidate graphs
    insert all images not in the destination network from the input list of
    networks.

    Parameters
    ----------
    destination_network : object
                          NetworkCandidateGraph object (can be empty)

    networks : list
               of NetworkCandidateGraphs objects from which images are merged

    Returns
    -------
    None
    """

    sql = """INSERT INTO {0}.images 
    SELECT t.*
    FROM
    (SELECT * FROM {0}.images
    UNION 
    SELECT * FROM {1}.images) AS t
    ON CONFLICT DO NOTHING"""

    with destination_network.session_scope() as d_session:
        d_schema = destination_network.schema
        for n in networks:
            s_schema = n.schema
            d_session.execute(sql.format(d_schema, s_schema))

def merge_overlaps(destination_network, networks):
    """
    Naively merges the overlap tables from the list of networks into the destination_network. This
    naive merge should be fine for now because we do not use the overlap/overlay table once
    we are at the point that the projects are merging. 

    TODO: update to recompute the overlaps for a true merge

    Parameters
    ----------
    destination_network : object
                          NetworkCandidateGraph object (can be empty)

    networks : list
               of NetworkCandidateGraphs objects from which images are merged

    Returns
    -------
    None  
    """
    
    sql = """INSERT INTO {0}.overlay
    SELECT *
    FROM
    {1}.overlay
    ON CONFLICT DO NOTHING"""
    with destination_network.session_scope() as d_session:
        d_schema = destination_network.schema
        for network in networks:
            s_schema = network.schema
            d_session.execute(sql.format(d_schema, s_schema))