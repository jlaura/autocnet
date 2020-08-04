import copy

from sqlalchemy import event
from sqlalchemy_utils import database_exists, create_database

from autocnet.io.db.triggers import valid_point_function, valid_point_trigger, valid_geom_function, valid_geom_trigger, ignore_image_function, ignore_image_trigger
from autocnet.io.db.model import Base, Measures, Images, Overlay, Edges, Costs, Matches, Cameras, Points, Keypoints


def create_new_from_existing(source_ncg, name):
    """
    Given a source NetworkCandidateGraph, create a destination NetworkCandidateGraph with
    an identical table mapping into a user specified name. This is a convenience function
    when seeking to merge networks. This function will create a new destination network 
    into a new schema that is structurally identical to the source_ncg.

    Parameters
    ----------
    source_ncg : object
                 NetworkCandidateGraph object (can be empty) that has a database config

    schema_name : str
                  A new schema name that will be created in the database associated with the 
                  source_ncg

    Returns
    -------
    destination_ncg : object
                      NetworkCandidateGraph object associated with the new schema.
    """
    # Create the ncg for the destination to get the DB instantiated
    destination_ncg = type(source_ncg)()
    # Have to deep copy or else we are messing with the source config dict...
    destination_config = copy.deepcopy(source_ncg.config)  
    # Update the schema name so that this is a new project
    destination_config['database']['name'] = name
    destination_ncg.config_from_dict(destination_config)
    return destination_ncg

def try_db_creation(engine, config):

    # Create the database
    if not database_exists(engine.url):
        create_database(engine.url, template='template_postgis')  # This is a hardcode to the local template

    # Trigger that watches for points that should be active/inactive
    # based on the point count.
    if not engine.dialect.has_table(engine, "points"):
        event.listen(Base.metadata, 'before_create', valid_point_function)
        event.listen(Measures.__table__, 'after_create', valid_point_trigger)
        event.listen(Base.metadata, 'before_create', valid_geom_function)
        event.listen(Images.__table__, 'after_create', valid_geom_trigger)
        event.listen(Base.metadata, 'before_create', ignore_image_function)
        event.listen(Images.__table__, 'after_create', ignore_image_trigger)

    Base.metadata.bind = engine

    # Set the class attributes for the SRIDs
    spatial = config['spatial']
    latitudinal_srid = spatial['latitudinal_srid']
    rectangular_srid = spatial['rectangular_srid']

    Points.rectangular_srid = rectangular_srid
    Points.semimajor_rad = spatial['semimajor_rad']
    Points.semiminor_rad = spatial['semiminor_rad']
    for cls in [Points, Overlay, Images, Keypoints, Matches]:
        setattr(cls, 'latitudinal_srid', latitudinal_srid)

    # If the table does not exist, this will create it. This is used in case a
    # user has manually dropped a table so that the project is not wrecked.
    Base.metadata.create_all(tables=[Overlay.__table__,
                                    Edges.__table__, Costs.__table__, Matches.__table__,
                                    Cameras.__table__, Points.__table__,
                                    Measures.__table__, Images.__table__,
                                    Keypoints.__table__])
