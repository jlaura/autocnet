import copy
from scipy.spatial import cKDTree

from autocnet.io.db.model import Points, Measures
from autocnet.transformation.spatial import ll_to_eqc

def create_new_from_existing(source_ncg, schema_name):
    """
    Given a source NetworkCandidateGraph, create a destination NetworkCandidateGraph with
    an identical table mapping into a user specified schema. This is a convenience function
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
    destination_ncg = NetworkCandidateGraph()
    # Have to deep copy or else we are messing with the source config dict...
    destination_config = copy.deepcopy(source_ncg.config)  
    # Update the schema name so that this is a new project
    destination_config['database']['schema'] = schema_name
    destination_ncg.config_from_dict(destination_config)
    return destination_ncg

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


def _subpixel_register_merged_measures(ncg, to_match, **kwargs):
    mids = []
    with ncg.session_scope() as session:
        for serial, pointid in to_match:
            mid = session.query(Measures).filter(Measures.serial == serial, Measures.pointid == pointid).one()
            mids.append(mid.id)

    # Then apply the registration on the cluster
    sql = f"SELECT * FROM merged.measures WHERE measures.id IN {tuple(mids)}"
    destination_ncg.apply('matcher.subpixel.subpixel_register_measure', on='measures', query_string=sql, **kwargs)

def _gdfnearest(gdA, gdB, semimajor, semiminor):
    """
    Compute the nearest neighbor between two pandas data frames. This func also
    takes the semi-major and semi-minor axes of the body as it projects the points
    from lat/lon to equirectangular space.
    
    Parameters
    ----------
    gdA : DataFrame
          With an 'id' column and a 'geom' column. This is the source dataframe
          that is used to query the destination dataframe (gdB).
          
    gdB : DataFrame
          With an 'id' column and a 'geom' column or an empty pandas data frame.
          This is the destination dataframe that against which the source (gdA)
          queries.
          
    semimajor : numeric
                Semi-major axis of the body
                
    semiminor : numeric
                Semi-minor axis of the body
                
    Returns
    -------
    gdf : DataFrame
          Containing the source point, the nearest destination point, and an added 
          'dist' column. If the destination is empty, the row contains only the source
          point and a dist equal to zero.
          
    """
    # Rename the id columns because these ar
    gdA.rename(columns={'id':'source_id'}, inplace=True)
    gdB.rename(columns={'id':'destin_id'}, inplace=True)
    # Convert the geom obj to lists
    nA = np.array(list(gdA['geom'].apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB['geom'].apply(lambda x: (x.x, x.y))))
    
    # Project into equirectangular (meters unit) so that distances
    # are intuitive for defining a threshold
    nA[:,0], nA[:,1] = lla_to_eqc(nA, semimajor, semiminor)
    
    # If nB is empty, set the distances to infinity

    if len(nB) == 0:
        dist = pd.Series(np.zeros(len(nA)), name='dist')
        dist[:] = np.inf
        gdf = pd.concat([gdA.reset_index(drop=True), dist], axis=1)
        return gdf
    
    # Else compute the distances
    nB[:,0], nB[:,1] = lla_to_eqc(nB, semimajor, semiminor)
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    
    # Find the nearest
    gdf = pd.concat(
        [gdA.reset_index(drop=True), gdB.loc[idx, gdB.columns != 'geometry'].reset_index(drop=True),
         pd.Series(dist, name='dist')], axis=1)
    return gdf

def merge(destination_ncg, 
          ncgs, 
          threshold=24,
          subpixel_kwargs={}):
    """
    Given a destination network and one or more input networks, merge the input network(s)
    into the destination network. If we have a network A, this function takes one or more 
    input networks (B,C,D) and merges B,C,D into A resulting in a network composed of
    A,B,C,D.

    The method copies the points, measures, images, and overlaps from the input networks
    into the destination networks.

    Parameters
    ----------
    destination_ncg : obj
                      An AutoCNet network candidate graph.

    ncgs : obj / list
           Either a single AutoCNet network candidate graph or a list of network
           candidate graphs.

    threshold : numeric 
                The distance, in meters, within which measures will be considered 
                co-incident. When measures are considered co-incident, they are
                re-subpixel registered to the reference measure in the
                destination graph.

    subpixel_kwargs : dict
                      With arguments passed to the subpixel matcher for those
                      measures that are within the threshold.
    """

    if isinstance(ncgs, NetworkCandidateGraph):
        ncgs = [ncgs]
    
    # Ensure that all the necessary images are in the destination database
    merge_images(destination_ncg, ncgs)

    # Ensure that all the overlaps are copied. This is naive at this point becase
    # the overlaps are not used once we are at this point. This happens because
    # the overlaps are a foreign key.
    merge_overlaps(destination_ncg, ncgs)

    # Merge the points/measures from each network. When points are within the
    # threshold, the measures are merged and then the merging measures are
    # asynchronously subpixel registered on the cluster.
    for to_be_merged_ncg in ncgs:
    
        # Compute the nearest neighbor for each point in the db
        gdA = to_be_merged_ncg.points
        gdB = destination_ncg.points
        gdf = _gdfnearest(gdA, gdB, semimajor, semiminor)

        # Create an emtpy container to hold objects so that we can efficiently insert into the DB
        to_add = []
        to_rematch = []
        ids_that_should_not_migrate = []
        with to_be_merged_ncg.session_scope() as s_session, destination_ncg.session_scope() as d_session:  
            for i, row in gdf.iterrows():
                if row.dist <= threshold:
                    updated_point_id = row['destin_id']
                    ids_that_should_not_migrate.append(row['source_id'])
                    # Determine which measures exist in the source that are not already in the destination
                    # Do this using the set difference of the serial numbers
                    destin_measures = d_session.query(Measures).filter(Measures.pointid == row.destin_id).all()
                    source_measures = s_session.query(Measures).filter(Measures.pointid == row.source_id).all()

                    destin_serials = set([i.serial for i in destin_measures])
                    source_serials = set([i.serial for i in source_measures])
                    serials_not_in_destin = source_serials - destin_serials

                    measures_to_register_and_add = [i for i in source_measures if i.serial in serials_not_in_destin] 

                    for m in measures_to_register_and_add:
                        new_measure = m.duplicate()
                        new_measure.pointid = row['destin_id']
                        to_rematch.append((new_measure.serial, new_measure.pointid))
                        to_add.append(new_measure)

                else:
                    # The point is outside the threshold, so simply copy it into the new DB
                    point_to_copy = s_session.query(Points).filter(Points.id == row.source_id).one()
                    to_add.append(point_to_copy.duplicate())

            d_session.add_all(to_add)

        # Subpixel register the measures that were added and within the given threshold. Since this happens
        # on the cluster, it is asynchronous. This happens as a second pass so that the above session add
        # gets all the primary keys setup.
        if to_rematch:
            _subpixel_register_merged_measures(destination_ncg, to_rematch, **subpixel_kwargs)
