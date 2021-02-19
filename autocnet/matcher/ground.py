import os

from plio.io.io_gdal import GeoDataset


from autocnet.io.db.model import Points, Measures, Images
from autocnet.graph.node import NetworkNode
from autocnet.matcher.subpixel import check_geom_func, check_match_func

from autocnet.spatial import isis
from autocnet.io.db.model import Images
from autocnet.transformation import roi


def find_ground_reference(point, 
                           ncg=None, 
                           Session=None,
                           geom_func='simple', 
                           match_func='classic', 
                           match_kwargs={},
                           geom_kwargs={"size_x": 16, "size_y": 16},
                           threshold=0.9,
                           cost_func=lambda x,y: (0*x)+y,
                           verbose=False):
    print(point.id)
    geom_func = check_geom_func(geom_func)
    match_func = check_match_func(match_func)
    
    # Get the roi to match from the base image
    with ncg.session_scope() as session:
        measures = session.query(Measures).filter(Measures.pointid == point.id).all()

        for m in measures:
            if m.measuretype == 0:
                base = m
                bsample = base.sample
                bline = base.line
        baseimage = base.serial # We are piggybacking the base image name onto the measure serial.
    if not os.path.exists(baseimage):
        raise FileNotFoundError(f'Unable to find {baseimage} to register the data to.')
    
    # Get the base image and the roi extracted that the image data will register to
    baseimage = GeoDataset(baseimage)
    
    # Select the images that the point is in.
    cost = -1
    sample = None
    line = None
    best_node = None
    
    with ncg.session_scope() as session:
        images = session.query(Images).filter(Images.geom.ST_Intersects(point._geom)).all()
        
        nodes = []
        for image in images:
            node = NetworkNode(node_id=image.id, image_path=image.path)
            nodes.append(node)
      
    for node in nodes:
        node.geodata
        image_geodata = node.geodata

        x, y, dist, metrics, _ = geom_func(baseimage, image_geodata,
                                            bsample, bline,
                                            match_func = match_func,
                                            match_kwargs = match_kwargs,
                                            verbose=verbose,
                                            **geom_kwargs)
        if x == None:
            print(f'Unable to match image {node["image_name"]} to {baseimage}.')
            continue

        current_cost = cost_func(dist, metrics)
        print(f'Results returned: {current_cost}.')
        if current_cost >= cost and current_cost >= threshold:
            cost = current_cost
            sample = x
            line = y
            best_node = node
        else:
            print(f'Cost function not met. Unable to use {node["image_name"]} as reference')
    if sample == None:
        print('Unable to register this point to a ground source.')
        return
    
    # A reference measure has been identified. This measure matched successfully to the ground.
    # Get the lat/lon from the sample/line
    reference_node = best_node
    print('Success...')
    # Setup the measures
    
    m = Measures(sample=sample,
                line=line,
                apriorisample=sample,
                aprioriline=line,
                imageid=node['node_id'],
                serial=node.isis_serial,
                measuretype=3,
                choosername='add_measures_to_ground')

    with ncg.session_scope() as session:
        point = session.query(Points).filter(Points.id == point.id).one()

        point.measures.append(m)
        point.reference_index = len(point.measures) - 1  # The measure that was just appended is the new reference

    print('successfully added a reference measure to the database.')    
