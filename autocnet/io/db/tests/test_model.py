from datetime import datetime
import json

import numpy as np
import pandas as pd
import pytest
import sqlalchemy
from unittest.mock import MagicMock, patch

from autocnet.io.db import model
from autocnet import Session, engine
from autocnet.graph.network import NetworkCandidateGraph

from shapely.geometry import MultiPolygon, Polygon, Point

@pytest.fixture
def tables():
    return engine.table_names()

@pytest.fixture
def session(tables, request):
    session = Session()

    def cleanup():
        session.rollback()  # Necessary because some tests intentionally fail
        for t in reversed(tables):
            # Skip the srid table
            if t != 'spatial_ref_sys':
                session.execute(f'TRUNCATE TABLE {t} CASCADE')
            # Reset the autoincrementing
            if t in ['Images', 'Cameras', 'Matches', 'Measures']:
                session.execute(f'ALTER SEQUENCE {t}_id_seq RESTART WITH 1')
        session.commit()

    request.addfinalizer(cleanup)

    return session

def test_keypoints_exists(tables):
    assert model.Keypoints.__tablename__ in tables

def test_edges_exists(tables):
    assert model.Edges.__tablename__ in tables

def test_costs_exists(tables):
    assert model.Costs.__tablename__ in tables

def test_matches_exists(tables):
    assert model.Matches.__tablename__ in tables

def test_cameras_exists(tables):
    assert model.Cameras.__tablename__ in tables

def test_measures_exists(tables):
    assert model.Measures.__tablename__ in tables

def test_create_camera_without_image(session):
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        model.Cameras.create(session, **{'image_id':1})

def test_create_camera(session):
    #with pytest.raises(sqlalchemy.exc.IntegrityError):
    c = model.Cameras.create(session)
    res = session.query(model.Cameras).first()
    assert c.id == res.id

def test_create_camera_unique_constraint(session):
    model.Images.create(session, **{'id':1})
    data = {'image_id':1}
    model.Cameras.create(session, **data)
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        model.Cameras.create(session, **data)

def test_images_exists(tables):
    assert model.Images.__tablename__ in tables

@pytest.mark.parametrize('data', [
    {'id':1},
    {'name':'foo',
     'path':'/neither/here/nor/there'},
    ])
def test_create_images(session, data):
    i = model.Images.create(session, **data)
    resp = session.query(model.Images).filter(model.Images.id==i.id).first()
    assert i == resp

@pytest.mark.parametrize('data', [
    {'id':1},
    {'serial':'foo'}
])
def test_create_images_constrined(session, data):
    """
    Test that the images unique constraint is being observed.
    """
    model.Images.create(session, **data)
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        model.Images.create(session, **data)

def test_overlay_exists(tables):
    assert model.Overlay.__tablename__ in tables

@pytest.mark.parametrize('data', [
    {'id':1},
    {'id':1, 'intersections':[1,2,3]},
    {'id':1, 'intersections':[1,2,3],
     'geom':Polygon([(0,0), (1,0), (1,1), (0,1), (0,0)])}

])
def test_create_overlay(session, data):
    d = model.Overlay.create(session, **data)
    resp = session.query(model.Overlay).filter(model.Overlay.id == d.id).first()
    assert d == resp

def test_points_exists(tables):
    assert model.Points.__tablename__ in tables

@pytest.mark.parametrize("data", [
    {'id':1, 'pointtype':2},
    {'pointtype':2, 'identifier':'123abc'},
    {'pointtype':3, 'apriori':Point(0,0,0)},
    {'pointtype':3, 'adjusted':Point(0,0,0)},
    {'pointtype':2, 'adjusted':Point(1,1,1), 'active':True}
])
def test_create_point(session, data):
    p = model.Points.create(session, **data)
    resp = session.query(model.Points).filter(model.Points.id == p.id).first()
    assert p == resp

@pytest.mark.parametrize("data, expected", [
    ({'pointtype':3, 'adjusted':Point(0,-1,0)}, Point(-90, 0)),
    ({'pointtype':3}, None)
])
def test_create_point_geom(session, data, expected):
    p = model.Points.create(session, **data)
    resp = session.query(model.Points).filter(model.Points.id == p.id).first()
    assert resp.geom == expected

@pytest.mark.parametrize("data, new_adjusted, expected", [
    ({'pointtype':3, 'adjusted':Point(0,-1,0)}, None, None),
    ({'pointtype':3, 'adjusted':Point(0,-1,0)}, Point(0,1,0), Point(90, 0)),
    ({'pointtype':3}, Point(0,-1,0), Point(-90, 0))
])
def test_update_point_geom(session, data, new_adjusted, expected):
    p = model.Points.create(session, **data)
    p.adjusted = new_adjusted
    session.commit()
    resp = session.query(model.Points).filter(model.Points.id == p.id).first()
    assert resp.geom == expected

def test_measures_exists(tables):
    assert model.Measures.__tablename__ in tables

@pytest.mark.parametrize("measure_data, point_data, image_data", [({'id': 1, 'pointid': 1, 'imageid': 1, 'serial': 'ISISSERIAL', 'measuretype': 3, 'sample': 0, 'line': 0},
                                                                   {'id':1, 'pointtype':2},
                                                                   {'id':1, 'serial': 'ISISSERIAL'})])
@patch('plio.io.io_controlnetwork.from_isis', return_value = pd.DataFrame.from_dict({'id': [1],
                                                                                     'serialnumber': ['ISISSERIAL'],
                                                                                     'pointJigsawRejected': [False],
                                                                                     'jigsawRejected': [False],
                                                                                     'sampleResidual': [0.1],
                                                                                     'lineResidual': [0.1],
                                                                                     'samplesigma': [0],
                                                                                     'linesigma': [0],
                                                                                     'adjustedCovar': [[]],
                                                                                     'apriorisample': [0],
                                                                                     'aprioriline': [0]}))
def test_jigsaw_append(mockFunc, session, measure_data, point_data, image_data):
    model.Images.create(session, **image_data)
    model.Points.create(session, **point_data)
    model.Measures.create(session, **measure_data)
    resp = session.query(model.Measures).filter(model.Measures.id == 1).first()
    assert resp.liner == None
    assert resp.sampler == None

    NetworkCandidateGraph.update_from_jigsaw(session, '/Some/Path/To/An/ISISNetwork.cnet')
    resp = session.query(model.Measures).filter(model.Measures.id == 1).first()
    assert resp.liner == 0.1
    assert resp.sampler == 0.1

def test_null_footprint(session):
    i = model.Images.create(session, footprint_latlon=None,
                                      serial = 'serial')
    assert i.footprint_latlon is None

def test_broken_bad_geom(session):
    # An irreperablly damaged poly
    geom = MultiPolygon([Polygon([(0,0), (1,1), (1,2), (1,1), (0,0)])])
    i = model.Images.create(session, footprint_latlon=geom,
                                      serial = 'serial')
    resp = session.query(model.Images).filter(model.Images.id==i.id).one()
    assert resp.active == False

def test_fix_bad_geom(session):
    geom = MultiPolygon([Polygon([(0,0), (0,1), (1,1), (0,1), (1,1), (1,0), (0,0) ])])
    i = model.Images.create(session, footprint_latlon=geom,
                                     serial = 'serial' )
    resp = session.query(model.Images).filter(model.Images.id==i.id).one()
    assert resp.active == True
    assert resp.footprint_latlon == MultiPolygon([Polygon([(0,0), (0,1), (1,1), (1,0), (0,0) ])])
