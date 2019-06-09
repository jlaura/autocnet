import os
import pytest
from unittest import mock
from autocnet.graph.network import NetworkCandidateGraph


@pytest.fixture(params=['test_a'])
def testdir(request):
    thisfile = os.path.realpath(__file__)
    thisdir = os.path.dirname(thisfile)
    testpath = os.path.join(thisdir, f'integration_tests/{request.param}')
    return testpath


@mock.patch('gdal.Open', return_value=lambda x: x)
def test_integration(testdir):
    filelist = os.path.join(testdir, 'filelist.lis')
    ncg = NetworkCandidateGraph.from_filelist(filelist)
    assert False

