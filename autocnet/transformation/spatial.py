import pyproj

def reproject(record, semi_major, semi_minor, source_proj, dest_proj, **kwargs):
    """
    Thin wrapper around PyProj's Transform() function to transform 1 or more three-dimensional
    point from one coordinate system to another. If converting between Cartesian
    body-centered body-fixed (BCBF) coordinates and Longitude/Latitude/Altitude coordinates,
    the values input for semi-major and semi-minor axes determine whether latitudes are
    planetographic or planetocentric and determine the shape of the datum for altitudes.
    If semi_major == semi_minor, then latitudes are interpreted/created as planetocentric
    and altitudes are interpreted/created as referenced to a spherical datum.
    If semi_major != semi_minor, then latitudes are interpreted/created as planetographic
    and altitudes are interpreted/created as referenced to an ellipsoidal datum.

    Parameters
    ----------
    record : object
             Pandas series object

    semi_major : float
                 Radius from the center of the body to the equater

    semi_minor : float
                 Radius from the pole to the center of mass

    source_proj : str
                         Pyproj string that defines a projection space ie. 'geocent'

    dest_proj : str
                      Pyproj string that defines a project space ie. 'latlon'

    Returns
    -------
    : list
      Transformed coordinates as y, x, z

    """
    source_pyproj = pyproj.Proj(proj=source_proj, a=semi_major, b=semi_minor, lon_wrap=180)
    dest_pyproj = pyproj.Proj(proj=dest_proj, a=semi_major, b=semi_minor, lon_wrap=180)

    y, x, z = pyproj.transform(source_pyproj, dest_pyproj, record[0], record[1], record[2], **kwargs)
    return y, x, z

def ll_to_eqc(points, semimajor, semiminor):
    """
    Project from Lat/Lon to Equirectangular. This func works with points
    in 2D.
    
    Parameters
    ----------
    points : ndarray
             (n,2) array of ppoint coordinates in form (lat, lon)
             
    semimajor : numeric
                Semi-major axis of the body
                
    semiminor : numeric
                Semi-minor axis of the body
                
    Returns
    -------
     : ndarray
       (n,2) array of coordinates in meters in an equirectangular projection
       
    """
    ll = f"+proj=latlon +a={semimajor} +b={semiminor}"
    eqc = f"+proj=eqc +units=m +a={semimajor} +b={semiminor}"
    
    return pyproj.transform(ll, eqc, points[:,0], points[:,1])