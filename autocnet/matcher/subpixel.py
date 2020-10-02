import json
from math import modf, floor
import numpy as np

from skimage.feature import register_translation
from skimage import transform as tf

from matplotlib import pyplot as plt

from plio.io.io_gdal import GeoDataset
from pysis.exceptions import ProcessError

from autocnet.matcher.naive_template import pattern_match, pattern_match_autoreg
from autocnet.matcher import ciratefi
from autocnet.io.db.model import Measures, Points, Images, JsonEncoder
from autocnet.graph.node import NetworkNode
from autocnet.transformation import roi
from autocnet import spatial
from autocnet.utils.utils import bytescale

import geopandas as gpd
import pandas as pd

import pvl

isis2np_types = {
        "UnsignedByte" : "uint8",
        "SignedWord" : "int16",
        "Real" : "float64"
}

# TODO: look into KeyPoint.size and perhaps use to determine an appropriately-sized search/template.
def _prep_subpixel(nmatches, nstrengths=2):
    """
    Setup the data strutures to return for subpixel matching.

    Parameters
    ----------
    nmatches : int
                The number of pixels to be subpixel matches

    nstrengths : int
                    The number of 'strength' values to be returned
                    by the subpixel matching method.

    Returns
    -------
    shifts_x : ndarray
               (nmatches, 1) to store the x_shift parameter

    shifts_y : ndarray
               (nmatches, 1) to store the y_shift parameter

    strengths : ndarray
                (nmatches, nstrengths) to store the strengths for each point

    new_x : ndarray
            (nmatches, 1) to store the updated x coordinates

    new_y : ndarray
            (nmatches, 1) to store the updated y coordinates
    """
    # Setup to store output to append to dataframes
    shifts_x = np.zeros(nmatches)
    shifts_y = np.zeros(nmatches)
    strengths = np.zeros((nmatches, nstrengths))

    new_x = np.empty(nmatches)
    new_y = np.empty(nmatches)

    return shifts_x, shifts_y, strengths, new_x, new_y

def check_image_size(imagesize):
    """
    Given an x,y tuple, ensure that the values
    are odd. Used by the subpixel template to also ensure
    that the template size is the one requested and not 2x
    the template size.

    Parameters
    ----------
    imagesize : tuple
                in the form (size_x, size_y)
    """
    if isinstance(imagesize, int):
        imagesize = (imagesize, imagesize)

    x = imagesize[0]
    y = imagesize[1]

    if x % 2 == 0:
        x += 1
    if y % 2 == 0:
        y += 1
    x = floor(x/2)
    y = floor(y/2)
    return x,y

def clip_roi(img, center_x, center_y, size_x=200, size_y=200, dtype="uint64"):
    """
    Given an input image, clip a square region of interest
    centered on some pixel at some size.

    Parameters
    ----------
    img : ndarray or object
          The input image to be clipped or an object
          with a read_array method that takes a pixels
          argument in the form [xstart, ystart, xstop, ystop]

    center_x : Numeric
               The x coordinate to the center of the roi

    center_y : Numeric
               The y coordinate to the center of the roi

    img_size : int
               1/2 of the total image size. This value is the
               number of pixels grabbed from each side of the center

    Returns
    -------
    clipped_img : ndarray
                  The clipped image
    """

    try:
        raster_size = img.raster_size
    except:
        # x,y form
        raster_size = img.shape[::-1]
    axr, ax = modf(center_x)
    ayr, ay = modf(center_y)

    if ax + size_x > raster_size[0]:
        size_x = floor(raster_size[0] - center_x)
    if ax - size_x < 0:
        size_x = int(ax)
    if ay + size_y > raster_size[1]:
        size_y =floor(raster_size[1] - center_y)
    if ay - size_y < 0:
        size_y = int(ay)

    # Read from the upper left origin
    pixels = [ax-size_x, ay-size_y, size_x*2, size_y*2]
    pixels = list(map(int, pixels))  #
    if isinstance(img, np.ndarray):
        subarray = img[pixels[1]:pixels[1] + pixels[3] + 1, pixels[0]:pixels[0] + pixels[2] + 1]
    else:
        try:
            subarray = img.read_array(pixels=pixels, dtype=dtype)
        except:
            return None, 0, 0
    return subarray, axr, ayr

def subpixel_phase(sx, sy, dx, dy,
                   s_img, d_img,
                   image_size=(51, 51),
                   **kwargs):
    """
    Apply the spectral domain matcher to a search and template image. To
    shift the images, the x_shift and y_shift, need to be subtracted from
    the center of the search image. It may also be necessary to apply the
    fractional pixel adjustment as well (if for example the center of the
    search is not an integer); this function do not manage shifting.

    Parameters
    ----------
    template : ndarray
               The template used to search

    search : ndarray
             The search image

    Returns
    -------
    x_offset : float
               Shift in the x-dimension

    y_offset : float
               Shift in the y-dimension

    strength : tuple
               With the RMSE error and absolute difference in phase
    """
    image_size = check_image_size(image_size)

    s_roi = roi.Roi(s_img, sx, sy, size_x=image_size[0], size_y=image_size[1])
    d_roi = roi.Roi(d_img, dx, dy, size_x=image_size[0], size_y=image_size[1])

    s_image = s_roi.clip()
    d_template = d_roi.clip()

    if s_image.shape != d_template.shape:

        s_size = s_image.shape
        d_size = d_template.shape
        updated_size_x = int(min(s_size[1], d_size[1]))
        updated_size_y = int(min(s_size[0], d_size[0]))

        # Have to subtract 1 from even entries or else the round up that
        # occurs when the size is split over the midpoint causes the
        # size to be too large by 1.
        if updated_size_x % 2 == 0:
            updated_size_x -= 1
        if updated_size_y % 2 == 0:
            updated_size_y -= 1

        # Since the image is smaller than the requested size, set the size to
        # the current maximum image size and reduce from there on potential
        # future iterations.
        size = check_image_size((updated_size_x, updated_size_y))
        s_roi = roi.Roi(s_img, sx, sy,
                        size_x=size[0], size_y=size[1])
        d_roi = roi.Roi(d_img, dx, dy,
                        size_x=size[0], size_y=size[1])
        s_image = s_roi.clip()
        d_template = d_roi.clip()

        if (s_image is None) or (d_template is None):
            return None, None, None

    (shift_y, shift_x), error, diffphase = register_translation(s_image, d_template, **kwargs)
    dx = d_roi.x - shift_x
    dy = d_roi.y - shift_y

    return dx, dy, (error, diffphase)

def subpixel_transformed_template(sx, sy, dx, dy,
                                  s_img, d_img,
                                  transform,
                                  image_size=(251, 251),
                                  template_size=(51, 51),
                                  template_buffer=5,
                                  func=pattern_match,
                                  verbose=False,
                                  **kwargs):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.

    Parameters
    ----------
    sx : Numeric
         Source X coordinate

    sy : Numeric
         Source y coordinate

    dx : Numeric
         The desintation x coordinate

    dy : Numeric
         The destination y coordinate

    s_img : GeoDataset
            The source image GeoDataset

    d_img : GeoDataset
            The destination image GeoDataset

    transform : object
                A skiage transform object that has scale. The transform object is
                used to project the template into the image.

    image_size : tuple
                 (xsize, ysize) of the image that is searched within (this should be larger
                 than the template size)

    template_size : tuple
                    (xsize, ysize) of the template to iterate over the image in order
                    to identify the area(s) of highest correlation.

    template_buffer : int
                      The inverse buffer applied to the transformed template image. When
                      the warp is applied to project from the template into the image, some
                      amount of no data exists around the edges. This variable is used to clip
                      some number of pixels off the edges of the template. The higher the rotation
                      the higher this value should be.

    func : callable
           The function used to pattern match

    verbose : bool
              If true, generate plots of the matches

    Returns
    -------
    x_shift : float
               Shift in the x-dimension

    y_shift : float
               Shift in the y-dimension

    strength : float
               Strength of the correspondence in the range [-1, 1]

    corrmap : ndarray
              An n,m array of correlation coefficients

    See Also
    --------
    autocnet.matcher.naive_template.pattern_match : for the kwargs that can be passed to the matcher
    autocnet.matcher.naive_template.pattern_match_autoreg : for the jwargs that can be passed to the autoreg style matcher
    """                           
    image_size = check_image_size(image_size)
    template_size = check_image_size(template_size)
    
    template_size_x = template_size[0] * transform.scale[0]
    template_size_y = template_size[1] * transform.scale[1]

    s_roi = roi.Roi(s_img, sx, sy, size_x=image_size[0], size_y=image_size[1])
    d_roi = roi.Roi(d_img, dx, dy, size_x=template_size_x, size_y=template_size_y)

    try:
        s_image_dtype = isis2np_types[pvl.load(s_img.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    except:
        s_image_dtype = None
    
    try:
        d_template_dtype = isis2np_types[pvl.load(d_img.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    except:
        d_template_dtype = None

    s_image = bytescale(s_roi.clip(dtype=s_image_dtype))
    d_template = bytescale(d_roi.clip(dtype=d_template_dtype))

    if verbose:
        fig, axs = plt.subplots(1, 5, figsize=(20,10))
        # Plot of the original image and template
        axs[0].imshow(s_image, cmap='Greys')
        axs[0].set_title('Destination')
        axs[1].imshow(d_template, cmap='Greys')
        axs[1].set_title('Original Source')

    # Build the transformation chance
    shift_x, shift_y = d_roi.center
    
    tf_rotate = tf.AffineTransform(rotation=transform.rotation, shear=transform.shear)
    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])

    # Define the full chain and the inverse
    trans = (tf_shift + (tf_rotate + tf_shift_inv))
    itrans = trans.inverse

    # Now apply the affine transformation
    transformed_roi = tf.warp(d_template,
                                itrans,
                                order=3)

    # Scale the source arr to the destination array size
    scale_y, scale_x = transform.scale
    template_shape_y, template_shape_x = d_template.shape
    scaled_roi = tf.resize(transformed_roi, (int(template_shape_x/scale_x), int(template_shape_y/scale_x)))
    
    # Clip the transformed template to avoid no data around around the edges
    buffered_template = scaled_roi[template_buffer:-template_buffer,template_buffer:-template_buffer]

    # Apply the matcher on the transformed array

    shift_x, shift_y, metrics, corrmap = func(bytescale(buffered_template), s_image, **kwargs)

    if verbose:
        axs[2].imshow(transformed_roi, cmap='Greys')
        axs[2].set_title('Affine Transformed Source')
        axs[3].imshow(buffered_template, cmap='Greys')
        axs[3].set_title('Scaled and Buffered Source')
        axs[4].imshow(corrmap)
        axs[4].set_title('Correlation')
        plt.show()

    # Project the center into the affine space
    projected_center = itrans(d_roi.center)[0]

    # Shifts need to be scaled back into full resolution, affine space
    shift_x *= scale_x
    shift_y *= scale_y

    # Apply the shifts (computed using the warped image) to the affine space center
    new_projected_x = projected_center[0] - shift_x
    new_projected_y = projected_center[1] - shift_y

    # Project the updated location back into image space
    new_unprojected_x, new_unprojected_y = trans([new_projected_x, new_projected_y])[0]

    # Apply the shift
    dx = d_roi.x - (d_roi.center[0] - new_unprojected_x)
    dy = d_roi.y - (d_roi.center[1] - new_unprojected_y)

    return dx, dy, metrics, corrmap

def subpixel_template(sx, sy, dx, dy,
                      s_img, d_img,
                      image_size=(251, 251),
                      template_size=(51,51),
                      func=pattern_match,
                      verbose=False,
                      **kwargs):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.

    Parameters
    ----------
    sx : Numeric
         Source X coordinate

    sy : Numeric
         Source y coordinate

    dx : Numeric
         The desintation x coordinate

    dy : Numeric
         The destination y coordinate

    s_img : GeoDataset
            The source image GeoDataset

    d_img : GeoDataset
            The destination image GeoDataset

    image_size : tuple
                 (xsize, ysize) of the image that is searched within (this should be larger
                 than the template size)

    template_size : tuple
                    (xsize, ysize) of the template to iterate over the image in order
                    to identify the area(s) of highest correlation.
    
    func : callable
           The function used to pattern match

    verbose : bool
              If true, generate plots of the matches

    Returns
    -------
    x_shift : float
               Shift in the x-dimension

    y_shift : float
               Shift in the y-dimension

    strength : float
               Strength of the correspondence in the range [-1, 1]

    corrmap : ndarray
              An n,m array of correlation coefficients

    See Also
    --------
    autocnet.matcher.naive_template.pattern_match : for the kwargs that can be passed to the matcher
    autocnet.matcher.naive_template.pattern_match_autoreg : for the jwargs that can be passed to the autoreg style matcher
    """
    image_size = check_image_size(image_size)
    template_size = check_image_size(template_size)

    template_size_x = template_size[0]
    template_size_y = template_size[1]

    s_roi = roi.Roi(s_img, sx, sy, size_x=image_size[0], size_y=image_size[1])
    d_roi = roi.Roi(d_img, dx, dy, size_x=template_size_x, size_y=template_size_y)

    try:
        s_image_dtype = isis2np_types[pvl.load(s_img.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    except:
        s_image_dtype = None
    
    try:
        d_template_dtype = isis2np_types[pvl.load(d_img.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    except:
        d_template_dtype = None

    s_image = bytescale(s_roi.clip(dtype=s_image_dtype))
    d_template = bytescale(d_roi.clip(dtype=d_template_dtype))
    
    if verbose:
        fig, axs = plt.subplots(1, 3, figsize=(20,10))
        axs[0].imshow(s_image, cmap='Greys')
        axs[1].imshow(d_template, cmap='Greys')
        axs[3].imshow(corrmap)
        plt.show()
    
    if (s_image is None) or (d_template is None):
        return None, None, None, None

    # Apply the matcher function
    shift_x, shift_y, metrics, corrmap = func(d_template, s_image, **kwargs)

    # Apply the shift to the center of the ROI object
    dx = d_roi.x - shift_x
    dy = d_roi.y - shift_y

    return dx, dy, metrics, corrmap

def subpixel_ciratefi(sx, sy, dx, dy, s_img, d_img, search_size=251, template_size=51, **kwargs):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.

    Parameters
    ----------
    sx : numeric
         The x position of the center of the template to be matched to
    sy : numeric
         The y position of the center of the template to be matched to
    dx : numeric
         The x position of the center of the search to be matched from
    dy : numeric
         The y position of the center of the search to be matched to
    s_img : object
            A plio geodata object from which the template is extracted
    d_img : object
            A plio geodata object from which the search is extracted
    search_size : int
                  An odd integer for the size of the search image
    template_size : int
                    A odd integer for the size of the template that is iterated
                    over the search images

    Returns
    -------
    x_offset : float
               Shift in the x-dimension

    y_offset : float
               Shift in the y-dimension

    strength : float
               Strength of the correspondence in the range [-1, 1]
    """
    t_roi = roi.Roi(d_img, dx, dy,
                              size_x=template_size, size_y=template_size)
    s_roi = roi.Roi(s_img, sx, sy,
                                size_x=search_size, size_y=search_size)
    template = t_roi.clip()
    search = s_roi.clip()

    if template is None or search is None:
        return None, None, None

    x_offset, y_offset, strength = ciratefi.ciratefi(template, search, **kwargs)
    dx += (x_offset + t_roi.axr)
    dy += (y_offset + t_roi.ayr)
    return dx, dy, strength

def iterative_phase(sx, sy, dx, dy, s_img, d_img, size=(51, 51), reduction=11, convergence_threshold=1.0, max_dist=50, **kwargs):
    """
    Iteratively apply a subpixel phase matcher to source (s_img) and destination (d_img)
    images. The size parameter is used to set the initial search space. The algorithm
    is recursively applied to reduce the total search space by reduction until the convergence criteria
    are met. Convergence is defined as the point at which the computed shifts (x_shift,y_shift) are
    less than the convergence_threshold. In instances where the size is reducted to 1 pixel the
    algorithm terminates and returns None.

    Parameters
    ----------
    sx : numeric
         The x position of the center of the template to be matched to
    sy : numeric
         The y position of the center of the template to be matched to
    dx : numeric
         The x position of the center of the search to be matched from
    dy : numeric
         The y position of the center of the search to be matched to
    s_img : object
            A plio geodata object from which the template is extracted
    d_img : object
            A plio geodata object from which the search is extracted
    size : tuple
           Size of the template in the form (x,y)
    reduction : int
                With each recursive call to this func, the size is reduced by this amount
    convergence_threshold : float
                            The value under which the result can shift in the x and y directions to force a break

    Returns
    -------
    dx : float
         The new x value for the match in the destination (d) image
    dy : float
         The new y value for the match in the destination (d) image
    metrics : tuple
              A tuple of metrics. In the case of the phase matcher this are difference
              and RMSE in the phase dimension.

    See Also
    --------
    subpixel_phase : the function that applies a single iteration of the phase matcher
    """

    # get initial destination location
    dsample = dx
    dline = dy

    while True:
        shifted_dx, shifted_dy, metrics = subpixel_phase(sx, sy, dx, dy, s_img, d_img, image_size=size, **kwargs)

        # Compute the amount of move the matcher introduced
        delta_dx = abs(shifted_dx - dx)
        delta_dy = abs(shifted_dy - dy)
        dx = shifted_dx
        dy = shifted_dy

        # Break if the solution has converged
        size = (size[0] - reduction, size[1] - reduction)
        dist = np.linalg.norm([dsample-dx, dline-dy])

        if min(size) < 1:
            return None, None, (None, None)
        if delta_dx <= convergence_threshold and\
           delta_dy<= convergence_threshold and\
           abs(dist) <= max_dist:
           break
    return dx, dy, metrics

def estimate_affine_transformation(destination_coordinates, source_coordinates):
    """
    Given a set of destination control points compute the affine transformation
    required to project the source control points into the destination.

    Parameters
    ----------
    destination_coordinates : array-like
                              An n,2 data structure containing the destination control points

    source_coordinates : array-like
                         An n,2 data structure containing the source control points

    Returns
    -------
     : object
       An skimage affine transform object
    """
    destination_coordinates = np.asarray(destination_coordinates)
    source_coordinates = np.asarray(source_coordinates)

    return tf.estimate_transform('affine', destination_coordinates, source_coordinates)

def geom_match(destination_cube,
               source_cube,
               bcenter_x,
               bcenter_y,
               template_kwargs={"image_size":(59,59), "template_size":(31,31)},
               phase_kwargs=None,
               verbose=True, 
               **kwargs):
    """
    Propagates a source measure into destination images and then perfroms subpixel registration.
    Measure creation is done by projecting the (lon, lat) associated with the source measure into the
    destination image. The created measure is then matched to the source measure using a quick projection
    of the destination image into source image space (using an affine transformation) and a naive
    template match with optional phase template match.

    Parameters
    ----------
    destination_cube:  plio.io.io_gdal.GeoDataset
                       The image to be matched to

    source_cube: plio.io.io_gdal.GeoDataset
                 The image that is transformed and matched into the destination_cube

    bcenter_x:  int
                sample location of source measure in base_cube

    bcenter_y:  int
                line location of source measure in base_cube

    size_x:     int
                half-height of the subimage used in the affine transformation

    size_y:     int
                half-width of the subimage used in affine transformation

    template_kwargs: dict
                     contains keywords necessary for autocnet.matcher.subpixel.subpixel_template

    phase_kwargs:    dict
                     contains kwargs for autocnet.matcher.subpixel.subpixel_phase

    verbose:    boolean
                indicates level of print out desired. If True, two subplots are output; the first subplot contains
                the source subimage and projected destination subimage, the second subplot contains the registered
                measure's location in the base subimage and the unprojected destination subimage with the corresponding
                template metric correlation map.

    Returns
    -------
    sample: int
            sample of new measure in destination image space

    line:   int
            line of new measures in destination image space

    dist:   np.float or tuple of np.float
            distance matching algorithm moved measure
            if template matcher only (default): returns dist_template
            if template and phase matcher:      returns (dist_template, dist_phase)

    metric: np.float or tuple of np.float
            matching metric output by the matcher
            if template matcher only (default): returns maxcorr
            if template and phase matcher:      returns (maxcorr, perror, pdiff)

    temp_corrmap: np.ndarray
                  correlation map of the naive template matcher

    See Also
    --------
    autocnet.matcher.subpixel.subpixel_template: for list of kwargs that can be passed to the matcher
    autocnet.matcher.subpixel.subpixel_phase: for list of kwargs that can be passed to the matcher

    """

    if not isinstance(source_cube, GeoDataset):
        raise Exception("source cube must be a geodataset obj")

    if not isinstance(destination_cube, GeoDataset):
        raise Exception("destination cube must be a geodataset obj")

    destination_size_x = template_kwargs['image_size'][0]
    destination_size_y = template_kwargs['image_size'][1]

    destination_startx = int(bcenter_x - destination_size_x)
    destination_starty = int(bcenter_y - destination_size_y)
    destination_stopx = int(bcenter_x + destination_size_x)
    destination_stopy = int(bcenter_y + destination_size_y)

    image_size = source_cube.raster_size
    match_size = destination_cube.raster_size

    # for now, require the entire window resides inside both cubes.
    if destination_stopx > match_size[0]:
        raise Exception(f"Window: {destination_stopx} > {match_size[0]}, center: {bcenter_x},{bcenter_y}")
    if destination_startx < 0:
        raise Exception(f"Window: {destination_startx} < 0, center: {bcenter_x},{bcenter_y}")
    if destination_stopy > match_size[1]:
        raise Exception(f"Window: {destination_stopy} > {match_size[1]}, center: {bcenter_x},{bcenter_y} ")
    if destination_starty < 0:
        raise Exception(f"Window: {destination_starty} < 0, center: {bcenter_x},{bcenter_y}")

    destination_corners = [(destination_startx,destination_starty),
                    (destination_startx,destination_stopy),
                    (destination_stopx,destination_stopy),
                    (destination_stopx,destination_starty)]

    # specifically not putting this in a try/except, this should never fail
    # 07/28 - putting it in a try/except because of how we ground points
    # Transform from the destination center to the source_cube center
    try:
        mlat, mlon = spatial.isis.image_to_ground(destination_cube.file_name, bcenter_x, bcenter_y)
        center_y, center_x = spatial.isis.ground_to_image(source_cube.file_name, mlon, mlat)
    except ProcessError as e:
            if 'Requested position does not project in camera model' in e.stderr:
                print(f'Skip geom_match; Region of interest center located at ({mlon}, {mlat}) does not project to image {source_cube.base_name}')
                print('This should only appear when propagating ground points')
                return None, None, None, None, None

    # Compute the mapping between the destination corners and the source_cube corners in
    # order to estimate an affine transformation
    source_corners = []
    for x,y in destination_corners:
        try:
            lat, lon = spatial.isis.image_to_ground(destination_cube.file_name, x, y)
            source_corners.append(spatial.isis.ground_to_image(source_cube.file_name, lon, lat)[::-1])
        except ProcessError as e:
            if 'Requested position does not project in camera model' in e.stderr:
                print(f'Skip geom_match; Region of interest corner located at ({lon}, {lat}) does not project to image {source_cube.base_name}')
                return None, None, None, None, None


    # Estimate the transformation
    affine = estimate_affine_transformation(destination_corners, source_corners)

    # Apply the subpixel matcher with an affine transformation
    restemplate = subpixel_transformed_template(bcenter_x, bcenter_y, 
                                                center_x, center_y, 
                                                destination_cube, source_cube, 
                                                affine,
                                                verbose=verbose,
                                                **template_kwargs,
                                                **kwargs)

    x, y, metric, temp_corrmap = restemplate
    
    if x is None or y is None:
        return None, None, None, None, None

    dist = np.linalg.norm([center_x-x, center_y-y])
    return x, y, dist, metric, temp_corrmap

def subpixel_register_measure(measureid,
                              iterative_phase_kwargs={},
                              subpixel_template_kwargs={},
                              cost_func=lambda x,y: 1/x**2 * y,
                              threshold=0.005,
                              ncg=None,
                              **kwargs):
    """
    Given a measure, subpixel register to the reference measure of its associated point.

    Parameters
    ----------
    ncg : obj
          the network candidate graph that the point is associated with; used for
          the DB session that is able to access the point.

    measureid : int or obj
              The identifier of the measure in the DB or a Measures object

    iterative_phase_kwargs : dict
                             Any keyword arguments passed to the phase matcher

    subpixel_template_kwargs : dict
                               Any keyword arguments passed to the template matcher

    cost : func
           A generic cost function accepting two arguments (x,y), where x is the
           distance that a point has shifted from the original, sensor identified
           intersection, and y is the correlation coefficient coming out of the
           template matcher.

    threshold : numeric
                measures with a cost <= the threshold are marked as ignore=True in
                the database.
    """

    if isinstance(measureid, Measures):
        measureid = measureid.id

    result = {'measureid':measureid,
              'status':''}

    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    with ncg.session_scope() as session:
        # Setup the measure that is going to be matched
        destination = session.query(Measures).filter(Measures.id == measureid).one()
        destinationid = destination.imageid
        res = session.query(Images).filter(Images.id == destinationid).one()
        destination_node = NetworkNode(node_id=destinationid, image_path=res.path)

        # Get the point id and set up the reference measure
        pointid = destination.pointid
        source = session.query(Measures).filter(Measures.pointid==pointid).order_by(Measures.id).first()
        source.weight = 1

        sourceid = source.imageid
        res = session.query(Images).filter(Images.id == sourceid).one()
        source_node = NetworkNode(node_id=sourceid, image_path=res.path)

        new_template_x, new_template_y, template_metric, _ = subpixel_template(source.sample,
                                                                source.line,
                                                                destination.sample,
                                                                destination.line,
                                                                source_node.geodata,
                                                                destination_node.geodata,
                                                                **subpixel_template_kwargs)
        if new_template_x == None:
            destination.ignore = True # Unable to template match
            result['status'] = 'Unable to template match.'
            return result

        new_phase_x, new_phase_y, phase_metrics = iterative_phase(source.sample,
                                                                    source.line,
                                                                    new_template_x,
                                                                    new_template_y,
                                                                    source_node.geodata,
                                                                    destination_node.geodata,
                                                                    **iterative_phase_kwargs)
        if new_phase_x == None:
            destination.ignore = True # Unable to phase match
            result['status'] = 'Unable to phase match.'
            return result

        dist = np.linalg.norm([new_phase_x-new_template_x, new_phase_y-new_template_y])
        cost = cost_func(dist, template_metric)

        if cost <= threshold:
            destination.ignore = True # Threshold criteria not met
            result['status'] = 'Cost metric not met.'
            return result

        # Update the measure
        if new_phase_x:
            destination.sample = new_phase_x
            destination.line = new_phase_y
            destination.weight = cost

        # In case this is a second run, set the ignore to False if this
        # measures passed. Also, set the source measure back to ignore=False
        destination.ignore = False
        source.ignore = False
        result['status'] = 'Success.'

    return result

def subpixel_register_point(pointid,
                            iterative_phase_kwargs={},
                            subpixel_template_kwargs={},
                            cost_func=lambda x,y: 1/x**2 * y,
                            threshold=0.005,
                            ncg=None,
                            **kwargs):

    """
    Given some point, subpixel register all of the measures in the point to the
    first measure.

    Parameters
    ----------
    pointid : int or obj
              The identifier of the point in the DB or a Points object

    iterative_phase_kwargs : dict
                             Any keyword arguments passed to the phase matcher

    subpixel_template_kwargs : dict
                               Ay keyword arguments passed to the template matcher

    cost : func
           A generic cost function accepting two arguments (x,y), where x is the
           distance that a point has shifted from the original, sensor identified
           intersection, and y is the correlation coefficient coming out of the
           template matcher.

    threshold : numeric
                measures with a cost <= the threshold are marked as ignore=True in
                the database.
    ncg : obj
          the network candidate graph that the point is associated with; used for
          the DB session that is able to access the point.
    """

    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    if isinstance(pointid, Points):
        pointid = pointid.id

    with ncg.session_scope() as session:
        measures = session.query(Measures).filter(Measures.pointid == pointid).order_by(Measures.id).all()
        source = measures[0]

        source.template_metric = 1
        source.template_shift = 0
        source.phase_error = 0
        source.phase_diff = 0
        source.phase_shift = 0

        sourceid = source.imageid
        res = session.query(Images).filter(Images.id == sourceid).one()
        source_node = NetworkNode(node_id=sourceid, image_path=res.path)
        source_node.parent = ncg

        print(f'Attempting to subpixel register {len(measures)-1} measures for point {pointid}')

        resultlog = []
        for measure in measures[1:]:
            currentlog = {'measureid':measure.id,
                        'status':''}
            cost = None
            destinationid = measure.imageid

            res = session.query(Images).filter(Images.id == destinationid).one()
            destination_node = NetworkNode(node_id=destinationid, image_path=res.path)
            destination_node.parent = ncg

            print('geom_match image:', res.path)
            try:
                new_x, new_y, dist, metric,  _ = geom_match(source_node.geodata, destination_node.geodata,
                                                        source.sample, source.line,
                                                        template_kwargs=subpixel_template_kwargs,
                                                        phase_kwargs=iterative_phase_kwargs)
            except Exception as e:
                print(f'geom_match failed on measure {measure.id} with exception -> {e}')
                measure.ignore = True
                currentlog['status'] = f"Failed to register measure {measure.id}"
                resultlog.append(currentlog)
                continue

            if new_x == None or new_y == None:
                measure.ignore = True # Unable to geom match
                currentlog['status'] = 'Failed to geom match.'
                resultlog.append(currentlog)
                continue

            if iterative_phase_kwargs:
                measure.template_metric = metric[0]
                measure.template_shift = dist[0]
                measure.phase_error = metric[1]
                measure.phase_diff = metric[2]
                measure.phase_shift = dist[1]
            else:
                measure.template_metric = metric
                measure.template_shift = dist

            cost = cost_func(measure.template_shift, measure.template_metric)

            if cost <= threshold:
                measure.ignore = True # Threshold criteria not met
                currentlog['status'] = f'Cost failed. Distance shifted: {measure.template_shift}. Metric: {measure.template_metric}.'
                resultlog.append(currentlog)
                continue

            # Update the measure
            measure.sample = new_x
            measure.line = new_y
            measure.weight = cost
            measure.choosername = 'subpixel_register_point'

            # In case this is a second run, set the ignore to False if this
            # measures passed. Also, set the source measure back to ignore=False
            measure.ignore = False
            source.ignore = False
            currentlog['status'] = f'Success.'
            resultlog.append(currentlog)

    return resultlog

def subpixel_register_points(iterative_phase_kwargs={'size': 251},
                             subpixel_template_kwargs={'image_size':(251,251)},
                             cost_kwargs={},
                             threshold=0.005,
                             Session=None):
    """
    Serial subpixel registration of all of the points in a given DB table.

    Parameters
    ----------
    Session : obj
              A SQLAlchemy Session factory.

    pointid : int
              The identifier of the point in the DB

    iterative_phase_kwargs : dict
                             Any keyword arguments passed to the phase matcher

    subpixel_template_kwargs : dict
                               Ay keyword arguments passed to the template matcher

    cost : func
           A generic cost function accepting two arguments (x,y), where x is the
           distance that a point has shifted from the original, sensor identified
           intersection, and y is the correlation coefficient coming out of the
           template matcher.

    threshold : numeric
                measures with a cost <= the threshold are marked as ignore=True in
                the database.
    """
    if not Session:
        raise BrokenPipeError('This func requires a database session.')
    session = Session()
    pointids = [point.id for point in session.query(Points)]
    session.close()
    for pointid in pointids:
        subpixel_register_point(pointid,
                                iterative_phase_kwargs=iterative_phase_kwargs,
                                subpixel_template_kwargs=subpixel_template_kwargs,
                                **cost_kwargs)

