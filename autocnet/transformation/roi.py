from math import modf, floor
import numpy as np


class Roi():
    """
    Region of interest (ROI) object that is a sub-image taken from
    a larger image or array. This object supports transformations
    between the image coordinate space and the ROI coordinate
    space.

    Attributes
    ----------

    x : float
        The x coordinate in image space

    y : float
        The y coordinate in image space

    size_x : int
             1/2 the total ROI width in pixels

    size_y : int
             1/2 the total ROI height in pixels

    origin_roi : tuple
                 in the form (x,y). The origin coordinates
                 of the ROI (center of the ROI)

    left_x : int
             The left pixel coordinate in image space

    right_x : int
              The right pixel coordinage in image space

    top_y : int
            The top image coordinate in image space

    bottom_y : int
               The bottom image coordinate in imge space
    """
    def __init__(self, geodataset, x, y, size_x=200, size_y=200):
        self.geodataset = geodataset

        self.x = x
        self.y = y
        self.size_x = size_x
        self.size_y = size_y
        
    @property
    def x(self):
        return self._x + self.axr

    @x.setter
    def x(self, x):
        self.axr, self._x = modf(x)
    
    @property
    def y(self):
        return self._y + self.ayr

    @y.setter
    def y(self, y):
        self.ayr, self._y = modf(y)

    @property
    def left_x(self):
        return self._left_x

    @left_x.setter
    def left_x(self, x):
        self._left_x = int(x)

    @property
    def right_x(self):
        return self._right_x

    @right_x.setter
    def right_x(self, x):
        self._right_x = int(x)

    @property
    def top_y(self):
        return self._top_y
    
    @top_y.setter
    def top_y(self, y):
        self._top_y = int(y)

    @property
    def bottom_y(self):
        return self._bottom_y

    @bottom_y.setter
    def bottom_y(self, y):
        self._bottom_y = int(y)

    @property
    def origin_roi(self):
        _ = self.image_extent
        return ((self.right_x - self.left_x) / 2,
                (self.bottom_y - self.top_y) / 2)

    @property
    def image_extent(self):
        """
        In full image space, this method computes the valid
        pixel indices that can be extracted.
        """
        try:
            # Geodataset object
            raster_size = self.geodataset.raster_size
        except:
            # Numpy array in y,x form
            raster_size = self.geodataset.shape[::-1]

        # what is the extent that can actually be extracted?
        self.left_x = self._x - self.size_x
        self.right_x = self._x + self.size_x
        self.top_y = self._y - self.size_y
        self.bottom_y = self.y + self.size_y

        if self._x - self.size_x < 0:
            self.left_x = 0
        if self._y - self.size_y < 0:
            self.top_y = 0
        if self._x + self.size_x > raster_size[0]:
            self.right_x = raster_size[0]
        if self._y + self.size_y > raster_size[1]:
            self.bottom_y = raster_size[1]

        return [self.left_x, self.right_x,
                self.top_y, self.bottom_y]

    def clip(self):
        pixels = self.image_extent
        if isinstance(self.geodataset, np.ndarray):
            array = self.geodataset[pixels[2]:pixels[3]+1, 
                                         pixels[0]:pixels[1]+1]
        else:
            # Have to reformat to [xstart, ystart, xnumberpixels, ynumberpixels]
            pixels = [pixels[0], pixels[2], pixels[1]-pixels[0], pixels[3]-pixels[2]]
            array = self.geodataset.read_array(pixels=pixels)

        return array

    def transform(self, x, y):
        """
        Convert arbitrary coordinates from the ROI coordinate system
        to the full image coordinate system.
        """
        pass
