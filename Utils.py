import numpy as np
from PIL import Image
from Kernels import Kernel, gaussian_laplacian
from scipy.signal import fftconvolve

# Alias definitions for focus zone parameters
Coordinate = int
Size = int
ZoneDescription = tuple[Coordinate, Coordinate, Size] | None

def std_sharpness(img_matrix: np.ndarray, zone: tuple[Coordinate, Coordinate, Size] = None, kernel: Kernel = gaussian_laplacian) -> float:
    """
    Returns a sharpness indicator for a given image using the standard deviation

    Parameters
    ----------
    img_matrix : NDArray
        The grayscale image in matrix form
    zone: tuple[Coordinate, Coordinate, Size], optional
        The square window on which to focus in the image, defined by the upper left corner and size of the window, format: (x, y, size), by default None
    kernel : Kernel, optional
        The kernel to be used as filter, by default gaussian_laplacian

    Returns
    -------
    float
        The STD sharpness indicator for the image
    """

    if zone:
        x, y, size = zone
        img_matrix = img_matrix[y : y+size, x : x+size]

    return np.std(fftconvolve(img_matrix, kernel.matrix, mode="valid"))

def std_sharpness_by_path(path: str, zone: tuple[Coordinate, Coordinate, Size] = None, kernel: Kernel = gaussian_laplacian) -> float:
    """
    Returns a sharpness indicator for a given image using the standard deviation

    Parameters
    ----------
    path : str
        The path to the image file
    zone: tuple[Coordinate, Coordinate, Size], optional
        The square window on which to focus in the image, defined by the upper left corner and size of the window, format: (x, y, size), by default None
    kernel : Kernel, optional
        The kernel to be used as filter, by default gaussian_laplacian

    Returns
    -------
    float
        The STD sharpness indicator for the image
    """

    img = Image.open(path).convert("L")
    img_matrix = np.asarray(img)

    return std_sharpness(img_matrix, zone, kernel)

def std_over_set(path_list: list[str], zone: tuple[Coordinate, Coordinate, Size] = None, kernel: Kernel = gaussian_laplacian) -> np.ndarray[float]:
    """
    Iterates over image files and returns their STD sharpness value in order

    Parameters
    ----------
    path_list : list[str]
        The paths to the images studied
    zone_coor : tuple[Coordinate, Coordinate, Size], optional
        The square window on which to focus in the images, defined by the upper left corner and size of the window, format: (x, y, size), by default None
    kernel : Kernel, optional
        The kernel to use for the convolution, by default gaussian_laplacian

    Returns
    -------
    NDArray[float]
        An array containing the STD sharpness value for each image
    """
    std = np.empty(len(path_list))

    for i, path in enumerate(path_list):

        # Opens and converts the images to grayscale matrices
        img = Image.open(path)
        img = img.convert("L")
        img_mat = np.asarray(img)

        std[i] = std_sharpness(img_mat, zone, kernel)

    return std

def display_img_matrix(img_matrix: np.ndarray) -> None:
    """
    Displays a picture from its matrix representation

    Parameters
    ----------
    img_matrix : NDArray
        Matrix representation of an image
    """
    img = Image.fromarray(img_matrix)
    img.show()

def save_img_matrix(img_matrix: np.ndarray, path: str) -> None:
    """
    Saves a matrix image on the drive.

    Parameters
    ----------
    img_matrix : NDArray
        Matrix representation of an image
    path : str
        Path to the location where the image should be saved

    Raises
    ------
    ValueError
        If the output format could not be determined.
    OSError
        If the file could not be written.
    """
    img = Image.fromarray(img_matrix)
    img.save(path)

def load_image_grayscale(path: str) -> np.ndarray:
    """
    Loads an image and returns its matrix form.

    Parameters
    ----------
    path : str
        The path to the image

    Returns
    -------
    NDArray
        The loaded image in grayscale matrix form.
    """
    img = Image.open(path).convert("L")
    return np.asarray(img)

def apply_filter(img_mat: np.ndarray, filter: Kernel, absolute_rect: bool = False) -> np.ndarray:
    """
    Filters the image then applies a rectification to the values.

    Parameters
    ----------
    img_mat : NDArray
        Matrix form of an image
    filter : Kernel
        The kernel to be used as filter
    absolute_rect : bool
        True to use an absolute rectification, False to use a relative rectification

    Returns
    -------
    NDArray
        The image matrix, filter applied and rectified
    """
    res = fftconvolve(img_mat, filter.matrix, mode="valid")

    if absolute_rect:
        res = rectify_absolute(res)
    else:
        res = rectify_relative(res)

    return res

def convert_to_grayscale(rgb_mat: np.ndarray) -> np.ndarray:
    """
    Converts an image to grayscale

    Parameters
    ----------
    rgb_mat : NDArray
        The image in matrix representation in RGB format (3 canals)

    Returns
    -------
    NDArray
        The matrix representation of the image in grayscale
    """

    # Extracts the three canals (red, green, blue) from the image
    R = rgb_mat[..., 0]
    G = rgb_mat[..., 1]
    B = rgb_mat[:, :, 2]

    # Multiplies by luminosity coefficients
    R = R*.299
    G = G*.587
    B = B*.114

    # Adds the canals together and round.
    grayscale_float = R+G+B
    grayscale_int = np.rint(grayscale_float)

    return grayscale_int

def convolve(matrix: np.ndarray, kernel: np.ndarray) -> np.ndarray[int]:
    """
    Computes the 2D convolution between a larger matrix and a smaller kernel with no padding.

    Parameters
    ----------
    matrix : NDArray
        The matrix to be convoluted upon
    kernel : NDArray
        The second matrix, supposed to be smaller than the first one

    Returns
    -------
    NDArray[int]
        The result of the convolution
    """

    height, width = matrix.shape
    h_kernel, w_kernel = kernel.shape

    result = np.zeros((height - h_kernel + 1, width - w_kernel + 1), dtype=np.int32)

    for y in range(height - h_kernel + 1):
        for x in range(width - w_kernel + 1):

            matrix_part = matrix[y:y+h_kernel, x:x+w_kernel]

            result[y, x] = smarter_dot(matrix_part, kernel)
    
    return result


def dot(m1: list[list], m2: list[list]):
    """
    Computes an element-wise multiplication between two matrices with the same dimensions and sums all resulting terms.

    Parameters
    ----------
    m1 : list(2D)
        A matrix
    m2 : list[list]
        A matrix

    Returns
    -------
    number
        The dot product of the two matrices.
    """

    height = len(m1)
    width = len(m1[0])

    ssum = 0

    for y in range(height):
        for x in range(width):
            ssum += m1[y][x] * m1[y][x]

    return ssum

def smarter_dot(m1: np.ndarray, m2: np.ndarray):
    """
    Computes the dot product between two matrices with the same dimensions, using numpy.

    Parameters
    ----------
    m1 : NDArray
        A matrix
    m2 : NDArray
        A matrix

    Returns
    -------
    number
        The dot product of the two matrices
    """
    return np.sum(m1 * m2)

def rectify_relative(matrix: np.ndarray) -> np.ndarray:
    """
    Rectifies the range of the values in the matrix so that they sit in [0, 255] for later visualisation.
    Takes into account the minimum and maximum values encountered in the matrix.
    The minimum value encountered is offset at 0.
    The maximum value encountered is offest at 255.

    Parameters
    ----------
    matrix : NDArray
        A matrix

    Returns
    -------
    NDArray
        The matrix with rectified values
    """
    matrix = matrix.astype('int32')

    maxi = np.max(matrix)
    mini = np.min(matrix)

    # Computes the coeff seperately to avoid overflow
    coeff =  255 / (maxi + abs(mini))

    return np.floor((matrix + abs(mini)) * coeff)

def rectify_absolute(matrix: np.ndarray, kernel: Kernel) -> np.ndarray:
    """
    Rectifies the range of the values in the matrix so that they sit in [0, 255] for later visualisation
    Takes into account the min and max values that can be reached using the specified kernel.
    This function may be used to compare two images visually.

    Parameters
    ----------
    matrix : NDArray
        A matrix obtained after a convolution
    kernel : Kernel
        The kernel used for the convolution

    Returns
    -------
    NDArray
        The matrix with rectified values
    """

    min, max = kernel

    return np.rint((matrix + min)/(max + min)*255)