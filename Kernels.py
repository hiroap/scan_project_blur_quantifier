from numpy import array
class Kernel:
    """
    Represents a matrix kernel along with useful properties (height, width, max, min)
    """
    def __init__(self, matrix: list[list]) -> None:
        self.matrix = array(matrix)

        self.size = self.matrix.shape

        self.height = self.size[0]
        self.width = self.size[1]

        maxi, mini = 0, 0

        for y in range (self.height):
            for x in range (self.width):
                pixel_value = self.matrix[y][x]

                if pixel_value > 0 : 
                    maxi += pixel_value * 255
                
                else:
                    mini += pixel_value * 255
        
        self.max = maxi
        self.min = mini
    
    def __str__(self) -> str:
        return "Kernel({0}, {1})\n{2}".format(self.height, self.width, self.matrix)
    
    __repr__ = __str__

sobel_horizontal = Kernel([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

sobel_vertical = Kernel([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

laplacian = Kernel([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

diagonal_laplacian = Kernel([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
])

imp_laplacian = Kernel([
    [1, 4, 1],
    [4, -20, 4],
    [1, 4, 1]
])

larger_laplacian = Kernel([
    [0, 0, -1, 0, 0],
    [0, -1, -2, -1, 0],
    [-1, -2, 16, -2, -1],
    [0, -1, -2, -1, 0],
    [0, 0, -1, 0, 0]
])

gaussian_laplacian = Kernel([
    [0, 1, 1, 2, 2, 2, 1, 1, 0],
    [1, 2, 4, 5, 5, 5, 4, 2, 1],
    [1, 4, 5, 3, 0, 3, 5, 4, 1],
    [2, 5, 3, -12, -24, -12, 3, 5, 2],
    [2, 5, 0, -24, -40, -24, 0, 5, 2],
    [2, 5, 3, -12, -24, -12, 3, 5, 2],
    [1, 4, 5, 3, 0, 3, 5, 4, 1],
    [1, 2, 4, 5, 5, 5, 4, 2, 1],
    [0, 1, 1, 2, 2, 2, 1, 1, 0]
])

gaussian_7x7 = Kernel([
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0],
])