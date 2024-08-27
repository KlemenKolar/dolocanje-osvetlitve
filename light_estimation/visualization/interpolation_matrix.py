import numpy as np
import matplotlib.pylab as pl

def get_interpolated_matrix(a_bins, b_bins, label): 
    # Define the matrix dimensions and initialize with zeros
    matrix = np.zeros((a_bins, b_bins))

    # Define the integer value and its position
    value = 1
    pos = np.array([int(np.floor(label/b_bins)), int(label % b_bins)])  # Specify the position of the value

    # Create coordinate grids for the matrix
    x = np.arange(b_bins)
    y = np.arange(a_bins)
    xx, yy = np.meshgrid(x, y)

    # Calculate the distance of each point from the specified point
    distances = np.sqrt((abs(xx - pos[1])+0.3)**2 + (abs(yy - pos[0]) + 0.3)**2)

    # To avoid division by zero, we set the distance at the position to a very small number
    distances[pos[0], pos[1]] = 1e-10

    # Inverse distance weighting: value decreases with distance
    matrix = value / distances

    # Set the exact position to the initial value
    matrix[pos[0], pos[1]] = value

    print("Interpolated Matrix:")
    print(matrix)
    #softmax_matrix = special.softmax(matrix)
    #print(softmax_matrix)

    return np.round(np.array(matrix), decimals=2)


def draw_matrix(matrix):
    pl.figure()
    tb = pl.table(cellText=matrix, loc=(0,0), cellLoc='center')

    ax = pl.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    pl.show()


if __name__ == "__main__":
    matrix = get_interpolated_matrix(5, 5, 12)
    draw_matrix(matrix)