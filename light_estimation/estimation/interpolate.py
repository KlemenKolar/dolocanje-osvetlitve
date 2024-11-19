import numpy as np
from scipy import special
import torch
from scipy.ndimage import gaussian_filter


def get_interpolated_vector(N, indices, sigma=1.0):
    """
    Generates a list of vectors with Gaussian distribution values centered at specified indices.
    
    Parameters:
    - N (int): Length of each vector.
    - indices (list of int): List of indices where the Gaussian mean is centered.
    - sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
    - list of np.ndarray: List containing the Gaussian distribution vectors.
    """
    def gaussian(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Generate the range vector
    x = np.arange(N)
    
    # List to store the resulting vectors
    gaussian_vectors = []
    
    for index in indices:
        # Calculate Gaussian values for each point in the vector
        g = gaussian(x, index, sigma)
        
        # Append the normalized vector to the result list
        gaussian_vectors.append(torch.from_numpy(g))
    
    return torch.softmax(torch.tensor(torch.stack(gaussian_vectors, dim=0), requires_grad=True), dim=1)


def get_interpolated_matrix(a_bins, b_bins, labels, zero=False): 
    # Define the matrix dimensions and initialize with zeros
    softmax_matrices = []
    for label in labels:
        matrix = np.zeros((a_bins, b_bins))

        # Define the integer value and its position
        value = 10
        pos = np.array([int(np.floor(label/b_bins)), int(label % b_bins)])  # Specify the position of the value

        # Create coordinate grids for the matrix
        x = np.arange(b_bins)
        y = np.arange(a_bins)
        xx, yy = np.meshgrid(x, y)

        if not zero:
            # Calculate the distance of each point from the specified point
            distances = np.sqrt((abs(xx - pos[1])+0.3)**2 + (abs(yy - pos[0]) + 0.3)**2)

            # To avoid division by zero, we set the distance at the position to a very small number
            distances[pos[0], pos[1]] = 1e-10

            # Inverse distance weighting: value decreases with distance
            matrix = value / distances

        # Set the exact position to the initial value
        matrix[pos[0], pos[1]] = value

        #print("Interpolated Matrix:")
        #print(matrix)
        #softmax_matrix = special.softmax(matrix)
        #print(softmax_matrix)
        softmax_matrix = torch.from_numpy(matrix.flatten())
        softmax_matrices.append(softmax_matrix)

    return torch.softmax(torch.tensor(torch.stack(softmax_matrices, dim=0), requires_grad=True), dim=1)


def get_interpolated_gauss_matrix(a_bins, b_bins, labels, sigma=1, bins=False):
    matrices = []
    for label in labels:
        matrix = np.zeros((a_bins, b_bins))

        # Define the position where you want to center the Gaussian filter
        if not bins:
            pos = np.array([int(np.floor(label/b_bins)), int(label % b_bins)])
        else:
            pos = label
        value = 10  # Peak value at the center

        # Create a matrix with a single peak at the center
        matrix[pos[0], pos[1]] = value

        # Apply Gaussian filter to the matrix
        gaussian_matrix = gaussian_filter(matrix, sigma=sigma)

        normalized_matrix = (gaussian_matrix - gaussian_matrix.min()) / (gaussian_matrix.max() - gaussian_matrix.min())
        matrices.append(torch.from_numpy(np.array(normalized_matrix).flatten()))

    return torch.stack(matrices, dim=0).clone().detach().requires_grad_(True)


if __name__ == "__main__":
    #matrix = get_interpolated_matrix(32, 16, np.array([15]), True)
    #vec = get_interpolated_vector(32, np.array([8,2,22,9]), sigma=1.0)
    #print(vec)
    matrix = get_interpolated_gauss_matrix(32, 16, [5], sigma=1)
    pass
