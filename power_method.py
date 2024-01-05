import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    data_shape = data.shape
    eigenvector = np.ones(data_shape[1])

    for i in range(num_steps):
        vector_preceding = eigenvector
        eigenvector = np.dot(data, eigenvector)
        eigenvector /= np.linalg.norm(eigenvector)
        eigenvalue = np.dot(vector_preceding, np.dot(data, eigenvector)) / np.dot(vector_preceding, vector_preceding)

    return float(eigenvalue), eigenvector
