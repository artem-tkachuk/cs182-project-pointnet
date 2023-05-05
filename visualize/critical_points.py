import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from cs182_project_pointnet.visualize.show_points import show_points


def compute_principal_curvature(points, k=10):
    """
    Compute the principal curvature for each point in the point cloud.
    Args:
        points: A numpy array of shape (num_points, 3) representing the 3D point cloud.
        k: Number of nearest neighbors to consider for curvature estimation.
    Returns:
        principal_curvature: A numpy array of shape (num_points,) containing the principal curvature values.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)

    principal_curvature = []

    for index in indices:
        neighborhood = points[index]
        cov_matrix = np.cov(neighborhood.T)
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        principal_curvature.append(min(eigenvalues) / sum(eigenvalues))

    return np.array(principal_curvature)


def visualize_critical_points(points, pred_labels, curvatures, curvature_threshold=0.1):
    """
    Visualize the critical points in the point cloud.
    Args:
        points: A numpy array of shape (num_points, 3) representing the 3D point cloud.
        pred_labels: A numpy array of shape (num_points,) representing the predicted labels of each point.
        curvatures: A numpy array of shape (num_points,) containing the principal curvature values.
        curvature_threshold: A threshold for determining critical points based on the principal curvature.
    """
    cmap = plt.cm.get_cmap("hsv", 10)
    cmap = np.array([cmap(i) for i in range(10)])[:, :3]
    pred_colors = cmap[pred_labels, :]

    critical_points = points[curvatures > curvature_threshold]
    critical_point_colors = pred_colors[curvatures > curvature_threshold]

    show_points(critical_points, critical_point_colors, title='Critical Points')
