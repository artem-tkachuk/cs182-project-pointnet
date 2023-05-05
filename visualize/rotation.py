import numpy as np

# Helper Functions

rotY = lambda x: np.array([
    [np.cos(x), 0, np.sin(x)],
    [0, 1, 0],
    [-np.sin(x), 0, np.cos(x)]
])

rotZ = lambda x: np.array([
    [np.cos(x), -np.sin(x), 0],
    [np.sin(x), np.cos(x), 0],
    [0, 0, 1]
])

rot_view = lambda z, y: rotY(y) @ rotZ(z)

rot_degrees = lambda z, y: rot_view(np.deg2rad(z), np.deg2rad(y))
