from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from cs182_project_pointnet.visualize.rotation import rot_degrees


def show_points(point_cloud, pred_color, title):
    # uncomment this line if you have performance issues due to several plots
    # see below for what several plots mean
    # plt.close("all")
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])

    @widgets.interact(rotation=(-90, 90, 5), elevation=(-90, 90, 5))
    def update(rotation=0, elevation=0):
        r = np.random.choice(point_cloud.shape[0], point_cloud.shape[0], replace=False)
        ptcloud = point_cloud[r, :]
        pred_color_sampled = pred_color[r, :]
        show = (rot_degrees(rotation, elevation) @ ptcloud.T).T
        ax.clear()
        ax.scatter(show[:, 0], show[:, 1], show[:, 2], c=pred_color_sampled)
        ax.view_init(0, 0)
        ax.grid(False)
        ax.axis("off")
        ax.set_title(title)
        plt.show();