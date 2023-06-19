"""Script for creating Rx, Ry plots for a single manually specified scenario"""
import numpy as np

from gmhazard_2 import plotting
from gmhazard_2 import utils

import matplotlib.pyplot as plt


# Create the rupture coordinates
segment_nztm_coords_1 = np.asarray(
    [
        [1571512.2113763038, 5179366.911408591, 0.0],
        [1571520.2113763038, 5179366.911408591, 30],
        [1571512.2113763038, 5179366.911408591 + 1e4, 0.0],
        [1571520.2113763038, 5179366.911408591 + 1e4, 30],
    ]
)

segment_nztm_coords_2 = np.asarray(
    [
        [1571512.2113763038 + 5e3, 5179366.911408591 + 1e4, 0.0],
        [1571520.2113763038 + 3.1e3, 5179366.911408591 + 1e4, 30],
        [1572512.2113763038 + 5e3, 5179366.911408591 + 3e3, 0.0],
        [1572520.2113763038 + 3.1e3, 5179366.911408591 + 3e3, 30],
    ]
)
segment_nztm_coords = np.stack(
    (segment_nztm_coords_1, segment_nztm_coords_2), axis=-1
)

### Segments for Figure 8 of Spudich paper
# segment_nztm_coords_1 = np.asarray(
#     [[0.0, 0.0, 0.0], [1.0, 0.5, 0.1], [-0.5, 3.0, 0.0], [0.5, 3.5, 0.1]]
# )
#
# segment_nztm_coords_2 = np.asarray(
#     [[-0.5, 2.0, 0.0], [-1.0, 0.0, 0.15], [5.0, 0.0, 0.0], [4.5, -2.0, 0.15]]
# )
#
# segment_nztm_coords_3 = np.asarray(
#     [[-0.5, 2, 0.0], [0.0, 2.5, 0.2], [-4.5, 3.5, 0.0], [-4.0, 4.0, 0.2]]
# )
#
# segment_nztm_coords = np.stack(
#     (segment_nztm_coords_1, segment_nztm_coords_2, segment_nztm_coords_3), axis=-1
# )

# Create the mesh
x_min, x_max = (
    segment_nztm_coords[:, 0, :].ravel().min() - 1e4,
    segment_nztm_coords[:, 0, :].ravel().max() + 1e4,
)
y_min, y_max = (
    segment_nztm_coords[:, 1, :].ravel().min() - 1e4,
    segment_nztm_coords[:, 1, :].ravel().max() + 1e4,
)

p_x, p_y = np.meshgrid(np.linspace(x_min, x_max, 99), np.linspace(y_min, y_max, 99))


# Result arrays
scenario_Rx, scenario_Ry, scenario_Rrup, scenario_Rjb = utils.compute_mesh_distances(
    segment_nztm_coords, p_x, p_y, np.asarray([0, 1]), np.asarray([0, 1])
)

# Plot the results
fig = plt.figure(figsize=(8, 8))
plotting.create_distance_plot(
    segment_nztm_coords,
    p_x,
    p_y,
    scenario_Rx,
    -30,
    30,
    "coolwarm",
    plot_downdip_points=True,
    title="Rx",
)
plt.gca().set_aspect("equal")
plt.show()
plt.close()

fig = plt.figure(figsize=(8, 8))
plotting.create_distance_plot(
    segment_nztm_coords,
    p_x,
    p_y,
    scenario_Ry,
    -30,
    30,
    "coolwarm",
    plot_downdip_points=True,
    title="Ry",
)
plt.gca().set_aspect("equal")
plt.show()
plt.close()
