import numpy as np
import matplotlib.pyplot as plt

def create_points_sphere(radius: float, n: int):
    """
        Generate n random points on the surface of a semisphere with a given radius.

        The semi-sphere is defined by:
            φ ∈ [0°, 80°]
            θ ∈ [0, 2π)

        Args:
            radius (float): Radius of the sphere from which points are sampled.
            n (int) : Number of points to generate.

        Returns:
            x, y, z (list of float) : Cartesian coordinates of sampled points.
        """

    # Convert phi limits from degrees to radians
    phi_min = np.deg2rad(0)
    phi_max = np.deg2rad(80)

    # Sample θ and φ uniformly over their set ranges
    u = np.random.rand(n)
    v = np.random.uniform(np.cos(phi_max), np.cos(phi_min), n)

    # Convert sampled values to spherical coordinates
    theta = 2 * np.pi * u
    phi = np.arccos(v)

    # Convert spherical coordinates into Cartesian
    x = []
    y = []
    z = []
    for i in range(n):
        x.append(radius * np.sin(phi[i]) * np.cos(theta[i]))
        y.append(radius * np.sin(phi[i]) * np.sin(theta[i]))
        z.append(radius * np.cos(phi[i]))

    return x,y,z

# Generate grasp poses at different radii
x_1,y_1,z_1 = create_points_sphere(0.21, 300)
x_2,y_2,z_2 = create_points_sphere(0.25, 250)
x_3,y_3,z_3 = create_points_sphere(0.3, 450)

# Plot the results in 3D
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plots for each radius
ax.scatter(x_1, y_1, z_1, s=5, c='blue', alpha=0.6, label='0.21')
ax.scatter(x_2, y_2, z_2, s=5, c='red', alpha=0.6, label='0.25')
ax.scatter(x_3, y_3, z_3, s=5, c='green', alpha=0.6, label='0.3')

# Axis labels, title + legend - display the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Sampled Grasp Candidate Positions Around the Object')
ax.legend(title = 'Sampling radius')
plt.show()
