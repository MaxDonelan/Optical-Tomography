import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from matplotlib.tri import Triangulation



class Mesh:
    def __init__(self, coords, triangles, boundary):
        self.coords = coords
        self.triangles = triangles
        self.boundary = boundary
        self.n_nodes = len(coords)
        self.n_triangles = len(triangles)


class CircularMesh(Mesh):
    def __init__(self, radius, center, n_rings):
        coords = np.array([0,0])
        ring_spacing = radius / n_rings
        for ring in range(1, n_rings):
            angles = 2*np.pi * np.arange(ring*6) / (6*ring)
            ring_radius = ring_spacing*ring
            node_coords = ring_radius * np.column_stack([np.cos(angles), np.sin(angles)])
            coords = np.vstack([coords, node_coords])

        boundary_coords = radius * np.column_stack([np.cos(angles), np.sin(angles)])
        # we don't need the inner boundary right now but here's how to find it
        # inner_boundary = np.zeros(len(coords), dtype=bool)
        # inner_boundary[-(6*n_rings):] = True

        # final high resolution band
        hres_angle_shift = np.pi / (6*(n_rings-1))
        hres_angles = angles + hres_angle_shift
        hres_radius = (radius - 0.5*ring_spacing) / np.cos(hres_angle_shift)
        hres_coords = hres_radius * np.column_stack([np.cos(hres_angles), np.sin(hres_angles)])

        coords = np.vstack([coords, hres_coords, boundary_coords]) + np.asarray(center)

        triangles = Delaunay(coords).simplices

        boundary = np.zeros(len(coords), dtype=bool)
        boundary[-len(boundary_coords):] = True

        super().__init__(coords, triangles, boundary)
        self.radius = radius
        self.center = center
        self.n_rings = n_rings


def plot_mesh(mesh: Mesh, savefile):
    fig, ax = plt.subplots()
    ax.triplot(mesh.coords[:, 0], mesh.coords[:, 1], mesh.triangles, lw = 0.5, color = "grey")
    ax.plot(mesh.coords[mesh.boundary, 0], mesh.coords[mesh.boundary, 1], "o", ms = 0.75,  color="red")
    ax.plot(mesh.coords[~mesh.boundary, 0], mesh.coords[~mesh.boundary, 1], "o", ms = 0.75, color="blue")
    ax.set_aspect("equal")
    fig.savefig(savefile, dpi = 250)


def plot_function_on_mesh(mesh: Mesh, z, savefile, log_scale=False, title=None):
    fig, ax = plt.subplots()
    if log_scale:
        z = np.log(z + 1e-16)
    tri = Triangulation(x=mesh.coords[:, 0], y=mesh.coords[:, 1], triangles=mesh.triangles)
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    fig.colorbar(ax.tricontourf(tri, z))
    fig.savefig(savefile, dpi=250)




if __name__ == "__main__":
    mesh = CircularMesh(25, (0,0), 25)

    print(mesh.triangles)
    print(mesh.coords)
    print(mesh.boundary)

    print(f"Mesh Facts\nNumber of nodes: {mesh.n_nodes}\nNumber of triangles: {mesh.n_triangles}")

    plot_mesh(mesh, "images/test_mesh.png")


    