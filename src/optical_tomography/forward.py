import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy.sparse as sp

from scipy.linalg import inv, solve

import optical_tomography.matrix_assembly as ma
from optical_tomography.model_region import get_absorption, get_diffusion
from optical_tomography.mesh_creation import Mesh, CircularMesh, plot_function_on_mesh



def optical_tomography_forward(mesh: Mesh, absorption, diffusion, sources: np.ndarray | list[np.ndarray]):
    """
    Computes $\\Phi$ on the mesh given the absorption and diffusion function, 
    as well as the sources (as a tuple of coordinate arrays and values). The 
    sources may be given as a list, in which case each element is taken 
    individually and the resulting list of $\\Phi$'s is returned. This is so 
    that the inversion of $C + K$ only needs to be performed once.
    """
    # should be able to take in multiple sets of sources at a time to avoid recomputing the inverse
    def absorption_func(u1, u2, triangle, i, j, boundary):
        return ma.send_func_to_cartesian_coords(absorption, triangle, u1, u2) * ma.phi(u1, u2, i, boundary[i]) * ma.phi(u1, u2, j, boundary[j])

    def diffusion_func(u1, u2, triangle, i, j, boundary):
        return ma.send_func_to_cartesian_coords(diffusion, triangle, u1, u2) * la.multi_dot([ma.gradient_phi(u1, u2, i, boundary[i]) @ ma.inverse_J_gK(triangle), ma.gradient_phi(u1, u2, j, boundary[j]) @ ma.inverse_J_gK(triangle)])

    C = ma.calculate_global_matrix(mesh=mesh, func=absorption_func)
    K = ma.calculate_global_matrix(mesh=mesh, func=diffusion_func)

    forward = C + K
    forward_invs = inv(forward[np.ix_(~mesh.boundary, ~mesh.boundary)])

    single_source_set = not isinstance(sources, list)
    if single_source_set:
        sources = [sources]

    Phi = []
    for instance in sources:
        q0 = ma.calculate_source_vector(mesh=mesh, sources=instance)

        Phi_part = forward_invs @ q0[~mesh.boundary]
        Phi_instance = np.zeros(shape=q0.shape, dtype=float)
        Phi_instance[~mesh.boundary, ] += Phi_part
        Phi.append(Phi_instance)

    if single_source_set:
        Phi = Phi[0]

    return Phi


def generate_equidistant_sources(radius, n, values, offset = 0):
    source_angles = 2*np.pi * np.linspace(0, 1, n, endpoint=False) + offset
    source_coords = radius * np.column_stack([np.cos(source_angles), np.sin(source_angles)])
    source_values = values
    return source_coords, source_values




if __name__ == "__main__":
    mesh = CircularMesh(25, (0,0), 25)
    absorption_func = lambda u1, u2, triangle, i, j, boundary: ma.send_func_to_cartesian_coords(get_absorption, triangle, u1, u2) * ma.phi(u1, u2, i, boundary[i]) * ma.phi(u1, u2, j, boundary[j])
    C = ma.calculate_global_matrix(mesh=mesh,
                                func=absorption_func)

    diffusion_func = lambda u1, u2, triangle, i, j, boundary: ma.send_func_to_cartesian_coords(get_diffusion, triangle, u1, u2) * la.multi_dot([ma.gradient_phi(u1, u2, i, boundary[i]) @ ma.inverse_J_gK(triangle), ma.gradient_phi(u1, u2, j, boundary[j]) @ ma.inverse_J_gK(triangle)])
    K = ma.calculate_global_matrix(mesh=mesh,
                                func=diffusion_func)

    source_angles = 2*np.pi * np.linspace(0, 1, 6*(mesh.n_rings-1), endpoint=False) + np.pi / (6*(mesh.n_rings-1))
    source_radius = mesh.radius - 0.75*(mesh.radius / mesh.n_rings)
    source_coords = source_radius * np.column_stack([np.cos(source_angles), np.sin(source_angles)])
    source_values = np.repeat(0.0, len(source_angles))
    source_values[0] = 2.5e8
    #source_values = np.repeat(1.0*2.5e8, len(source_angles))
    sources = source_coords, source_values

    q0 = ma.calculate_source_vector(mesh, sources)

    forward = C + K
    Phi = solve(forward[np.ix_(~mesh.boundary, ~mesh.boundary)], q0[~mesh.boundary])
    Phi_full = np.zeros(shape=q0.shape, dtype=float)
    Phi_full[~mesh.boundary, ] += Phi


    plot_function_on_mesh(mesh, Phi_full, "images/zero_boundary_test_phi.png", log_scale=True)