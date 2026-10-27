import time
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.sparse import coo_matrix
from functools import partial

from optical_tomography.model_region import get_absorption, get_diffusion
from optical_tomography.mesh_creation import Mesh, CircularMesh



def phi(x, y, i, boundary):
    """the basis functions in barycentric coordinates"""
    if boundary:
        return 0
    else:
        if i == 0:
            return 1 - x - y
        elif i == 1:
            return x
        elif i == 2:
            return y
        else:
            return 0


def gradient_phi(x, y, i, boundary):
    """
    The Gradient of $\\varphi_i$ with respect to barycentric coordinates
    """
    if boundary:
        return np.asarray([0, 0])
    else:
        if i == 0:
            return np.asarray([-1, -1])
        elif i == 1:
            return np.asarray([1, 0])
        elif i == 2:
            return np.asarray([0, 1])
        else:
            return np.asarray([0, 0]) 


def J_gK(triangle):
    (N11, N12), (N21, N22), (N31, N32) = triangle
    return np.asarray([[N21 - N11, N31 - N11],
                       [N22 - N12, N32 - N12]])


def det_J_gK(triangle):
    return la.det(J_gK(triangle))


def inverse_J_gK(triangle):
    """Computes $(J_{g_K})^{-1}$ which is the inverse of $J_{g_K}$ (i.e. $(J_{g_K})^{-1} J_{g_K} = I_2$)"""
    (N11, N12), (N21, N22), (N31, N32) = triangle
    invs = np.asarray([[N32 - N12, N11 - N31],
                      [N12 - N22, N21 - N11]])
    return (1 / det_J_gK(triangle)) * invs


def send_func_to_cartesian_coords(func, triangle, x, y):
    cartesian_x = triangle[0, 0] + x*(triangle[1, 0] - triangle[0, 0]) + y*(triangle[2, 0] - triangle[0, 0])
    cartesian_y = triangle[0, 1] + x*(triangle[1, 1] - triangle[0, 1]) + y*(triangle[2, 1] - triangle[0, 1])
    return func(cartesian_x, cartesian_y)


def cartesian_to_barycentric(point, triangle):
    x = point - triangle[0] 
    return inverse_J_gK(triangle) @ x
    

def trapizoidal_integration_0to1(func, breaks = 25):
    resolution = 1 / (breaks-1)
    x_axis = np.linspace(0, 1, num=breaks)
    discretized_func = np.zeros((breaks, breaks))

    for i, x in enumerate(x_axis):
        for j, y in enumerate(np.linspace(0, 1-x, num = breaks-i)):
            discretized_func[i, j] = func(x, y)

    integral = 0
    for i in range(discretized_func.shape[0]-1):
        for j in range(breaks-i-1):
            integral += (resolution**2) * np.mean([discretized_func[i, j], discretized_func[i+1, j], discretized_func[i, j+1], discretized_func[i+1, j+1]])

    return integral


def one_point_quadrature_integration(func):
    return func(1/3, 1/3) * 0.5


def three_point_quadrature_integration(func):
    return (func(0.5, 0) + func(0, 0.5) + func(0.5, 0.5))*(1/6)


def is_in_triangle(point, triangle):
    bc_point = cartesian_to_barycentric(point, triangle)
    if bc_point.sum() <= 1 and np.all(bc_point >= 0):
        return True
    else:
        return False


def calculate_source_vector(mesh: Mesh, sources):
    source_coords, source_values = sources
    source_vector = np.zeros(mesh.n_nodes)
    for i, source in enumerate(source_coords):
        for j in range(mesh.n_triangles):
            triangle_indices = mesh.triangles[j, ]
            triangle = mesh.coords[triangle_indices]
            boundary = mesh.boundary[triangle_indices]
            if is_in_triangle(source, triangle):
                u1, u2 = cartesian_to_barycentric(source, triangle)
                for k in range(3):
                    source_vector[triangle_indices[k]] += source_values[i] * phi(u1, u2, k, boundary[k])
                break

    return source_vector
                

def calculate_element_matrix(triangle, func, boundary, integration = three_point_quadrature_integration):
    element_matrix = np.empty((3, 3))
    for i in range(3):
        for j in range(i+1):
            element_matrix[i, j] = element_matrix[j, i] = integration(partial(func, triangle=triangle, i=i, j=j, boundary=boundary))

    element_matrix *= abs(det_J_gK(triangle))
    return element_matrix


def calculate_global_matrix(mesh: Mesh, func, integration = three_point_quadrature_integration) -> np.ndarray:
    """
    Computes the global stiffness matrix of the given function on the given mesh 
    using the given integration method on each triangle. Note that each triangle 
    is a 3x2 array representing the three pairs of coordinates corresponding to 
    the vertices of the triangle.
    """
    global_matrix = np.zeros((mesh.n_nodes, mesh.n_nodes))
    for i in range(mesh.n_triangles):
        triangle_indices = mesh.triangles[i, ]
        triangle = mesh.coords[triangle_indices]
        boundary = mesh.boundary[triangle_indices]
        element_matrix = calculate_element_matrix(triangle, func, boundary, integration)
        for a in range(3):
            for b in range(3):
                global_matrix[triangle_indices[a], triangle_indices[b]] += element_matrix[a, b]

    return global_matrix

def plot_global_matrix(mat, savefile):
    mat = coo_matrix(mat)
    fig, ax = plt.subplots()
    ax.spy(mat, ms=1, color="black")
    plt.savefig(savefile, dpi=250)


# Might be useful later
# def calculate_element_vector(triangle, func, integration = quadrature_integration):
#     element_vector = np.empty(3)
#     for i in range(3):
#         element_vector[i] = integration(partial(func, triangle=triangle, i=i))

#     element_vector *= abs(det_J_gK(triangle))
#     return element_vector

# def calculate_global_vector(mesh: Mesh, func, integration = quadrature_integration):
#     global_vector = np.zeros(mesh.n_nodes)
#     for i in range(mesh.n_triangles):
#             triangle_indices = mesh.triangles[i, ]
#             triangle = mesh.coords[triangle_indices]
#             element_vector = calculate_element_vector(triangle, func, integration)
#             for a in range(3):
#                 global_vector[triangle_indices[a]] += element_vector[a]
    
#     return global_vector





if __name__ == "__main__":
    # working towards calculating C, K, and q0
    mesh = CircularMesh(25, (0,0), 25)
    integration_methods = {
        #"Trapizoidal | Breaks = 25": trapizoidal_integration_0to1, # is slow
        "Quadrature | 1 point": one_point_quadrature_integration,
        "Quadrature | 3 points": three_point_quadrature_integration,
    }
    for name, integration in integration_methods.items():
        start = time.perf_counter()
        test_matrix = calculate_global_matrix(mesh=mesh,
                                              func=lambda x, y, triangle, i, j, boundary: phi(x, y, i, boundary[i]) * phi(x, y, j, boundary[j]),
                                              integration=integration)
        elapsed = time.perf_counter() - start
        print(f"[{name}] Test function area on mesh: {test_matrix.sum():.2f} (true value should be approx. 1963.50), took {elapsed:.4f}s")

    absorption_func = lambda u1, u2, triangle, i, j, boundary: send_func_to_cartesian_coords(get_absorption, triangle, u1, u2) * phi(u1, u2, i, boundary[i]) * phi(u1, u2, j, boundary[j])
    C = calculate_global_matrix(mesh=mesh,
                                func=absorption_func)
    plot_global_matrix(C, "images/zero_boundary_c_matrix.png")

    diffusion_func = lambda u1, u2, triangle, i, j, boundary: send_func_to_cartesian_coords(get_diffusion, triangle, u1, u2) * la.multi_dot([gradient_phi(u1, u2, i, boundary[i]) @ inverse_J_gK(triangle), gradient_phi(u1, u2, j, boundary[j]) @ inverse_J_gK(triangle)])
    K = calculate_global_matrix(mesh=mesh,
                                func=diffusion_func)
    plot_global_matrix(K, "images/zero_boundary_k_matrix.png")

    source_angles = 2*np.pi * np.linspace(0, 1, 6*(mesh.n_rings-1), endpoint=False) + np.pi / (6*(mesh.n_rings-1))
    source_radius = mesh.radius - 0.75*(mesh.radius / mesh.n_rings)
    source_coords = source_radius * np.column_stack([np.cos(source_angles), np.sin(source_angles)])
    source_values = np.repeat(1, len(source_angles))
    sources = source_coords, source_values

    q0 = calculate_source_vector(mesh, sources)
    print(q0)
    print(sum(q0))
    print(sum(source_values))

