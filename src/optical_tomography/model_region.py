import numpy as np
import matplotlib.pyplot as plt



c = float(2.5e8)


def get_absorption(x, y):
    r = 25
    variation_r = 3.125
    if (x-5)**2 + (y-5)**2 < variation_r**2:
        absorption = 0.05
    elif x**2 + y**2 < r**2:
        absorption = 0.025
    else:
        absorption = 0.0
    return absorption * c


def get_scattering(x, y):
    r = 25
    variation_r = 3.125
    if (x-5)**2 + (y-5)**2 < variation_r**2:
        scattering = 4.0
    elif x**2 + y**2 < r**2:
        scattering = 2.0
    else:
        scattering = 0.0
    return scattering


def get_diffusion(x, y):
    absorption = get_absorption(x, y) / c # remove the scaling by c
    scattering = get_scattering(x, y)
    if absorption == 0.0 and scattering == 0.0: # i.e. we're outside region of interest
        diffusion = 0.0
    else:
        diffusion = c / (3 * (absorption + scattering)) # assuming \bar{f} = 0
    return diffusion


def plot_function_on_R(f, x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound, file):
    X, Y = np.meshgrid(np.linspace(x_lower_bound, x_upper_bound, 512), np.linspace(y_lower_bound, y_upper_bound, 512))
    fxy = np.vectorize(f)(X, Y)
    fig, ax = plt.subplots()

    im = ax.imshow(
        fxy, 
        origin="lower",
        extent=[x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound],
        vmin = fxy.min(),
        vmax=fxy.max()
    )
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.savefig(file)
    plt.close(fig)


if __name__ == "__main__":
    plot_function_on_R(get_absorption, -40, 40, -40, 40, "images/absorption_region.png")
    plot_function_on_R(get_scattering, -40, 40, -40, 40, "images/scattering_region.png")
    plot_function_on_R(get_diffusion, -40, 40, -40, 40, "images/diffusion_region.png")
