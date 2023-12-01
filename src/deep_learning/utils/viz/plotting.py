import numpy as np


def interp(data, interpol=1):
    n, m = data.shape
    data_inter = np.zeros((n + (n - 1) * (interpol - 1), m + (m - 1) * (interpol - 1)))
    x_i = np.linspace(0, 1, interpol + 1)
    y_i = np.linspace(0, 1, interpol + 1)

    for row in range(data.shape[0] - 1):
        for col in range(data.shape[1] - 1):
            d = data[row : (row + 2), col : (col + 2)]
            inter_d = (
                np.array([1 - x_i, x_i])
                .T.dot(d)
                .dot(np.array([[1 - y_i], [y_i]])[:, 0, :])
            )
            data_inter[
                row * interpol : (row + 1) * interpol + 1,
                col * interpol : (col + 1) * interpol + 1,
            ] = inter_d

    return data_inter


def plot_mesh(axis, data, ranges, interpolate=1, ticks=None, zlims=[]):
    x_range, y_range = ranges
    data_interp = interp(data, interpolate)

    x_r = np.linspace(min(x_range), max(x_range), data_interp.shape[1])
    y_r = np.linspace(max(y_range), min(y_range), data_interp.shape[0])

    x, y = np.meshgrid(x_r, y_r)
    axis.plot_surface(
        x,
        y,
        data_interp,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    # axis.plot_surface(x, y, data)
    axis.view_init(azim=-25)
    axis.set_xlabel("House Shows")
    axis.set_ylabel("Player Shows")
    axis.set_zlabel("Value")
    if ticks is not None:
        axis.set(yticks=x_range, yticklabels=ticks)
    if zlims:
        axis.set_zlim(*zlims)


__all__ = ["plot_mesh"]
