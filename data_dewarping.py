import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator, CubicSpline, interp1d
import h5py
import matplotlib.pyplot as plt

print('Dewarping...')

def read_h5(case,file_directory ):
    with h5py.File(file_directory, 'r') as f:
        case_idx = case - 1
        u_ref = f['u'][case_idx, 0]
        v_ref = f['v'][case_idx, 0]
        x = f['x'][()] * 1e-3
        y = f['y'][()] * 1e-3
        u_data = f[u_ref][:]
        v_data = f[v_ref][:]
    return x,y,u_data,v_data

def crop_plot(x, y, u_data, v_data, x_left, x_right, y_bottom, y_top):
    x = x[x_left:-x_right, y_bottom:-y_top]
    y = y[x_left:-x_right, y_bottom:-y_top]
    u_data = u_data[:, x_left:-x_right, y_bottom:-y_top]
    v_data = v_data[:, x_left:-x_right, y_bottom:-y_top]
    return x, y, u_data, v_data

def read_airfoil(x,file_directory):
    foil_coords = np.loadtxt(file_directory)
    chord_length = 0.2
    foil_coords[:, 0] *= chord_length
    foil_coords[:, 1] *= chord_length
    foil_trimmed = foil_coords[(foil_coords[:, 0] >= np.min(x)) & (foil_coords[:, 0] <= np.max(x))]
    return foil_trimmed


def dewarp(u_data, v_data, x, y, foil, num_normal_lines, num_points, plot_initial=False):
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)

    def calculate_line_limits(x_foil, y_foil, normal):
        nx, ny = normal
        t_values = []
        if nx != 0:
            t_xmax = (x_max - x_foil) / nx
            t_values.append(t_xmax)
        if ny != 0:
            t_ymax = (y_max - y_foil) / ny
            t_values.append(t_ymax)
        valid_t = [t for t in t_values if t > 0 and
                   (x_min <= x_foil + t * nx <= x_max) and
                   (y_min <= y_foil + t * ny <= y_max)]
        return 0, min(valid_t) if valid_t else 0

    def create_dense_normals(x_foil, y_foil, num_normal_lines):
        dx = np.diff(x_foil)
        dy = np.diff(y_foil)
        arc_length = np.concatenate(([0], np.cumsum(np.sqrt(dx ** 2 + dy ** 2))))
        cs_x = CubicSpline(arc_length, x_foil)
        cs_y = CubicSpline(arc_length, y_foil)
        s_new = np.linspace(0, arc_length[-1], num_normal_lines)
        x_dense = cs_x(s_new)
        y_dense = cs_y(s_new)
        dxds = cs_x(s_new, 1)
        dyds = cs_y(s_new, 1)
        normals = np.column_stack((dyds, -dxds))
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norms == 0, 1, norms)
        centroid = np.array([np.mean(x_foil), np.mean(y_foil)])
        for i in range(len(normals)):
            vec_to_centroid = centroid - np.array([x_dense[i], y_dense[i]])
        if np.dot(normals[i], vec_to_centroid) > 0:
            normals[i] *= -1
        return x_dense, y_dense, normals, s_new

    x_foil, y_foil = foil[:, 0], foil[:, 1]
    x_dense, y_dense, normals, arc_length = create_dense_normals(x_foil, y_foil, num_normal_lines)

    x_coords = np.unique(x[:, 0])
    y_coords = np.unique(y[0, :])
    interp_u = RegularGridInterpolator((x_coords, y_coords), u_data, method='linear', bounds_error=False,
                                       fill_value=np.nan)
    interp_v = RegularGridInterpolator((x_coords, y_coords), v_data, method='linear', bounds_error=False,
                                       fill_value=np.nan)
    t_max_list = []
    for i in range(len(x_dense)):
        _, t_max = calculate_line_limits(x_dense[i], y_dense[i], normals[i])
        if t_max > 0:
            t_max_list.append(t_max)
    global_min_t_max = min(t_max_list) if t_max_list else 0
    if global_min_t_max > 0:
        t_values = np.linspace(0, global_min_t_max, num_points)
        x_points = x_dense[:, None] + t_values * normals[:, 0][:, None]
        y_points = y_dense[:, None] + t_values * normals[:, 1][:, None]
        all_points = np.column_stack((x_points.ravel(), y_points.ravel()))
        u_vals = interp_u(all_points).reshape(len(x_dense), num_points)
        v_vals = interp_v(all_points).reshape(len(x_dense), num_points)
        nx = normals[:, 0, None]
        ny = normals[:, 1, None]
        rejections_u = u_vals * ny + v_vals * nx
        projections_v = u_vals * nx + v_vals * ny
        max_distance = global_min_t_max
    else:
        rejections_u = np.full((len(x_dense), num_points), np.nan)
        projections_v = np.full((len(x_dense), num_points), np.nan)
        max_distance = 0

    cropped_normals = []
    if global_min_t_max > 0:
        for i in range(len(x_dense)):
            points = np.column_stack((
                x_dense[i] + t_values * normals[i, 0],
                y_dense[i] + t_values * normals[i, 1]
            ))
            cropped_normals.append(points)

    if plot_initial and global_min_t_max > 0:
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        cmap = plt.get_cmap('jet')
        ax1 = axes[0]
        c1 = ax1.pcolormesh(x.T, y.T, np.ma.masked_invalid(u_data.T), shading='auto', cmap=cmap)
        ax1.plot(foil[:, 0], foil[:, 1], 'k', linewidth=2)
        for line in cropped_normals:
            ax1.plot(line[:, 0], line[:, 1], 'k-', linewidth=0.5, alpha=1)
        ax1.set_title("Horizontal Velocity with Normal Lines")
        ax1.axis('equal')
        fig.colorbar(c1, ax=ax1, label='u')
        ax2 = axes[1]
        c2 = ax2.pcolormesh(x.T, y.T, np.ma.masked_invalid(v_data.T), shading='auto', cmap=cmap)
        ax2.plot(foil[:, 0], foil[:, 1], 'k', linewidth=2)
        for line in cropped_normals:
            ax2.plot(line[:, 0], line[:, 1], 'k-', linewidth=0.5, alpha=1)
        ax2.set_title("Vertical Velocity with Normal Lines")
        ax2.axis('equal')
        fig.colorbar(c2, ax=ax2, label='v')
        plt.tight_layout()
        plt.show()

    delta = global_min_t_max / num_points if global_min_t_max > 0 else 0
    y_grid = np.linspace(0, max_distance, num_points)

    grid_shape = (len(y_grid), len(arc_length))
    rej_grid = np.full(grid_shape, np.nan)
    proj_grid = np.full(grid_shape, np.nan)

    if global_min_t_max > 0 and len(y_grid) > 0:
        indices = np.digitize(t_values, y_grid) - 1
        indices = np.clip(indices, 0, len(y_grid) - 1)

        for i, idx in enumerate(indices):
            rej_grid[idx, :] = rejections_u[:, i]
            proj_grid[idx, :] = projections_v[:, i]

    dewarp_extent = [arc_length[0], arc_length[-1], 0, max_distance]
    return np.fliplr(rej_grid), np.fliplr(proj_grid), dewarp_extent

dewarped_data = {}
for case in range(1, 6):
    x, y, u_data, v_data = read_h5(case, 'TAS_DATA.h5')
    x, y, u_data, v_data = crop_plot(x, y, u_data, v_data, 35, 70, 10, 70)
    if case == 1:
        foil = read_airfoil(x, 'NACA0018.txt')
    dewarped_u_list = []
    dewarped_v_list = []
    for i in range(u_data.shape[0]):
        if case == 1 and i == 0:
            dewarped_u_frame, dewarped_v_frame, dewarped_foil = dewarp(u_data[i], v_data[i], x, y, foil, 400, 100, False)
        else:
            dewarped_u_frame, dewarped_v_frame, _ = dewarp(u_data[i], v_data[i], x, y, foil, 400, 100, False)
        dewarped_u_list.append(dewarped_u_frame)
        dewarped_v_list.append(dewarped_v_frame)
    dewarped_u = np.stack(dewarped_u_list, axis=0)
    dewarped_v = np.stack(dewarped_v_list, axis=0)
    dewarped_data[f'case_{case}_dewarped_u'] = dewarped_u
    dewarped_data[f'case_{case}_dewarped_v'] = dewarped_v
np.savez('dewarped_data.npz', dewarped_foil=dewarped_foil, **dewarped_data)

print("Data saved successfully.")