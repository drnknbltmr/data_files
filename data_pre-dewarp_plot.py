import numpy as np
from scipy.interpolate import CubicSpline
import h5py
import matplotlib.pyplot as plt

def read_h5(case, file_directory):
    with h5py.File(file_directory, 'r') as f:
        case_idx = case - 1
        u_ref = f['u'][case_idx, 0]
        v_ref = f['v'][case_idx, 0]
        x = f['x'][()] * 1e-3
        y = f['y'][()] * 1e-3
        u_data = f[u_ref][:]
        v_data = f[v_ref][:]
    return x, y, u_data, v_data

def crop_plot(x, y, u_data, v_data, x_left, x_right, y_bottom, y_top):
    x = x[x_left:-x_right, y_bottom:-y_top]
    y = y[x_left:-x_right, y_bottom:-y_top]
    u_data = u_data[:, x_left:-x_right, y_bottom:-y_top]
    v_data = v_data[:, x_left:-x_right, y_bottom:-y_top]
    return x, y, u_data, v_data

def read_airfoil(x, file_directory):
    foil_coords = np.loadtxt(file_directory)
    chord_length = 0.2
    foil_coords[:, 0] *= chord_length
    foil_coords[:, 1] *= chord_length
    foil_trimmed = foil_coords[(foil_coords[:, 0] >= np.min(x)) & (foil_coords[:, 0] <= np.max(x))]
    return foil_trimmed

def create_normal_lines_plot(x, y, u_data, v_data, foil,dpi, num_normal_lines=400):
    x_foil, y_foil = foil[:, 0], foil[:, 1]

    # Create dense normals along airfoil
    dx = np.diff(x_foil)
    dy = np.diff(y_foil)
    arc_length = np.concatenate(([0], np.cumsum(np.sqrt(dx**2 + dy**2))))
    cs_x = CubicSpline(arc_length, x_foil)
    cs_y = CubicSpline(arc_length, y_foil)
    s_new = np.linspace(0, arc_length[-1], num_normal_lines)
    x_dense = cs_x(s_new)
    y_dense = cs_y(s_new)

    # Calculate normals
    dxds = cs_x(s_new, 1)
    dyds = cs_y(s_new, 1)
    normals = np.column_stack((dyds, -dxds))
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norms == 0, 1, norms)

    # Orient normals outward
    centroid = np.array([np.mean(x_foil), np.mean(y_foil)])
    for i in range(len(normals)):
        vec_to_centroid = centroid - np.array([x_dense[i], y_dense[i]])
        if np.dot(normals[i], vec_to_centroid) > 0:
            normals[i] *= -1

    # Find valid data regions
    mask = ~np.isnan(u_data[0]) & ~np.isnan(v_data[0])
    x_valid = x[mask]
    y_valid = y[mask]
    x_min, x_max = np.min(x_valid), np.max(x_valid)
    y_min, y_max = np.min(y_valid), np.max(y_valid)

    # Calculate maximum normal length (t_min)
    t_list = []
    for xp, yp, nx, ny in zip(x_dense, y_dense, normals[:, 0], normals[:, 1]):
        t_candidates = []
        if nx != 0:
            t_x = (x_max - xp)/nx if nx > 0 else (x_min - xp)/nx
            if t_x > 0: t_candidates.append(t_x)
        if ny != 0:
            t_y = (y_max - yp)/ny if ny > 0 else (y_min - yp)/ny
            if t_y > 0: t_candidates.append(t_y)
        t_list.append(min(t_candidates) if t_candidates else 0)
    t_min = min(t_list) if t_list else 0.02

    # Get global min/max for color scaling
    u_frame = np.nanmean(u_data,axis=0)
    v_frame = np.nanmean(v_data,axis=0)
    global_min = min(np.nanmin(u_frame), np.nanmin(v_frame))
    global_max = max(np.nanmax(u_frame), np.nanmax(v_frame))

    # === U Plot ===
    fig_u, ax_u = plt.subplots(figsize=(12, 7))
    cf_u = ax_u.pcolormesh(x.T, y.T, u_frame.T, shading='auto', cmap='jet',
                           vmin=global_min, vmax=global_max)
    ax_u.plot(foil[:, 0], foil[:, 1], 'k-', lw=2, label='Airfoil')
    for i, (xp, yp, nx, ny) in enumerate(zip(x_dense, y_dense, normals[:, 0], normals[:, 1])):
        ax_u.plot([xp, xp + nx * t_min], [yp, yp + ny * t_min], 'k-', lw=0.5)
    ax_u.set_title(f'Case {case} - Horizontal Velocity (u)')
    ax_u.set_xlabel('x (m)')
    ax_u.set_ylabel('y (m)')
    ax_u.set_aspect('equal')
    ax_u.set_xlim(x_min, x_max)
    ax_u.set_ylim(y_min, y_max)
    fig_u.colorbar(cf_u, ax=ax_u, label='magnitude (m/s)', shrink=0.4)
    plt.tight_layout()
    fig_u.savefig(f'pre_dewarp_u_case_{case}.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig_u)

    # === V Plot ===
    fig_v, ax_v = plt.subplots(figsize=(12, 7))
    cf_v = ax_v.pcolormesh(x.T, y.T, v_frame.T, shading='auto', cmap='jet',
                           vmin=global_min, vmax=global_max)
    ax_v.plot(foil[:, 0], foil[:, 1], 'k-', lw=2, label='Airfoil')
    for i, (xp, yp, nx, ny) in enumerate(zip(x_dense, y_dense, normals[:, 0], normals[:, 1])):
        ax_v.plot([xp, xp + nx * t_min], [yp, yp + ny * t_min], 'k-', lw=0.5)
    ax_v.set_title(f'Case {case} - Vertical Velocity (v)')
    ax_v.set_xlabel('x (m)')
    ax_v.set_ylabel('y (m)')
    ax_v.set_aspect('equal')
    ax_v.set_xlim(x_min, x_max)
    ax_v.set_ylim(y_min, y_max)
    fig_v.colorbar(cf_v, ax=ax_v, label='magnitude (m/s)', shrink=0.4)
    plt.tight_layout()
    fig_v.savefig(f'pre_dewarp_v_case_{case}.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig_v)

# Main processing
case = 2

x, y, u_data, v_data = read_h5(case, 'TAS_DATA.h5')
x, y, u_data, v_data = crop_plot(x, y, u_data, v_data, 35, 70, 10, 70)
foil = read_airfoil(x, 'NACA0018.txt')

# Generate and save plots
create_normal_lines_plot(x, y, u_data, v_data, foil,1000,400)

