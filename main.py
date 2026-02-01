import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numba import njit, float64, int64, uint64, boolean, prange
from numba.typed import List
import math

seed = 64
np.random.seed(seed)

# -----------------------------------------------------------------------------
# SPATIAL HASHING (NUMPY & NUMBA)
# -----------------------------------------------------------------------------

def build_spatial_index(points, radius):
    """
    Builds a spatial index using uint64 bit-packing and Numpy argsort.
    Upper 32 bits = Y index, Lower 32 bits = X index.
    Returns:
        sorted_keys: uint64 array of cell hashes
        sort_idx: indices to map sorted keys back to original point array
        grid_params: tuple of (min_x, min_y, cell_size)
    """
    if len(points) == 0:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.int64), (0.0, 0.0, 1.0)

    cell_size = 2.0 * radius
    min_xy = points.min(axis=0)
    
    # Vectorized grid coordinate calculation
    # We add a small epsilon to avoid boundary issues, though floor handles it mostly
    grid_coords = np.floor((points - min_xy) / cell_size).astype(np.uint64)
    
    gx = grid_coords[:, 0]
    gy = grid_coords[:, 1]
    
    # Pack bits: Y in upper 32, X in lower 32
    # keys = (gy << 32) | gx
    keys = (gy << np.uint64(32)) | gx
    
    # Sort indices based on keys
    sort_idx = np.argsort(keys)
    sorted_keys = keys[sort_idx]
    
    return sorted_keys, sort_idx, (min_xy[0], min_xy[1], cell_size)

@njit(inline='always')
def get_key(gx, gy):
    """Helper to pack coordinates into uint64 key."""
    return (uint64(gy) << uint64(32)) | uint64(gx)

@njit(fastmath=True)
def get_grid_coord(val, min_val, cell_size):
    return int64((val - min_val) / cell_size)

@njit(fastmath=True)
def is_circle_empty_packed(center, points, sorted_keys, sort_idx, min_x, min_y, cell_size, check_radius, ignore_i, ignore_j):
    """
    Checks if a circle is empty using binary search on the packed spatial keys.
    """
    cx, cy = center[0], center[1]
    r2 = check_radius * check_radius
    
    # Determine grid bounds for the circle
    min_gx = get_grid_coord(cx - check_radius, min_x, cell_size)
    min_gy = get_grid_coord(cy - check_radius, min_y, cell_size)
    max_gx = get_grid_coord(cx + check_radius, min_x, cell_size)
    max_gy = get_grid_coord(cy + check_radius, min_y, cell_size)
    
    # Iterate potential overlapping cells
    for gy in range(min_gy, max_gy + 1):
        if gy < 0: continue
        for gx in range(min_gx, max_gx + 1):
            if gx < 0: continue
            
            # Construct key for this cell
            key = get_key(gx, gy)
            
            # Find range in sorted array (Binary Search)
            start = np.searchsorted(sorted_keys, key)
            
            # If start is out of bounds or key doesn't match, cell is empty
            if start >= len(sorted_keys) or sorted_keys[start] != key:
                continue
                
            # Iterate through all points in this cell
            # We don't need 'end', we can just loop until key changes
            k = start
            while k < len(sorted_keys) and sorted_keys[k] == key:
                idx = sort_idx[k]
                k += 1
                
                if idx == ignore_i or idx == ignore_j:
                    continue
                
                dx = points[idx, 0] - cx
                dy = points[idx, 1] - cy
                
                if (dx*dx + dy*dy) < r2:
                    return False
                    
    return True

# -----------------------------------------------------------------------------
# GRAPH LOGIC (NUMBA)
# -----------------------------------------------------------------------------

@njit(parallel=False) # parallel often adds overhead for small N, keeping serial for robustness
def compute_connections_packed(points, sorted_keys, sort_idx, grid_params, radius):
    n = len(points)
    adj = List()
    for _ in range(n):
        adj.append(List.empty_list(int64))
    edges = List()
    
    if n < 2:
        return edges, adj
        
    min_x, min_y, cell_size = grid_params
    
    EPS = 1e-4
    check_r = radius * (1.0 - EPS)
    diameter_sq = (2 * radius) ** 2
    
    # Iterate over all points
    for i in range(n):
        px, py = points[i, 0], points[i, 1]
        
        # Grid coords of current point
        pgx = get_grid_coord(px, min_x, cell_size)
        pgy = get_grid_coord(py, min_y, cell_size)
        
        # Search 3x3 neighbor cells (since cell_size = 2*radius, 3x3 covers everything)
        # Actually since cell_size is diameter, neighbors are guaranteed to be in adjacent cells
        for dy in range(-1, 2):
            ny = pgy + dy
            if ny < 0: continue
            
            for dx in range(-1, 2):
                nx = pgx + dx
                if nx < 0: continue
                
                key = get_key(nx, ny)
                
                # Binary search for neighbor cell
                start = np.searchsorted(sorted_keys, key)
                
                if start >= len(sorted_keys) or sorted_keys[start] != key:
                    continue
                
                k = start
                while k < len(sorted_keys) and sorted_keys[k] == key:
                    j = sort_idx[k]
                    k += 1
                    
                    if j <= i: continue # Avoid duplicates
                    
                    p2 = points[j]
                    vec_x = p2[0] - px
                    vec_y = p2[1] - py
                    d2 = vec_x*vec_x + vec_y*vec_y
                    
                    if d2 > diameter_sq or d2 < 1e-9:
                        continue
                        
                    # Geometric Check (Rolling Ball)
                    mid_x = (px + p2[0]) * 0.5
                    mid_y = (py + p2[1]) * 0.5
                    
                    h_sq = radius**2 - d2/4.0
                    h = np.sqrt(max(0.0, h_sq))
                    
                    # Normal vector (rotated 90 deg)
                    norm_x, norm_y = vec_y, -vec_x
                    norm_len = np.sqrt(norm_x*norm_x + norm_y*norm_y)
                    norm_x /= norm_len
                    norm_y /= norm_len
                    
                    c1 = np.array([mid_x + h * norm_x, mid_y + h * norm_y])
                    c2 = np.array([mid_x - h * norm_x, mid_y - h * norm_y])
                    
                    valid1 = is_circle_empty_packed(c1, points, sorted_keys, sort_idx, min_x, min_y, cell_size, check_r, i, j)
                    valid2 = False
                    if not valid1:
                        valid2 = is_circle_empty_packed(c2, points, sorted_keys, sort_idx, min_x, min_y, cell_size, check_r, i, j)
                        
                    if valid1 or valid2:
                        edges.append((i, j))
                        adj[i].append(j)
                        adj[j].append(i)
                        
    return edges, adj

@njit
def find_components_numba(n_points, adj):
    visited = np.zeros(n_points, dtype=boolean)
    components = List()
    
    for start_node in range(n_points):
        if not visited[start_node]:
            if len(adj[start_node]) == 0:
                continue
                
            comp = List()
            stack = List([start_node])
            visited[start_node] = True
            comp.append(start_node)
            
            head = 0
            while head < len(stack):
                u = stack[head]
                head += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        comp.append(v)
                        stack.append(v)
            components.append(comp)
    return components

@njit(fastmath=True)
def get_angle_numba(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

@njit
def trace_boundaries_numba(points, adj):
    n_points = len(points)
    components = find_components_numba(n_points, adj)
    boundaries = List()
    
    for k in range(len(components)):
        comp = components[k]
        if len(comp) < 2: continue
            
        start_node = comp[0]
        min_y = points[start_node, 1]
        min_x = points[start_node, 0]
        
        # Find strictly bottom-left node
        for idx in comp:
            py = points[idx, 1]
            px = points[idx, 0]
            if py < min_y or (py == min_y and px < min_x):
                min_y = py
                min_x = px
                start_node = idx
                
        curr_node = start_node
        incoming_angle = -math.pi / 2 
        
        path = List()
        path.append(start_node)
        
        first_edge_start = -1
        first_edge_end = -1
        
        max_steps = len(comp) * 4 + 200
        
        for _ in range(max_steps):
            neighbors = adj[curr_node]
            if len(neighbors) == 0: break
                
            best_neighbor = -1
            min_angle_diff = 1e18
            best_angle = 0.0
            
            for n in neighbors:
                angle_to_n = get_angle_numba(points[curr_node], points[n])
                diff = (angle_to_n - incoming_angle) % (2 * math.pi)
                if diff < 1e-9: diff = 2 * math.pi
                    
                if diff < min_angle_diff:
                    min_angle_diff = diff
                    best_neighbor = n
                    best_angle = angle_to_n
            
            if best_neighbor == -1: break
            
            if first_edge_start == -1:
                first_edge_start = curr_node
                first_edge_end = best_neighbor
            elif curr_node == first_edge_start and best_neighbor == first_edge_end:
                break
            
            path.append(best_neighbor)
            curr_node = best_neighbor
            incoming_angle = best_angle + math.pi
            
        boundaries.append(path)
        
    return boundaries

# -----------------------------------------------------------------------------
# PYTHON VISUALIZATION
# -----------------------------------------------------------------------------

def generate_lidar_scene():
    all_points = []
    
    def add_rect(cx, cy, w, h, angle):
        num_pts = np.random.randint(10, 20)
        pw = np.linspace(-w/2, w/2, num_pts)
        ph = np.linspace(-h/2, h/2, num_pts)
        pts = np.vstack([
            np.column_stack([pw, np.full_like(pw, -h/2)]),
            np.column_stack([pw, np.full_like(pw, h/2)]),
            np.column_stack([np.full_like(ph, -w/2), ph]),
            np.column_stack([np.full_like(ph, w/2), ph])
        ])
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return pts @ R.T + [cx, cy]

    for _ in range(np.random.randint(4, 7)):
        choice = np.random.choice(['rect', 'arc', 'line'])
        x, y = np.random.uniform(-15, 15, 2)
        if choice == 'rect':
            all_points.append(add_rect(x, y, np.random.uniform(2, 6), 
                                       np.random.uniform(2, 6), np.random.uniform(0, np.pi)))
        elif choice == 'arc':
            r = np.random.uniform(1, 4)
            t = np.linspace(0, np.random.uniform(np.pi, 2*np.pi), 15)
            all_points.append(np.column_stack([x + r*np.cos(t), y + r*np.sin(t)]))
        else:
            length = np.random.uniform(5, 12)
            t = np.linspace(0, length, 15)
            ang = np.random.uniform(0, np.pi)
            line = np.column_stack([t - length/2, np.zeros_like(t)])
            c, s = np.cos(ang), np.sin(ang)
            all_points.append(line @ np.array([[c, -s], [s, c]]).T + [x, y])

    if not all_points: return np.empty((0, 2), dtype=np.float64)
    points = np.vstack(all_points)
    points += np.random.normal(0, 0.1, points.shape)
    return points.astype(np.float64)

def update_visualization():
    ax.clear()
    points = generate_lidar_scene()
    if len(points) == 0:
        fig.canvas.draw()
        return

    radius_param = 1.2
    
    # 1. Build Spatial Index using Numpy (uint64 packing)
    sorted_keys, sort_idx, grid_params = build_spatial_index(points, radius_param)
    
    # 2. Compute Graph using Numba
    edges, adj = compute_connections_packed(points, sorted_keys, sort_idx, grid_params, radius_param)
    
    # 3. Trace Boundaries using Numba
    boundary_indices = trace_boundaries_numba(points, adj)
    
    # -- Plotting --
    ax.scatter(points[:, 0], points[:, 1], color='gray', s=10, alpha=0.4, label='LIDAR Points', zorder=1)

    if len(edges) > 0:
        from matplotlib.collections import LineCollection
        segs = [[points[e[0]], points[e[1]]] for e in edges]
        lc = LineCollection(segs, colors='green', linewidths=0.5, alpha=0.3, zorder=1)
        ax.add_collection(lc)
    
    for b_inds in boundary_indices:
        path_points = points[np.array(b_inds)]
        ax.plot(path_points[:, 0], path_points[:, 1], color='red', linewidth=1.5, alpha=0.8, zorder=2)
        ax.scatter(path_points[:, 0], path_points[:, 1], color='red', s=15, zorder=3)

    ax.set_title(f"Edges: {len(edges)} | Boundaries: {len(boundary_indices)}", fontsize=10)
    
    custom_lines = [
        Line2D([0], [0], color='green', lw=1, alpha=0.3, label='Internal Graph'),
        Line2D([0], [0], color='red', lw=1.5, label='Outer Trace')
    ]
    ax.legend(handles=custom_lines, loc='upper right', fontsize='small')
    ax.axis('equal')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.canvas.draw()

def on_key(event):
    global seed
    if event.key == 'enter':
        seed += 1
        print(f'{seed=}')
        np.random.seed(seed)
        update_visualization()

fig, ax = plt.subplots(figsize=(10, 8))
fig.canvas.mpl_connect('key_press_event', on_key)
print("Compiling Numba functions (Spatial Hash)...")
update_visualization()
print("Ready. Press 'Enter' to regenerate.")
plt.show()
