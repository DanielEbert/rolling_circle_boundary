import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import KDTree
from collections import defaultdict


seed = 64
np.random.seed(seed)


def find_all_connections_within_radius(points, radius):
    """
    Find all valid connections between points using rolling ball criterion.
    """
    if len(points) < 2:
        return [], defaultdict(set)
    
    tree = KDTree(points)
    edges = []
    adjacency = defaultdict(set)
    EPS = 1e-4
    
    for i, p1 in enumerate(points):
        neighbor_indices = tree.query_ball_point(p1, 2 * radius)
        
        for j in neighbor_indices:
            if j <= i: continue
                
            p2 = points[j]
            d_vec = p2 - p1
            d2 = np.sum(d_vec**2)
            
            if d2 > (2 * radius)**2 or d2 < 1e-9: continue
            
            mid = (p1 + p2) / 2
            h_sq = radius**2 - d2/4
            h = np.sqrt(max(0, h_sq))
            
            norm_vec = np.array([d_vec[1], -d_vec[0]])
            norm_len = np.linalg.norm(norm_vec)
            if norm_len < 1e-9: continue
            norm_vec = norm_vec / norm_len
            
            center1 = mid + h * norm_vec
            center2 = mid - h * norm_vec
            
            points_in_circle1 = tree.query_ball_point(center1, radius * (1 - EPS))
            points_in_circle2 = tree.query_ball_point(center2, radius * (1 - EPS))
            
            circle1_valid = set(points_in_circle1) <= {i, j}
            circle2_valid = set(points_in_circle2) <= {i, j}
            
            if circle1_valid or circle2_valid:
                edges.append((i, j))
                adjacency[i].add(j)
                adjacency[j].add(i)
    
    return edges, adjacency


def find_connected_components(adjacency):
    if not adjacency:
        return []
    
    visited = set()
    components = []
    
    nodes = sorted(list(adjacency.keys())) # Sort for deterministic behavior
    for start_node in nodes:
        if start_node in visited:
            continue
        
        component = set()
        queue = [start_node]
        visited.add(start_node)
        component.add(start_node)
        
        while queue:
            node = queue.pop(0)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        
        components.append(list(component))
    
    return components


def get_angle(p1, p2):
    """Returns angle of vector p1->p2 in range [0, 2pi)"""
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])


def trace_outer_boundaries(points, adjacency):
    """
    Traces the perimeter of the graph components using a Left-Wall Follower 
    (CCW pivoting) algorithm. This handles loops, single lines, and trees.
    """
    components = find_connected_components(adjacency)
    boundaries = []
    
    for comp in components:
        if len(comp) < 2:
            continue
            
        # 1. Start at the bottom-left-most point.
        # This point is guaranteed to be on the convex hull.
        start_node = min(comp, key=lambda i: (points[i][1], points[i][0]))
        
        # 2. Determine initial direction.
        # We simulate coming from directly BELOW the start node.
        # Virtual previous node is (x, y-1).
        prev_node_idx = -1 # Sentinel
        curr_node_idx = start_node
        
        # Current heading is UP (90 deg), because we came from below.
        # We want to scan CCW from the vector (0, -1) -> (0, 0).
        # Vector Start->Prev is Down (-90 deg / 270 deg).
        incoming_angle = -np.pi / 2 
        
        path_indices = [start_node]
        
        # We need to detect when we traverse the *first edge* again to stop.
        first_edge = None
        
        # Safety limit
        max_steps = len(adjacency) * 4 + 100
        
        for _ in range(max_steps):
            neighbors = list(adjacency[curr_node_idx])
            
            # If isolated (shouldn't happen given comp check, but for safety)
            if not neighbors:
                break

            # Find the best neighbor by sweeping Counter-Clockwise from the incoming edge.
            best_neighbor = -1
            min_angle_diff = np.inf
            best_angle = 0
            
            for n in neighbors:
                # Angle of outgoing edge to neighbor
                angle_to_n = get_angle(points[curr_node_idx], points[n])
                
                # Calculate diff in range [0, 2pi) moving CCW from incoming_angle
                # We subtract incoming_angle from angle_to_n to get rotation
                # But 'incoming_angle' is the angle of vector (Curr->Prev).
                # Wait, standard logic:
                # Vector In: P->C. Vector Out: C->N.
                # We want the angle C->N to be the smallest CCW rotation from P->C.
                # Angle P->C is (incoming_angle + pi).
                
                # Actually, simpler: 
                # Angle of "Wall" is incoming_angle. We want the first neighbor 
                # appearing after incoming_angle in CCW direction.
                
                diff = (angle_to_n - incoming_angle) % (2 * np.pi)
                
                # If diff is 0, it means we are going exactly backwards. 
                # This is allowed (dead end), but we prefer other branches if they exist.
                # However, mathematically, 0 is 2pi in a sweep.
                # Let's treat 0 as 2pi so it's the last resort (u-turn) unless it's the only option.
                if diff < 1e-9:
                    diff = 2 * np.pi

                if diff < min_angle_diff:
                    min_angle_diff = diff
                    best_neighbor = n
                    best_angle = angle_to_n
            
            # Record the first edge we take
            if first_edge is None:
                first_edge = (curr_node_idx, best_neighbor)
            elif (curr_node_idx, best_neighbor) == first_edge:
                # We are about to walk the starting edge again. Cycle complete.
                break
            
            # Move
            path_indices.append(best_neighbor)
            
            # Update state for next step
            # The new incoming angle is the angle of vector (New_Curr -> Old_Curr)
            # which is (best_angle + pi)
            prev_node_idx = curr_node_idx
            curr_node_idx = best_neighbor
            incoming_angle = best_angle + np.pi
            
        boundaries.append(points[path_indices])
        
    return boundaries


def generate_lidar_scene():
    """Generates a complex LIDAR scene with multiple obstacle clusters."""
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

    points = np.vstack(all_points)
    points += np.random.normal(0, 0.1, points.shape)
    return points


def update_visualization():
    ax.clear()
    points = generate_lidar_scene()
    
    radius_param = 1.2
    edges, adjacency = find_all_connections_within_radius(points, radius_param)
    
    # LIDAR points
    ax.scatter(points[:, 0], points[:, 1], color='gray', s=10, alpha=0.4, 
               label='LIDAR Points', zorder=1)

    # valid connections (edges)
    for i, j in edges:
        p1, p2 = points[i], points[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=0.5, alpha=0.3, zorder=1)
    
    boundary_paths = trace_outer_boundaries(points, adjacency)
    
    for b_path in boundary_paths:
        # Plot the boundary line
        # Slightly offset the red line width or alpha to see retracing on single lines
        ax.plot(b_path[:, 0], b_path[:, 1], color='red', linewidth=1.5, 
                alpha=0.8, zorder=2)
        
        # Mark vertices on boundary
        ax.scatter(b_path[:, 0], b_path[:, 1], color='red', s=15, zorder=3)
        
        # Mark start
        # ax.plot(b_path[0, 0], b_path[0, 1], 'bx', markersize=8, markeredgewidth=2, zorder=4)

    # Stats
    total_edges = len(edges)
    ax.set_title(f"Edges: {total_edges} | Boundaries: {len(boundary_paths)}", fontsize=10)
    
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


# Setup Figure
fig, ax = plt.subplots(figsize=(10, 8))
fig.canvas.mpl_connect('key_press_event', on_key)

print("Visualization ready. Press 'Enter' to regenerate.")
update_visualization()
plt.show()