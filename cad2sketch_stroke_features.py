import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import splprep, splev, CubicSpline





def build_final_edges_json(final_edges_json):
    node_features_list = []

    for key in final_edges_json.keys():
        stroke = final_edges_json[key]

        geometry = stroke["geometry"]

        node_feature = build_node_features(geometry)

        node_features_list.append(node_feature)

    node_features_matrix = np.array(node_features_list)

    return node_features_matrix



def build_all_edges_json(all_edges_json):
    node_features_list = []

    for stroke in all_edges_json:
        geometry = stroke["geometry"]

        node_feature = build_node_features(geometry)
        
        node_features_list.append(node_feature)

    node_features_matrix = np.array(node_features_list)

    return node_features_matrix


# ------------------------------------------------------------------------------------# 


# Straight Line: 10 values + type 1
# 0-2: point1, 3-5:point2, 6:alpha_value, 7-9: 0

# Circle Feature: 10 values + type 2
# 0-2: center, 3-5:normal, 6:alpha_value, 7:radius, 8-9: 0

# Arc Feature: 10 values + type 3
# 0-2: point1, 3-5:point2, 6:alpha_value, 7-9:center

# Ellipse Feature: 10 values + type 4
# 0-2: center, 3-5:normal, 6:alpha_value, 7: major axis, 8: minor axis, 9: orientation

# Closed Line: 10 values + type 5
# 0-2: point1, 3-5: point2, 6:alpha_value, 7-9: random point in the line

# Curved Line: 10 values + type 6
# 0-2: point1, 3-5: point2, 6:alpha_value, 7-9: random point in the line


def build_node_features(geometry):
    num_points = len(geometry)

    # Case 1: Check if the geometry has only 2 points  -> (Straight Line)
    if num_points == 2:
        point1 = geometry[0]
        point2 = geometry[1]

        return point1 + point2 + [0, 0, 0, 1]

    # Check if geometry is closed
    distance, closed = is_closed_shape(geometry)

    if not closed or len(geometry) < 5:
        center_circle, radius_circle, normal_circle, circle_residual = fit_circle_3d(geometry)

        if circle_residual < dist(geometry) * 2:
            # Case 3: Arc
            point1 = geometry[0]
            point2 = geometry[-1]
            return point1 + point2 + center_circle + [3]

        # Case 6: Curved Line
        point1 = geometry[0]
        point2 = geometry[-1]
        random_point = geometry[len(geometry) // 2]
        return point1 + point2 + random_point + [6]



    # Try fitting a circle
    center_circle, radius_circle, normal_circle, circle_residual = fit_circle_3d(geometry)

    if circle_residual > dist(geometry) * 2:
        # Case 5: Closed Shape
        point1 = geometry[0]
        point2 = geometry[-1]
        random_point = geometry[len(geometry) // 2]

        tolerance = dist(geometry) * 2

        return point1 + point2 + random_point + [5]



    # Try fitting an ellipse
    center_ellipse, normal_ellipse, axes_lengths, theta, ellipse_residual = fit_ellipse_3d(geometry)
    major_axis, minor_axis = axes_lengths

    if abs(major_axis - minor_axis) < dist(geometry) * 2:
        # Case 4: Ellipse
        return center_ellipse + normal_ellipse+ [major_axis, minor_axis, 0, 4]
    
    # Case 2: Circle
    return center_circle + normal_circle + [radius_circle, 0, 0, 2]



# ------------------------------------------------------------------------------------# 
def fit_circle_3d(points):
    """
    Fit a circle directly in 3D space using non-linear least squares optimization.
    The normal vector is pre-computed and used to simplify the fitting process.

    Parameters:
        points (np.ndarray): An (N, 3) array of 3D points.

    Returns:
        center (np.ndarray): The center of the fitted circle.
        radius (float): The radius of the fitted circle.
        normal (np.ndarray): The normal vector of the fitted circle.
        mean_residual (float): The mean residual of the fit.
    """
    
    # Pre-compute the normal using the given points
    normal = compute_normal(points)

    
    def residuals(params, points, normal):
        """
        Compute residuals (distances) from the points to the circle defined by params.
        
        Parameters:
            params: [x_c, y_c, z_c, radius]
                - (x_c, y_c, z_c): Center of the circle
                - radius: Radius of the circle
            points: The input 3D points.
            normal: The normal vector of the plane.
        Returns:
            Residuals as distances from the points to the circle.
        """
        center = params[:3]
        radius = params[3]
        
        # Normalize the normal vector to ensure it has unit length
        normal = normal / np.linalg.norm(normal)
        
        # Calculate vector from center to each point
        vecs = points - center
        
        # Check if any input is invalid
        if not np.isfinite(vecs).all() or not np.isfinite(normal).all():
            return np.full(len(points), 1e10)  # Return a large value if inputs are invalid

        # Project the vectors onto the plane defined by the normal
        dot_products = np.dot(vecs, normal)
        vecs_proj = vecs - np.outer(dot_products, normal)
        
        # Calculate distances from the projected points to the circle's radius
        distances = np.linalg.norm(vecs_proj, axis=1) - radius
        
        # Replace any NaNs or infinite values with a large number
        distances = np.nan_to_num(distances, nan=1e10, posinf=1e10, neginf=-1e10)

        return distances

    # Step 1: Estimate initial parameters
    center_init = np.mean(points, axis=0)
    radius_init = np.mean(np.linalg.norm(points - center_init, axis=1))
    params_init = np.hstack([center_init, radius_init])

    # Step 2: Use least squares optimization to fit the circle
    result = least_squares(residuals, params_init, args=(points, normal))
    
    # Extract optimized parameters
    center_opt = result.x[:3]
    radius_opt = result.x[3]
    final_residuals = residuals(result.x, points, normal)

    # print("points", points[0], points[-1])
    # print("center_init", center_init)
    # print("center_opt:", center_opt)
    # print("radius_opt", radius_opt)
    # print("normal:", normal)
    # print("np.mean(np.abs(final_residuals))", np.mean(np.abs(final_residuals)))
    # print("-------")

    return list(center_opt), radius_opt, list(normal), np.mean(np.abs(final_residuals))



def check_if_arc(points, center, radius, normal):
    # Step 1: Calculate vectors from the center to each point
    vecs = points - center
    
    # Step 2: Project the vectors onto the plane defined by the normal vector
    vecs_proj = vecs - np.outer(np.dot(vecs, normal), normal)
    
    # Step 3: Calculate angles of the projected points relative to a reference vector
    ref_vector = vecs_proj[0] / np.linalg.norm(vecs_proj[0])
    angles = np.arctan2(
        np.dot(vecs_proj, np.cross(normal, ref_vector)),
        np.dot(vecs_proj, ref_vector)
    )
    
    # Normalize angles to [0, 2*pi]
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    
    # Step 4: Calculate the angular range covered by the points
    min_angle = np.min(angles)
    max_angle = np.max(angles)
    raw_angle = max_angle - min_angle
    angle_range = min(raw_angle, (6.28-raw_angle))
    
    # Step 5: Determine if the points form an arc or a full circle
    is_arc = angle_range < 2 * np.pi - 0.01  # Allow a small tolerance for numerical errors
    return angle_range, is_arc



def fit_ellipse_3d(points):
    
    def residuals(params, points):
        center = params[:3]
        normal = params[3:6]
        a = params[6]
        b = params[7]
        theta = params[8]
        
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Calculate vectors from center to each point
        vecs = points - center
        
        # Project vectors onto the plane defined by the normal vector
        vecs_proj = vecs - np.outer(np.dot(vecs, normal), normal)
        
        # Define the major and minor axis direction vectors in the plane
        major_axis_dir = np.array([np.cos(theta), np.sin(theta), 0])
        minor_axis_dir = np.array([-np.sin(theta), np.cos(theta), 0])
        
        # Project onto the ellipse axes
        x_proj = np.dot(vecs_proj, major_axis_dir)
        y_proj = np.dot(vecs_proj, minor_axis_dir)
        
        # Compute the residuals using the ellipse equation
        residuals = (x_proj / a)**2 + (y_proj / b)**2 - 1
        return residuals

    # Step 1: Use fit_circle_3d to find an initial estimate for the plane
    center_init, _, normal_init, _ = fit_circle_3d(points)
    center_init = np.array(center_init)
    normal_init = np.array(normal_init)

    
    # Step 2: Estimate initial parameters for the ellipse
    a_init = np.max(np.linalg.norm(points - center_init, axis=1))
    b_init = a_init * 0.5  # Initial guess for minor axis
    theta_init = 0.0  # Initial guess for the orientation
    params_init = np.hstack([center_init, normal_init, a_init, b_init, theta_init])

    # Step 3: Use least squares optimization to fit the ellipse in 3D
    result = least_squares(residuals, params_init, args=(points,))
    
    # Extract optimized parameters
    center_opt = result.x[:3]
    normal_opt = result.x[3:6]
    a_opt = result.x[6]
    b_opt = result.x[7]
    theta_opt = result.x[8]
    
    # Normalize the normal vector
    normal_opt = normal_opt / np.linalg.norm(normal_opt)
    
    # Calculate major and minor axis directions
    major_axis_dir = np.array([np.cos(theta_opt), np.sin(theta_opt), 0])
    minor_axis_dir = np.array([-np.sin(theta_opt), np.cos(theta_opt), 0])
    
    # Calculate the mean residual
    final_residuals = residuals(result.x, points)
    mean_residual = np.mean(np.abs(final_residuals))
    
    return list(center_opt), list(normal_opt), (a_opt, b_opt), theta_opt, mean_residual



def is_closed_shape(points):
    points = np.array(points)
    distance = np.linalg.norm(points[0] - points[-1])
    tolerance = dist(points) * 2
    

    return distance, distance < tolerance


def dist(points):
    points = np.array(points)
    distance = np.linalg.norm(points[0] - points[1])
    
    return distance


def compute_normal(points):
    points = np.array(points)

    A = points[0]
    B = points[len(points) // 2]
    C = points[len(points) // 4]
    
    AB = B - A
    AC = C - A
    
    normal = np.cross(AB, AC)
    
    normal /= np.linalg.norm(normal)
    
    return normal




# ------------------------------------------------------------------------------------# 



def vis_final_edges(final_edges_json):
    """
    Visualize strokes in 3D space from the provided JSON data.
    Each stroke is defined by a series of 3D points.
    
    Parameters:
    - final_edges_json (dict): Dictionary containing stroke data with key as stroke ID and 
      value as a dictionary containing geometry (list of 3D points).
    """
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Loop through all strokes in the JSON
    for key in final_edges_json.keys():
        stroke = final_edges_json[key]
        geometry = stroke["geometry"]

        node_features = build_node_features(geometry)
        node_type = node_features[-1]

        if node_type == 1:
            color = 'black'
        if node_type == 2:
            color = 'green'
        if node_type == 3:
            color = 'blue'
        if node_type == 5:
            color = 'red'
        if node_type == 6:
            color = 'red'

        if len(geometry) < 2:
            continue
        
                
        # Plot each substroke (line segment between consecutive points)
        for j in range(1, len(geometry)):
            start = geometry[j - 1]
            end = geometry[j]
            
            # Extract coordinates for plotting
            x_values = [start[0], end[0]]
            y_values = [start[1], end[1]]
            z_values = [start[2], end[2]]
            
            # Plot the substroke as a line
            ax.plot(x_values, y_values, z_values, color=color, linewidth=0.5)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()




def vis_all_edges(all_edges_json):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Loop through all strokes in the JSON
    for stroke in all_edges_json:
        geometry = stroke['geometry']
        
        # Ensure geometry has at least two points
        if len(geometry) < 2:
            continue

        # Extract x, y, z coordinates
        x_values = [point[0] for point in geometry]
        y_values = [point[1] for point in geometry]
        z_values = [point[2] for point in geometry]

        # Plot the line
        ax.plot(x_values, y_values, z_values, color='black', linewidth=0.5)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()



def vis_all_edges_selected(all_edges_json, mask):
    """
    Visualizes all edges, with selected edges in black and unselected edges in red.

    Parameters:
    - all_edges_json: List of dictionaries containing stroke geometry.
    - mask: NumPy array of shape (num_edges, 1), where values > 0.5 indicate selected edges.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.set_axis_off()

    # Loop through all strokes and visualize them with different colors based on the mask
    for i, stroke in enumerate(all_edges_json):
        geometry = stroke['geometry']
        
        if len(geometry) < 2:
            continue  # Ensure at least two points exist

        # Extract x, y, z coordinates
        x_values = [point[0] for point in geometry]
        y_values = [point[1] for point in geometry]
        z_values = [point[2] for point in geometry]

        # Determine color: black if selected, red if not selected
        color = 'black' if mask[i, 0] > 0.5 else 'red'

        # Plot the stroke
        ax.plot(x_values, y_values, z_values, color=color, linewidth=0.5)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()



def vis_all_edges_only_selected(all_edges_json, mask):
    """
    Visualizes only the selected strokes where mask > 0.5 in black with linewidth=0.5.

    Parameters:
    - all_edges_json: List of dictionaries containing stroke geometry.
    - mask: NumPy array of shape (num_edges, 1), where values > 0.5 indicate selected edges.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.set_axis_off()

    # Loop through all strokes and visualize only the selected ones
    for i, stroke in enumerate(all_edges_json):
        if mask[i, 0] > 0.5:  # Only visualize selected strokes
            geometry = stroke['geometry']
            
            if len(geometry) < 2:
                continue  # Ensure at least two points exist

            # Extract x, y, z coordinates
            x_values = [point[0] for point in geometry]
            y_values = [point[1] for point in geometry]
            z_values = [point[2] for point in geometry]

            # Plot only selected strokes
            ax.plot(x_values, y_values, z_values, color='red', linewidth=0.5)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()

# ------------------------------------------------------------------------------------# 




def close(p1, p2, tol=1e-4):
    """
    Checks if two 3D points are close within a tolerance.

    Parameters:
    - p1, p2: Tuples or lists of (x, y, z) coordinates.
    - tol: Tolerance for closeness check.

    Returns:
    - Boolean indicating if points are close.
    """
    return all(abs(a - b) < tol for a, b in zip(p1, p2))


def simple_build_final_edges_features(final_edges_json, all_edges_json):
    """
    Builds a binary matrix indicating if each all_edges_json edge is found in final_edges_json.

    Parameters:
    - final_edges_json (dict): A dictionary where keys are edge IDs, and values contain edge geometries.
    - all_edges_json (list): A list of dictionaries where each represents an edge with its geometry.

    Returns:
    - numpy.ndarray: A matrix of shape (num_all_edges, 1) with 1 indicating a match and 0 otherwise.
    """
    num_all_edges = len(all_edges_json)
    match_matrix = np.zeros((num_all_edges, 1), dtype=np.int32)  # Initialize with 0

    for idx, all_edges_stroke in enumerate(all_edges_json):
        all_edges_geometry = all_edges_stroke.get("geometry", [])
        
        if len(all_edges_geometry) < 2:
            print(f"Skipping all_edges index {idx}: Insufficient geometry points.")
            continue  # Skip invalid edges

        all_edges_start = tuple(all_edges_geometry[0])  # First point
        all_edges_end = tuple(all_edges_geometry[-1])  # Last point

        # Check if this all_edges stroke matches any final_edges stroke
        for key, stroke in final_edges_json.items():
            final_edges_geometry = stroke.get("geometry", [])
            
            if len(final_edges_geometry) < 2:
                continue  # Skip invalid edges

            final_edges_start = tuple(final_edges_geometry[0])  # First point
            final_edges_end = tuple(final_edges_geometry[-1])  # Last point

            if (close(all_edges_start, final_edges_start) and close(all_edges_end, final_edges_end)) or \
               (close(all_edges_start, final_edges_end) and close(all_edges_end, final_edges_start)):
                match_matrix[idx] = 1  # Mark as used
                break  # No need to check further once matched

    # print(f"Number of matches: {np.sum(match_matrix)}")
    return match_matrix



def simple_build_all_edges_features(all_edges_json):
    """
    Builds a matrix with shape (num_all_edges, 7) where each row contains:
    - Start and end points (x, y, z) of an edge (6 values).
    - A binary flag (1 if stroke type is 'feature_line', else 0).

    Parameters:
    - all_edges_json (list): A list of dictionaries where each dictionary represents a stroke 
      with its geometry and type.

    Returns:
    - numpy.ndarray: A matrix of shape (num_all_edges, 7) with start, end points, and stroke type flag.
    """
    edge_features = []

    for stroke in all_edges_json:
        geometry = stroke['geometry']
        stroke_type = stroke.get('type', '')  # Get stroke type, default to empty if not found

        # Assign 1 if 'feature_line', otherwise 0
        type_flag = 1 if stroke_type == 'feature_line' else 0

        if len(geometry) < 2:
            print(f"Skipping stroke: Insufficient geometry points.")
            continue  # Skip if there aren't at least two points
        
        start = geometry[0]  # First point
        end = geometry[-1]  # Last point

        # Ensure we extract x, y, z coordinates only
        if len(start) >= 3 and len(end) >= 3:
            node_feature = [start[0], start[1], start[2], end[0], end[1], end[2], type_flag]
            edge_features.append(node_feature)
        else:
            print(f"Skipping stroke: Invalid start or end point format.")

    return np.array(edge_features, dtype=np.float32) if edge_features else np.empty((0, 7), dtype=np.float32)



def build_intersection_matrix(strokes_dict_data):
    """
    Builds an intersection matrix indicating which strokes intersect with others.
    
    Parameters:
    - strokes_dict_data (list): A list of dictionaries where each dictionary represents a stroke 
      and contains an 'intersections' key, which is a list of sublists with intersecting stroke indices.

    Returns:
    - numpy.ndarray: A matrix of shape (num_strokes_dict_data, num_strokes_dict_data),
      where a value of 1 indicates that a stroke intersects another stroke in a one-way manner.
    """
    num_strokes = len(strokes_dict_data)
    intersection_matrix = np.zeros((num_strokes, num_strokes), dtype=np.int32)  # Initialize with 0s

    for idx, stroke_dict in enumerate(strokes_dict_data):
        intersect_strokes = stroke_dict.get("intersections", [])  # Get intersection lists
        
        # Unfold the sublists to get all intersecting stroke indices
        intersecting_indices = {stroke_idx for sublist in intersect_strokes for stroke_idx in sublist}

        # Mark intersections in the matrix (acyclic, so only row updates)
        for intersecting_idx in intersecting_indices:
            if 0 <= intersecting_idx < num_strokes:  # Ensure index is valid
                intersection_matrix[idx, intersecting_idx] = 1  # One-way intersection

    return intersection_matrix
