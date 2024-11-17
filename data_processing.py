
import scipy.interpolate as interpolate
from scipy.spatial import distance
from scipy.spatial import KDTree
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from math import radians, cos, sin, asin, sqrt
from scipy.spatial.distance import directed_hausdorff, squareform
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from shapely.geometry import MultiLineString, LineString, Point, MultiPolygon
from pyproj import Transformer
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely import wkt
from tqdm import tqdm
from shapely.wkt import loads
import re
import math
def hausdorff(u, v):

    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d

def chamfer_distance(u, v):
    u=np.array(u, dtype=float)
    v=np.array(v, dtype=float)

    dist_matrix_uv = cdist(u, v, 'euclidean')
    min_dist_uv = np.min(dist_matrix_uv, axis=1)

    dist_matrix_vu = cdist(v, u, 'euclidean')
    min_dist_vu = np.min(dist_matrix_vu, axis=1)


    d = (np.sum(min_dist_uv) + np.sum(min_dist_vu)) / (len(u) + len(v))
    return d

def chamfer_Matrix(trajs1, trajs2):

    D_chamfer = np.zeros((len(trajs1), len(trajs2)))

    for i in range(len(trajs1)):
        for j in range(len(trajs2)):
            distance = chamfer_distance(trajs1[i], trajs2[j])
            D_chamfer[i, j] = distance
    return D_chamfer
def hausdorffMatrix(trajs1, trajs2):

    D_hsdf = np.zeros((len(trajs1), len(trajs2)))

    for i in range(len(trajs1)):
        for j in range(len(trajs2)):
            distance = hausdorff(trajs1[i], trajs2[j])
            D_hsdf[i, j] = distance
    return D_hsdf
def trans_crs(df):

    df=df.copy()
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
    df['point'] = df.apply(lambda x: transformer.transform(x.y, x.x), axis=1)
    df['x_proj'] = df['point'].apply(lambda x: x[0])
    df['y_proj'] = df['point'].apply(lambda x: x[1])
    df=df.drop(columns=['point'])
    return df

def resample_trajectory(traj, n_points):
    """
    Resample a trajectory by linear interpolation based on cumulative distance.
    Works for trajectories with multiple dimensions (e.g., [x, y]).
    
    :param traj: numpy array of shape (m, d), where m is the number of points in the trajectory, and d is the dimension.
    :param n_points: number of points to resample to.
    :return: numpy array of shape (n_points, d) containing the resampled trajectory.
    """
    if len(traj) < 2:
        raise ValueError("Trajectory must contain at least two points.")
    
    # Calculate distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)  # Total distance along the trajectory
    
    # Target distances for interpolation
    max_distance = cumulative_distances[-1]
    target_distances = np.linspace(0, max_distance, n_points)
    
    # Initialize the resampled trajectory array
    resampled_traj = np.zeros((n_points, traj.shape[1]))
    
    # Perform linear interpolation for each dimension
    for dim in range(traj.shape[1]):
        resampled_traj[:, dim] = np.interp(target_distances, cumulative_distances, traj[:, dim])

    return resampled_traj

def calculate_frechet_distance(traj1, traj2):
    # Implementation of the discrete Fréchet distance calculation
    # Here, a simple placeholder is used, please replace it with the actual calculation.
    from scipy.spatial.distance import cdist
    ca = np.full((len(traj1), len(traj2)), -1.0)
    
    def _c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = np.linalg.norm(traj1[0] - traj2[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i-1, 0), np.linalg.norm(traj1[i] - traj2[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j-1), np.linalg.norm(traj1[0] - traj2[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(min(_c(i-1, j), _c(i, j-1), _c(i-1, j-1)), np.linalg.norm(traj1[i] - traj2[j]))
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    return _c(len(traj1) - 1, len(traj2) - 1)


def point_to_line_dist(point, start, end):

    line_vec = end - start
    point_vec = point - start
    line_len = np.dot(line_vec, line_vec)
    if line_len == 0:
        return np.linalg.norm(point_vec)

    t = np.dot(point_vec, line_vec) / line_len

    if t < 0.0:
        nearest = start
    elif t > 1.0:
        nearest = end
    else:
        nearest = start + t * line_vec
    return np.linalg.norm(point - nearest)
def calculate_sequence_dis(traj1, traj2, n):
    def calculate_total_length(traj):

        distances = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
        return np.sum(distances)

    def resample_trajectory(traj, n):

        total_length = calculate_total_length(traj)

        segment_length = total_length / (n - 1)


        sampled_points = [traj[0]]
        current_length = 0
        current_segment_length = 0

        for i in range(1, len(traj)):

            point_distance = np.sqrt(np.sum((traj[i] - traj[i - 1])**2))
            current_length += point_distance

            while current_length >= current_segment_length + segment_length and len(sampled_points) < n:
                current_segment_length += segment_length
                sampled_points.append(traj[i])


        if len(sampled_points) < n:
            sampled_points.append(traj[-1])

        return np.array(sampled_points)

    traj1P_array = resample_trajectory(traj1, n)
    traj2P_array = resample_trajectory(traj2, n)

    dist_matrix_traj1_to_traj2 = distance.cdist(traj1P_array, traj2P_array)

    min_dists_traj1_to_traj2 = np.min(dist_matrix_traj1_to_traj2, axis=1)
    

    dist_matrix_traj2_to_traj1 = distance.cdist(traj2P_array, traj1P_array)

    min_dists_traj2_to_traj1 = np.min(dist_matrix_traj2_to_traj1, axis=1)
    

    sequence_dis = np.minimum(min_dists_traj1_to_traj2, min_dists_traj2_to_traj1)
    
    return np.array(sequence_dis)    
 

def angle_difference(a, b):

    a = (a + 180) % 360 - 180
    b = (b + 180) % 360 - 180

    diff = b - a

    diff = 1 - abs((diff + 180) % 360 - 180)/180
    return diff
def calculate_direction_angle(x1, y1, x2, y2):
    angle_radians = math.atan2(y2 - y1, x2 - x1)

    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def compute_distances(df):
    rows_list = []  # List to store row dictionaries

    for inter_id, inter_data in tqdm(df.groupby('inter_id')):
        trajectories = inter_data['trajectory_id'].unique()
        for i in range(len(trajectories)):
            for j in range(i+1, len(trajectories)):
                traj1 = inter_data[inter_data['trajectory_id'] == trajectories[i]][['x_proj', 'y_proj']].values
                traj2 = inter_data[inter_data['trajectory_id'] == trajectories[j]][['x_proj', 'y_proj']].values
#
                hsdf_dis = hausdorff(traj1, traj2)
                
                # Calculate start and end distances
                start_dis = np.linalg.norm(traj1[0] - traj2[0])
                end_dis = np.linalg.norm(traj1[-1] - traj2[-1])
#                 extreme_dis=start_dis+end_dis
#                 frechet_dis = calculate_frechet_distance(resample_trajectory(traj1, 10), resample_trajectory(traj2, 10))
                sequence_dis_list=calculate_sequence_dis(traj1, traj2, 10)
#                 print(sequence_dis_list)
                row_data = {'inter_id': inter_id, 'traj1': trajectories[i], 'traj2': trajectories[j], 
                                  'hsdf_dis': hsdf_dis}
                for index, dis in enumerate(sequence_dis_list):
                    row_data[f'dis{index}'] = dis
                row_data['dis0']=start_dis
                row_data['dis9']=end_dis
                rows_list.append(row_data)

    # Create df_edge DataFrame from the list of row dictionaries
    df_edge = pd.DataFrame(rows_list)
    return df_edge
def normalize_trajectory_data(df):


    def normalize_group(group):

        max_range = max(group['x_proj'].max() - group['x_proj'].min(), group['y_proj'].max() - group['y_proj'].min())
        if max_range > 0:  # 防止除以0
            group['x_norm'] = (group['x_proj'] - group['x_proj'].min())/ (group['x_proj'].max() - group['x_proj'].min())
            group['y_norm'] = (group['y_proj'] -  group['y_proj'].min())/(group['y_proj'].max() - group['y_proj'].min())
        else:
            group['x_norm'] = 0
            group['y_norm'] = 0
        return group

    normalized_df = df.groupby('inter_id', group_keys=False).apply(normalize_group)

    return normalized_df

def hausdorff(u, v):
    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d

def chamfer_distance(u, v):
    u=np.array(u, dtype=float)
    v=np.array(v, dtype=float)

    dist_matrix_uv = cdist(u, v, 'euclidean')
    min_dist_uv = np.min(dist_matrix_uv, axis=1)


    dist_matrix_vu = cdist(v, u, 'euclidean')
    min_dist_vu = np.min(dist_matrix_vu, axis=1)


    d = (np.sum(min_dist_uv) + np.sum(min_dist_vu)) / (len(u) + len(v))
    return d

def chamfer_Matrix(trajs1, trajs2):

    D_chamfer = np.zeros((len(trajs1), len(trajs2)))

    for i in range(len(trajs1)):
        for j in range(len(trajs2)):
            distance = chamfer_distance(trajs1[i], trajs2[j])
            D_chamfer[i, j] = distance
    return D_chamfer
def hausdorffMatrix(trajs1, trajs2):

    D_hsdf = np.zeros((len(trajs1), len(trajs2)))

    for i in range(len(trajs1)):
        for j in range(len(trajs2)):
            distance = hausdorff(trajs1[i], trajs2[j])
            D_hsdf[i, j] = distance
    return D_hsdf

def convert_multilinestring_to_linestring(geometry):

    if isinstance(geometry, MultiLineString):

        if len(geometry.geoms) > 0:
            return geometry.geoms[0]

    return geometry

def df_coordinate_projection(df):

        df=df.copy()
        transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
        df['point'] = df.apply(lambda x: transformer.transform(x.y, x.x), axis=1)
        df['x_proj'] = df['point'].apply(lambda x: x[0])
        df['y_proj'] = df['point'].apply(lambda x: x[1])

        # Group the dataframe by groupID
        grouped = df.groupby('trajectory_id')

        # Resample trajectories
        trajs = []
        for groupID, group in grouped:
            # Initialize resampled trajectory
            resampled_traj = [group.iloc[0][['x_proj', 'y_proj']].values]
#             current_time = group.iloc[0]['timestamp']

            # Iterate over points in the group
            for i in range(1, len(group)):
                resampled_traj.append(group.iloc[i][['x_proj', 'y_proj']].values)
            trajs.append(np.array(resampled_traj))
        return trajs
def line_coordinate_projection(gdf):

    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    

    trajs = []
    

    for geometry in gdf.geometry:

        x_proj, y_proj = transformer.transform(*zip(*geometry.coords))
        

        traj = np.column_stack((x_proj, y_proj))
        trajs.append(traj)
    
    return trajs

def resample_lines(gdf, d):
    """
    Resample the lines in the given GeoDataFrame (gdf) such that the distance between any two
    consecutive points in each line is at most d meters. This is achieved by adding additional
    points along each line as necessary.

    Parameters:
    - gdf: GeoDataFrame with a 'geometry' column containing LineString objects.
    - d: The maximum allowed distance between consecutive points along the line, in degrees.

    Returns:
    - A new GeoDataFrame with the resampled LineStrings and other original attributes.
    """
    resampled_data = []

    for _, row in gdf.iterrows():
        original_line = row['geometry']
        resampled_points = [original_line.coords[0]]  # Start with the first point
        
        # Iterate over line segments in the original line
        for start, end in zip(original_line.coords[:-1], original_line.coords[1:]):
            start_point = Point(start)
            end_point = Point(end)
            segment_length = start_point.distance(end_point)
            
            if segment_length > d:
                # Calculate how many new points to add
                num_new_points = int(segment_length // d)
                # Calculate the vector for the line segment
                dx = (end[0] - start[0]) / segment_length
                dy = (end[1] - start[1]) / segment_length
                
                # Add new points along the segment
                for i in range(1, num_new_points + 1):
                    new_x = start[0] + dx * d * i
                    new_y = start[1] + dy * d * i
                    resampled_points.append((new_x, new_y))
            
            # Add the end point of the current segment
            resampled_points.append(end)
        
        # Create a new LineString with the resampled points
        resampled_line = LineString(resampled_points)
        
        # Add the new line and other attributes to the data list
        resampled_data.append({**row, 'geometry': resampled_line})
    
    # Create a new GeoDataFrame with the resampled lines and other original attributes
    resampled_gdf = gpd.GeoDataFrame(resampled_data)
    
    return resampled_gdf
def trans_crs(df):

    df=df.copy()
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
    df['point'] = df.apply(lambda x: transformer.transform(x.y, x.x), axis=1)
    df['x_proj'] = df['point'].apply(lambda x: x[0])
    df['y_proj'] = df['point'].apply(lambda x: x[1])
    df=df.drop(columns=['point'])
    return df

def frechetMatrix(trajs1, trajs2):
    D_frechet = np.zeros((len(trajs1), len(trajs2)))
    trajs1=[resample_trajectory(item.astype(float), 20) for item in trajs1]
    trajs2=[resample_trajectory(item.astype(float), 20) for item in trajs2]
    for i in range(len(trajs1)):
        for j in range(len(trajs2)):
            distance = calculate_frechet_distance(trajs1[i], trajs2[j])
            D_frechet[i, j] = distance
    return D_frechet
    
import pandas as pd
import numpy as np
from io import StringIO
import math
def adjust_angle_diff(diff):

    if diff > 180:
        return 360 - diff
    elif diff < -180:
        return 360 + diff
    return diff
def calculate_sin_cos_components(x1, y1, x2, y2):
    # Δx 和 Δy
    dx = x2 - x1
    dy = y2 - y1
    

    distance = math.sqrt(dx**2 + dy**2)
    

    cos_theta = dx / distance
    sin_theta = dy / distance
    
    return sin_theta, cos_theta
def get_node_df(traj_norm):
    if isinstance(traj_norm, str):
        traj_norm = pd.read_csv(StringIO(traj_norm), sep="\t")
    
    results = []
    for tid, group in tqdm(traj_norm.groupby('trajectory_id')):
        group = group.sort_values(by=['timestamp']).reset_index(drop=True)
        

        group['dx'] = group['x_norm'].diff(-1).fillna(0).abs()
        group['dy'] = group['y_norm'].diff(-1).fillna(0).abs()
        group['distance'] = np.sqrt(group['dx'] ** 2 + group['dy'] ** 2)
        group['azimuth_diff'] = group['azimuth'].diff().fillna(0)
        group['azimuth_diff'] = group['azimuth_diff'].apply(adjust_angle_diff)

        total_distance = group['distance'].iloc[:-1].sum()  # 排除最后一个NaN距离
        total_angle_change = np.radians(group['azimuth_diff']).sum()
        curvature = total_angle_change / total_distance if total_distance != 0 else 0
        
        

        line_distance = np.sqrt((group['x_norm'].iloc[-1] - group['x_norm'].iloc[0]) ** 2 +
                                (group['y_norm'].iloc[-1] - group['y_norm'].iloc[0]) ** 2)
        
        # Tortuosity
        tortuosity = total_distance / line_distance if line_distance != 0 else 0
        

        angle_change = group['azimuth'].iloc[-1] - group['azimuth'].iloc[0]
        angle_change_sin = np.sin(np.radians(angle_change))
        angle_change_cos = np.cos(np.radians(angle_change))
        angle_start_sin=np.sin(group['azimuth'].iloc[0])
        angle_start_cos=np.cos(group['azimuth'].iloc[0])
        angle_end_sin=np.sin(group['azimuth'].iloc[-1])
        angle_end_cos=np.cos(group['azimuth'].iloc[-1])
        direct_sin, direct_cos=calculate_sin_cos_components(group['x_norm'].iloc[0], group['y_norm'].iloc[0], 
                                                            group['x_norm'].iloc[-1], group['y_norm'].iloc[-1])

        target_distances = [total_distance * i / 4 for i in range(1, 4)]
        cumulative_distance = 0
        indices = [0]
        for i, row in group.iterrows():
            cumulative_distance += row['distance']
            while len(indices) < 4 and cumulative_distance >= target_distances[len(indices) - 1]:
                indices.append(i)
        indices.append(len(group) - 1)
        
        points = group.iloc[indices]
        coords = points[['x_norm', 'y_norm']].values.flatten()
        columns = ['startX', 'startY', 'secondX', 'secondY', 'thirdX', 'thirdY', 'forthX', 'forthY', 'endX', 'endY']
        
        record = {
            'traj_id': tid, 
            'inter_id': group['inter_id'].iloc[0],
            'tortuosity': tortuosity,
            'angle_change_sin': angle_change_sin,
            'angle_change_cos': angle_change_cos,
            'angle_start_sin': angle_start_sin,
            'angle_start_cos': angle_start_cos,
            'angle_end_sin': angle_end_sin,
            'angle_end_cos': angle_end_cos,
            'curvature': curvature,
            'direct_sin':direct_sin,
            'direct_cos':direct_cos,
            'startX':coords[0],
            'startY':coords[1],
            'endX':coords[-2],
            'endY':coords[-1]
        }
#         record.update(dict(zip(columns, coords)))
        results.append(record)
    
    node_df = pd.DataFrame(results)
    node_df.sort_values(by=['inter_id', 'traj_id'], inplace=True)
    
    return node_df
def node_norm(df):

    df=df.copy()
    tortuosity_mean = df['tortuosity'].mean()
    tortuosity_std = df['tortuosity'].std()
    curvature_mean = df['curvature'].mean()
    curvature_std = df['curvature'].std()
    

    df['tortuosity'] = (df['tortuosity'] - tortuosity_mean) / tortuosity_std
    df['curvature'] = (df['curvature'] - curvature_mean) / curvature_std
    
    return df

def node_augment(df):

    augmented_data = df


    for inter_id in df['inter_id'].unique():
        sub_df = df[df['inter_id'] == inter_id]
        

        center_x = np.mean([sub_df[col].mean() for col in ['startX', 'secondX', 'thirdX', 'forthX', 'endX']])
        center_y = np.mean([sub_df[col].mean() for col in ['startY', 'secondY', 'thirdY', 'forthY', 'endY']])


        for angle, increment in zip([90, 180, 270], [200, 400, 600]):

            rad = np.deg2rad(angle)
            cos_angle, sin_angle = np.cos(rad), np.sin(rad)


            new_df = sub_df.copy()
            for col_x, col_y in [('startX', 'startY'), ('secondX', 'secondY'), ('thirdX', 'thirdY'), ('forthX', 'forthY'), ('endX', 'endY')]:
                new_df[col_x] = cos_angle * (sub_df[col_x] - center_x) - sin_angle * (sub_df[col_y] - center_y) + center_x
                new_df[col_y] = sin_angle * (sub_df[col_x] - center_x) + cos_angle * (sub_df[col_y] - center_y) + center_y

            new_df['inter_id'] += increment  # Update inter_id for each rotation
            new_df['traj_id'] += increment/1000


            augmented_data = pd.concat([augmented_data, new_df], ignore_index=True)


    return augmented_data

def edge_augment(df):

    df_200 = df.copy()
    df_400 = df.copy()
    df_600 = df.copy()

    df_200['inter_id'] += 200
    df_200['traj1'] += 0.2
    df_200['traj2'] += 0.2
    df_400['inter_id'] += 400
    df_400['traj1'] += 0.4
    df_400['traj2'] += 0.4
    df_600['inter_id'] += 600
    df_600['traj1'] += 0.6
    df_600['traj2'] += 0.6
    

    result_df = pd.concat([df, df_200, df_400, df_600], ignore_index=True)
    
    return result_df

def get_noise_index(D, noise_rate,seed):
    np.random.seed(seed)

    if not isinstance(D, np.ndarray):
        D = np.array(D)
    

    min_indices = np.argmin(D, axis=0)
    

    selected_indices = set(min_indices)
    

    all_indices = set(range(D.shape[0]))
    

    remaining_indices = list(all_indices - selected_indices)
    num_to_select = int(np.ceil(noise_rate * len(remaining_indices)))
    noise_indices = np.random.choice(remaining_indices, num_to_select, replace=False)
    

    selected_indices.update(noise_indices)
    

    return sorted(list(selected_indices))
def get_subgraph_index(D_list,traj_grouped,train_inter_ids,noise_rate,seed):
    selected_traj_ids_list=[]
    index_lists=[]
    for i in range(0,len(train_inter_ids)):
        inter_id=train_inter_ids[i]
        inter_traj=traj_grouped[traj_grouped['inter_id']==inter_id]
        traj_ids = np.sort(inter_traj['trajectory_id'].unique())
        D=D_list[i]
        index_list=get_noise_index(D, noise_rate,seed)
        selected_traj_ids = traj_ids[index_list]
        index_lists.append(index_list)
        selected_traj_ids_list.append(selected_traj_ids )
    selected_traj_ids_list =np.concatenate(selected_traj_ids_list)
    return index_lists,selected_traj_ids_list 