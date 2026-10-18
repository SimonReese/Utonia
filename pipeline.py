import os
from pathlib import Path
from typing import Tuple, Union

from sklearn.cluster import DBSCAN
import trimesh
import numpy as np
import open3d as o3d
from trimesh import Scene
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import utonia
import torch
import torch.nn as nn

from fast_pytorch_kmeans import KMeans
from utonia.model import PointTransformerV3
from utonia.structure import Point
try:  
    import flash_attn   # https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu128torch2.11-cp311-cp311-linux_x86_64.whl
except ImportError:  
    flash_attn = None
print(flash_attn)

# ------------ TORCH PARAMETERS ----------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------ PATHS ---------------------------------------------------
POINTCLOUD_DATASETS_FOLDER_PATH = "data-generation/pointclouds"
POINTCLOUD_DATASET_NAME = "arrange-train"

POINTCLOUD_EPISODE_SUBFOLDER = "da3"
POINTCLOUD_FILENAME = "scene.glb"

EMBEDDINGS_SUBFOLDER = "embeds"
EMBEDDINGS_FILENAME = "spatial_pos_embeddings_single.npy"

DEBUG_DATA_FOLDER = "pipeline-temp-data"

# ------------ PIPELINE PARAMETERS -------------------------------------
# TABLE REMOVAL
TABLE_REMOVAL_ITERATIONS = 4
COLOR_BIN_SIZE = 48
# ------------
# Open3D NORMAL ESTIMATION
O3D_NORMAL_RADIUS = 0.1
O3D_NORMAL_MAX_NN = 30
# ------------
# UTONIA PARAMETERS
UTONIA_SCALE_FIRST_PASS = 42
UTONIA_SCALE_SECOND_PASS = 10
UTONIA_LAYER_CONCATENATION = 4
"""Utonia has 4 layers of features. Usually just the last 2 layers are concatenated, but we will go up to 4"""
# --------------
# SPATIAL EMBEDDINGS
SPATIAL_BIN_SIZE = (3, 3, 3)
# ----------------------------------------------------------------------

def main():
    # Load utonia model
    utonia_model = load_utonia()

    # Open dataset
    dataset_path = os.path.join(POINTCLOUD_DATASETS_FOLDER_PATH, POINTCLOUD_DATASET_NAME)
    tasks_list = get_tasks(dataset_path)
    for task in tasks_list:
        print(f"Opening task {task}")
        task_path = os.path.join(dataset_path, task)
        variation_ids = get_variations(task_path)
        for variation in variation_ids:
            print(f"Opening variation {variation}")
            variation_path = os.path.join(task_path, f"variation{variation}")
            episode_ids = get_episodes(variation_path)
            for episode in episode_ids:
                print(f"Opening episode {episode}")
                episode_path = os.path.join(variation_path, f"episode{episode}")
                print(episode_path)
                
                # Open scene.glb
                pcd_path = os.path.join(episode_path, POINTCLOUD_EPISODE_SUBFOLDER, POINTCLOUD_FILENAME)
                assert os.path.exists(pcd_path)
                
                # DEBUG
                # ep_folder = Path(pcd_path).parents[1]
                # if (os.path.exists(f"pipeline-temp-data/{ep_folder}/cleaned.png") and
                #     os.path.exists(f"pipeline-temp-data/{ep_folder}/clusters.png") and
                #     os.path.exists(f"pipeline-temp-data/{ep_folder}/filtered-clusters.png")):
                #     print(f"Found {task}/{variation}/episode{episode}")
                #     continue

                spatial_pos_embeddings = pipeline(utonia_model, pcd_path)
                
                if True:
                    # Save output
                    dst = os.path.join(episode_path, EMBEDDINGS_SUBFOLDER)
                    os.makedirs(dst, exist_ok=True)
                    dst_file = os.path.join(dst, EMBEDDINGS_FILENAME)
                    
                    # Cleanup
                    if os.path.exists(f"{dst}/clusters.png"): 
                        print(f"{txtcolors.FAIL}Deleting {txtcolors.ENDC}{dst}/clusters.png")
                        os.remove(f"{dst}/clusters.png")
                    if os.path.exists(f"{dst}/spatial_pos_embeddings.npy"): 
                        print(f"{txtcolors.FAIL}Deleting {txtcolors.ENDC}{dst}/spatial_pos_embeddings.npy")
                        os.remove(f"{dst}/spatial_pos_embeddings.npy")
                    
                    if os.path.exists(dst_file): print(f"{txtcolors.WARNING}Overwriting {txtcolors.ENDC}{dst_file}")
                    np.save(dst_file, spatial_pos_embeddings)

def pipeline(utonia_model, pcd_path: str) -> np.ndarray: #Tuple[np.ndarray, bytes]:
    
    # Construct debug visualization data
    EPISODE_FOLDER_SUBPATH = Path(pcd_path).parents[1]
    os.makedirs(DEBUG_EP_FOLDER_PATH := os.path.join(DEBUG_DATA_FOLDER, EPISODE_FOLDER_SUBPATH), exist_ok=True)

    # Load and clean pcd
    scene = load_pointcloud(pcd_path)
    cleaned = remove_table(scene)
    # Convert pcd to dict following utonia convention
    input = pcd_to_utonia_dict(cleaned)
    # Extract colors and coords
    original_coords: np.ndarray = input["coord"].copy() # type: ignore
    original_colors: np.ndarray = input["color"].copy() # type: ignore
    del input, scene, cleaned

    # Remove sparse points
    cleaned_coords, cleaned_colors, _ = remove_radius_outliers(original_coords, original_colors, points_nb=60, radius=0.03)
    cleaned_coords, cleaned_colors, _ = remove_radius_outliers(cleaned_coords, cleaned_colors, points_nb=100, radius=0.03)
    cleaned_coords, cleaned_colors, _ = remove_sparse_points(cleaned_coords, cleaned_colors, nb_neighbors=400)
    
    # convert to trimesh and save scene representation
    scene = numpy_to_trimesh(cleaned_coords, cleaned_colors/255.0)
    save_trimesh_image(scene, DEBUG_EP_FOLDER_PATH, "cleaned.png")                          # <-------- CLEANED 

    # ------- Clustering ----------
    # Cluster using spatial coordinates with Density-Based Spatial Clustering 
    original_labels = cluster_spatial_dbscan(coords=cleaned_coords, eps=0.03, min_points=400)
    # Save scene in image
    scene = visualize_cluster(cleaned_coords, original_labels, K_clusters=None)
    save_trimesh_image(scene, DEBUG_EP_FOLDER_PATH, "clusters.png")                         # <-------- CLUSTERS

    # Keep only 3 cluster for the 3 objects
    top3_mask, top3_cluster_ids = keep_top_k_clusters(cleaned_coords, original_labels, K=3)
    final_coords = cleaned_coords[top3_mask]
    final_colors = cleaned_colors[top3_mask]
    final_labels = original_labels[top3_mask]
    scene = visualize_cluster(final_coords, final_labels, K_clusters=None)
    save_trimesh_image(scene, DEBUG_EP_FOLDER_PATH, "filtered-clusters.png")                # <-------- FILTERED CLUSTERS
    
    # -------- Utonia passage ------
    # Construct final pointcloud with real colors
    scene = numpy_to_trimesh(final_coords, final_colors/255.0)
    # Prepare for utonia
    input = pcd_to_utonia_dict(scene)
    # Transform input
    transform = utonia.transform.default(scale=UTONIA_SCALE_SECOND_PASS)
    input: dict[str, torch.Tensor] = transform(input)
    input = dict_to_cuda(input)
    # Inference
    output_point = utonia_inference(input, utonia_model, depth=UTONIA_LAYER_CONCATENATION)
    embeds = mean_embeddings(
        utonia_output=output_point,
        cluster_labels=final_labels,
        K_clusters=len(top3_cluster_ids)
    )
    print(f"{txtcolors.OKGREEN}{embeds.shape}{txtcolors.ENDC}")
    return embeds
    
# ------------ FILE LOADING ---------------------------------------------
def load_pointcloud(pcd_path) -> trimesh.Scene:
    input_scene: trimesh.Scene = trimesh.load_scene(pcd_path)
    return input_scene

# ------------ UTONIA INFERENCE ----------------------------------------
def load_utonia() -> PointTransformerV3:
    if flash_attn is not None:  
        model = utonia.load("utonia", repo_id="Pointcept/Utonia").to(device)
        print(f"{txtcolors.OKGREEN}Loaded Utonia with flash attention{txtcolors.ENDC}") 
    else:
        print(f"{txtcolors.WARNING}Warning: Utonia loaded without flash attention{txtcolors.ENDC}: Loading with config: enc_patch_size=[1024]*5, enable_flash=False")
        custom_config = dict(enc_patch_size=[1024]*5, enable_flash=False)  
        model = utonia.load("utonia", repo_id="Pointcept/Utonia", custom_config=custom_config).to(device)  
    model.eval()
    return model

def utonia_inference(transformed_input: dict[str, torch.Tensor], utonia_model: PointTransformerV3, depth = 4) -> Point:
    with torch.inference_mode():
        # Inference step
        output_point: utonia.structure.Point = utonia_model(transformed_input)  
  
        # First we upcast multi-scale features
        for _ in range(UTONIA_LAYER_CONCATENATION):
            # For each Point, we extract the previous Point struct, and concatenate the parent.feat with the child.feat
            parent  = output_point.pop("pooling_parent")  
            inverse = output_point.pop("pooling_inverse")  
            parent.feat = torch.cat([parent.feat, output_point.feat[inverse]], dim=-1)  
            output_point = parent  
        # Now output_point is the grid scaled (if range(4))
        # However if we just extracted the last two layers range(2), we still need to reach the stage 0 layer and propagate to finest pointclod
        # Eventually propagation 
        while "pooling_parent" in output_point.keys():  # if range(4) no more pooling_parent are present in the dict
            parent  = output_point.pop("pooling_parent")  
            inverse = output_point.pop("pooling_inverse")  
            parent.feat = output_point.feat[inverse]  
            output_point = parent
    return output_point

# ------------ CLUSTERING ----------------------------------------------
def cluster_points(utonia_output: Point , K_cluster = 3) -> np.ndarray:
    """Perfoms clustering among points using utonia features and returns the label for each original point"""
    # Feature normalization 
    feat = utonia_output.feat  # (N_grid, D)  
    feat_norm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
    kmeans = KMeans(n_clusters=K_cluster, mode="euclidean", verbose=1)
    # Predict label for each N_grid point
    labels = kmeans.fit_predict(feat_norm)  # (N_grid,) int tensor  

    # Map cluster to original colors 
    labels_np = labels.cpu().numpy() 
    #cmap = plt.get_cmap("tab20", K)  
    #cluster_colors = np.array([cmap(i)[:3] for i in range(K)], dtype=np.float32)  # (K, 3)  
    # Assign color to each N_grid point
    #grid_colors = cluster_colors[labels_np]  # (N_grid, 3)  
    
    # Propagate to original scale using point.inverse
    #original_cluster_colors = grid_colors[utonia_output.inverse.cpu().numpy()]

    points_labels = labels_np[utonia_output.inverse.cpu().numpy()]  # (N_original,)
    return points_labels 

def cluster_points_DBSCAN(utonia_output: Point, eps: float = 0.05, min_samples: int = 15) -> np.ndarray:
    """
    Esegue il clustering DBSCAN sulle feature estratte da Utonia.
    Restituisce i label per i punti originali (-1 = rumore).

    Params
    --------
    :utonia_output: utonia output features
    :eps: cosine distance threshold for same cluster. Increase to enlarge cluster
    :min_samples: minumum points to  iindetify a cluster
    """
    # 1. Normalizzazione delle feature (fondamentale per la distanza coseno)
    feat = utonia_output.feat  # (N_grid, D)  
    feat_norm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
    
    # Passaggio a CPU/Numpy per scikit-learn
    feat_np = feat_norm.cpu().numpy()

    # 2. Esecuzione DBSCAN
    # n_jobs=-1 usa tutti i core della CPU per velocizzare il calcolo
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    
    print("Esecuzione DBSCAN in corso sulle feature...")
    labels_np = dbscan.fit_predict(feat_np)  # (N_grid,) int array (-1 per il rumore)
    
    # 3. Propagazione ai punti originali tramite punto.inverse
    points_labels = labels_np[utonia_output.inverse.cpu().numpy()]  # (N_original,)
    
    n_clusters = len(set(labels_np)) - (1 if -1 in labels_np else 0)
    print(f"DBSCAN ha trovato {n_clusters} cluster e {(labels_np == -1).sum()} punti di rumore (su griglia).")
    
    return points_labels

def cluster_spatial_dbscan(coords: np.ndarray, eps: float = 0.03, min_points: int = 20) -> np.ndarray:
    """
    Esegue il clustering spaziale DBSCAN direttamente sulle coordinate 3D.
    Restituisce un array di label di dimensione (N_original,).
    I punti di rumore avranno label -1.
    """
    # Inizializza Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    
    # Clustering spaziale (molto più veloce e leggero di scikit-learn)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    # Calcola il numero di cluster trovati (ignorando il rumore -1)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Clustering Spaziale: trovati {num_clusters} oggetti fisici.")
    
    return labels

def remove_largest_cluster(coords, colors, labels) -> tuple[np.ndarray, np.ndarray]:
    """Removes largest cluster spanning in xy area, returning the refined pointcloud and colors"""
    mask, removed = remove_largest_xy_surface_cluster(coords, labels)
    filtered_coord  = coords[mask]  
    filtered_colors = colors[mask]  
    return filtered_coord, filtered_colors

def remove_largest_xy_surface_cluster(coord, labels_np) -> tuple[np.ndarray, np.ndarray]:  
    """  
    Removes the biggest surfcace cluster over XY (AABB 2D).  
      
    Args:  
        coord:     array (N, 3) orgiinal coords 
        labels_np: array (N,) of clusters labels (numpy int)  
      
    Returns:  
        mask: array booleano (N,) True = points to keep   
        largest_cluster: removed cluster id  
    """  
    unique = np.unique(labels_np)  
      
    areas = {}  
    for c in unique:  
        pts = coord[labels_np == c]  
        x_range = pts[:, 0].max() - pts[:, 0].min()  
        y_range = pts[:, 1].max() - pts[:, 1].min()  
        areas[c] = x_range * y_range  
      
    largest_cluster = max(areas, key=areas.get)  
    mask = labels_np != largest_cluster  
      
    return mask, largest_cluster

def filter_cluster_noise_dbscan(coords: np.ndarray, labels: np.ndarray, eps: float = 0.03, min_points: int = 15) -> np.ndarray:
    """
    Applica DBSCAN spaziale per ogni cluster K-Means e mantiene 
    solo la componente densa principale di ciascun oggetto.
    
    eps: distanza massima tra due punti per essere considerati vicini (in metri/unità coordinate)
    min_points: numero minimo di punti per formare una componente densa
    """
    clean_mask = np.zeros(len(coords), dtype=bool)
    
    for k in np.unique(labels):
        mask_k = (labels == k)
        pts_k = coords[mask_k]
        indices_k = np.where(mask_k)[0]
        
        if len(pts_k) == 0:
            continue
            
        # DBSCAN open3d sulle coordinate 3D del singolo cluster
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_k)
        db_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        
        # Trova il sotto-cluster con il maggior numero di punti (escludendo il rumore -1)
        valid_labels = db_labels[db_labels != -1]
        if len(valid_labels) > 0:
            largest_sub_cluster = np.bincount(valid_labels).argmax()
            # Seleziona solo i punti della componente principale
            keep_in_k = (db_labels == largest_sub_cluster)
            clean_mask[indices_k[keep_in_k]] = True

    return clean_mask

def keep_top_k_clusters(coords: np.ndarray, labels: np.ndarray, K: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifica e mantiene solo i K cluster con il maggior numero di punti.
    
    Returns:
        valid_mask: Maschera booleana (N,) con True per i punti dei top K cluster.
        top_k_labels: Lista degli ID dei K cluster selezionati.
    """
    # Considera solo i label validi (esclude il rumore -1)
    valid_points_labels = labels[labels != -1]
    
    if len(valid_points_labels) == 0:
        print("Nessun cluster valido trovato!")
        return np.zeros(len(coords), dtype=bool), np.array([])

    # Calcola la frequenza di ciascun cluster
    unique_labels, counts = np.unique(valid_points_labels, return_counts=True)
    
    # Ordina i cluster per numero di punti in ordine decrescente
    sorted_indices = np.argsort(-counts)
    top_k_labels = unique_labels[sorted_indices[:K]]
    
    print(f"Cluster totali: {len(unique_labels)}. Mantengo i top {len(top_k_labels)} con punti: {counts[sorted_indices[:K]]}")
    
    # Maschera booleana dei soli punti appartenenti ai Top K cluster
    valid_mask = np.isin(labels, top_k_labels)
    
    return valid_mask, top_k_labels

def clean_post_clustering(
    coords: np.ndarray, 
    labels: np.ndarray, 
    eps: float = 0.03, 
    min_points: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filtra i cluster dopo il K-Means:
    1. Isola ogni cluster ed esegue un DBSCAN spaziale 3D.
    2. Mantiene solo la componente connessa più grande per ogni cluster.
    3. Scarta i frammenti di rumore ed eventuali sbavature ai bordi.

    Args:
        coords: (N, 3) Coordinate 3D dei punti.
        labels: (N,) Label assegnati dal K-Means.
        eps: Distanza massima (in unità/metri) per considerare due punti vicini.
        min_points: Numero minimo di punti per formare un sotto-cluster valido.

    Returns:
        valid_mask: Array booleano (N,) con True per i punti appartenenti agli oggetti puliti.
        cleaned_labels: Array (N,) con i label filtrati.
    """
    valid_mask = np.zeros(len(coords), dtype=bool)
    cleaned_labels = labels.copy()
    
    for k in np.unique(labels):
        # 1. Estrai solo i punti appartenenti al cluster k
        cluster_mask = (labels == k)
        cluster_indices = np.where(cluster_mask)[0]
        pts_k = coords[cluster_mask]
        
        if len(pts_k) == 0:
            continue
            
        # 2. Crea PointCloud Open3D locale
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_k)
        
        # 3. DBSCAN sulle sole coordinate spaziali del cluster k
        db_labels = np.array(
            pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
        )
        
        # 4. Individua il sotto-cluster con il maggior numero di punti (escludendo il rumore -1)
        valid_db_labels = db_labels[db_labels != -1]
        if len(valid_db_labels) > 0:
            largest_sub_cluster_id = np.bincount(valid_db_labels).argmax()
            
            # Maschera dei punti che appartengono al corpo principale dell'oggetto
            main_body_mask = (db_labels == largest_sub_cluster_id)
            
            # Salva i punti validi nella maschera globale
            valid_mask[cluster_indices[main_body_mask]] = True

    return valid_mask, cleaned_labels

# ------------ UTONIA TRANSFORMATIONS ----------------------------------
def pcd_to_utonia_dict(scene: trimesh.Scene) -> dict:
    """Converts trimesh scene pointcloud to utonia dictionary"""
    coords, colors = scene_to_ndarr(scene)
    normals = estimate_normals(coords)
    output = dict(
        coord=coords, 
        color=colors, 
        normal=normals
    )
    #print(f"{txtcolors.WARNING}showing pcd{txtcolors.ENDC}")
    #numpy_to_trimesh(coords, colors/255).show(line_settings={'point_size': 1.0})
    return output

# ------------ SPATIAL EMBEDDING ----------------------------------------
def extract_spatial_embeddings(utonia_output: Point, coord, labels: np.ndarray, K_clusters = 3) -> np.ndarray:
    feat: torch.Tensor = utonia_output.feat[utonia_output.inverse]
    labels = labels[utonia_output.inverse.cpu().numpy()]
    spatial_embeddings = compute_spatial_bin_embeddings(  
        feat=feat,  
        coord=coord,  
        labels=labels,  
        K=K_clusters,  
        bins=SPATIAL_BIN_SIZE  # 27 bin per cluster → 27*D elements  
    )

    emb_matrix = np.stack([spatial_embeddings[k] for k in range(K_clusters)])  # (K, 27*D)
    emb_matrix = emb_matrix.reshape((K_clusters, np.prod(SPATIAL_BIN_SIZE), -1))
    return emb_matrix

def compute_spatial_bin_embeddings(feat, coord, labels, K, bins=(3, 3, 3)) -> dict[int, np.ndarray]:  
    """  
    Compute spatial bin embeddings for each cluster.  
      
    Args:  
        feat:   (N, D) tensor GPU - features of ech point  
        coord:  (N, 3) numpy array - original coords  
        labels: (N,) numpy array int - cluster label for point  
        K:      number of cluster  
        bins:   grid for cluster  
    
    Returns:  
        embeddings: dict {cluster_id: np.array (n_bins * D,)}  
    """  
    feat_np = feat.cpu().detach().numpy()  # (N, D)  
    n_bins = bins[0] * bins[1] * bins[2]  
    D = feat_np.shape[-1] # 1386 for range(4) and hf checkpoint
    
    embeddings = {}  
    
    for k in range(K): 
        mask = labels == k  
        pts = coord[mask]       # (M, 3)  
        f   = feat_np[mask]     # (M, D)  
          
        if len(pts) == 0:  
            embeddings[k] = np.zeros(n_bins * D, dtype=np.float32)  
            continue  
          
        # Normalize coords in cluster bounding box  
        mins   = pts.min(axis=0)  
        maxs   = pts.max(axis=0)  
        ranges = np.maximum(maxs - mins, 1e-6)  
          
        # Bin index for point  
        idx = np.floor((pts - mins) / ranges * np.array(bins)).astype(int)  
        idx = np.clip(idx, 0, np.array(bins) - 1)  
        bin_idx = idx[:, 0] * bins[1] * bins[2] + idx[:, 1] * bins[2] + idx[:, 2]  
          
        # Mean feature per bin  
        result = np.zeros((n_bins, D), dtype=np.float32)  
        counts = np.zeros(n_bins, dtype=np.int32)  
        np.add.at(result, bin_idx, f)  
        np.add.at(counts, bin_idx, 1)  
        counts = np.maximum(counts, 1)  
        result /= counts[:, None]  
          
        embeddings[k] = result.flatten()  # (n_bins * D,)  
      
    return embeddings

def mean_embeddings(utonia_output: Point, cluster_labels: np.ndarray, K_clusters = 3) -> np.ndarray:
    feat: torch.Tensor = utonia_output.feat[utonia_output.inverse]
    
    # Recupera gli ID reali dei cluster presenti nelle etichette
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
    
    means = []
    for cluster_id in unique_clusters[:K_clusters]:
        mask_k = (cluster_labels == cluster_id)
        if mask_k.sum() > 0:
            mean_k = feat[mask_k].mean(dim=0)
        else:
            mean_k = torch.zeros(feat.shape[-1], device=feat.device)
        means.append(mean_k)
        
    mean_embeddings = torch.stack(means)  # (K, D)
    cluster_embeddings_norm = mean_embeddings / (mean_embeddings.norm(dim=-1, keepdim=True) + 1e-6)

    return cluster_embeddings_norm.cpu().numpy()

# ------------ POSITIONAL ENCODINGS ------------------------------------
def add_positional_encodings(coords, labels, spatial_embeddings: np.ndarray, K_clusters = 3) -> np.ndarray:
    # Compute centers
    bin_centers = compute_bin_centers(coords, labels, K_clusters, bins=SPATIAL_BIN_SIZE)  
    bin_centers_tensor = torch.from_numpy(bin_centers).float().to(device)  # (K, 27, 3)
    assert len(spatial_embeddings.shape) == 3
    D = spatial_embeddings.shape[-1]
    n_bins = np.prod(SPATIAL_BIN_SIZE)
    emb_matrix = np.stack([spatial_embeddings[k] for k in range(K_clusters)])  # (K, 37422)
    # Reshape → (K, 27, D)  
    emb_tensor = torch.from_numpy(emb_matrix).float()  # (K, 37422)  
    emb_tensor = emb_tensor.view(K_clusters, int(n_bins), D)          # (K, 27, 1386)
    #print(f"Embedding matrix sizes (K_clusters, N_bins, UTONIA_FEAT_SIZE): {emb_tensor.shape}")
    projector = nn.Linear(D, 2048, bias=True)  # 1386 → 2048  

    # To gpu
    projector = projector.to(device)  
    emb_tensor = emb_tensor.to(device)  
    
    # emb_tensor: (K, 27, 1386) — spatial bin embeddings  
    # bin_centers_tensor: (K, 27, 3) — bin centers coords
    
    # concat coords → (K, 27, 1386 + 6 = 1389)  
    emb_with_pos = torch.cat([emb_tensor.to(device), bin_centers_tensor], dim=-1)  
    return emb_with_pos.cpu().numpy()

def compute_bin_centers(coord, labels, K, bins=(3, 3, 3)) -> np.ndarray:  
    """  
    Compute 3D coords of each bin center for each cluster.  
    Shape: (K, n_bins, 3)  
    """  
    n_bins = bins[0] * bins[1] * bins[2]  
    centers = np.zeros((K, n_bins, 3), dtype=np.float32)  
    
    for k in range(K):  
        mask = labels == k  
        pts = coord[mask]  
          
        mins = pts.min(axis=0)  
        maxs = pts.max(axis=0)  
        ranges = np.maximum(maxs - mins, 1e-6)  
          
        # Bin center in origina coords 
        for i in range(bins[0]):  
            for j in range(bins[1]):  
                for l in range(bins[2]):  
                    bin_id = i * bins[1] * bins[2] + j * bins[2] + l  
                    center = mins + (np.array([i, j, l]) + 0.5) / np.array(bins) * ranges  
                    centers[k, bin_id] = center  
      
    return centers  # (K, 27, 3)  

# ------------ POINTCLOUD REFINING PIPELINES ----------------------------
def remove_table(scene: trimesh.Scene) -> trimesh.Scene:
    cleaned = scene.copy()
    # For 4 times remove the mostg popular color
    for _ in range(TABLE_REMOVAL_ITERATIONS):
        for name, geom in cleaned.geometry.items():  
            if not isinstance(geom, trimesh.PointCloud): continue  
            # remove points  
            cleaned.geometry[name] = remove_most_popular_color(geom, bin_size=COLOR_BIN_SIZE)
    return cleaned

# ------------ POINTCLOUD REFINING FUNCTIONS ----------------------------
def remove_most_popular_color(cloud: trimesh.PointCloud, bin_size=16) -> trimesh.PointCloud:  
    """  
    Remove points of mos popular bin color.  
      
    bin_size: in range _(1-256)_ the color's bin size

    returns a trimesh.PointCloud with only points not of the most popular color
    """  
    colors = cloud.colors  # (n, 4) uint8 RGBA  
    # Quatization  
    colors_binned = (colors.astype(np.uint32) // bin_size)  
    # Combine 4 channels in a single for comparison
    # (shift bit to bit) -> (:, aaaa aaaa bbbb bbbb gggg gggg rrrr rrrr)  
    colors_packed = (  
        colors_binned[:, 0]  
        | (colors_binned[:, 1] << 8)  
        | (colors_binned[:, 2] << 16)  
        | (colors_binned[:, 3] << 24)  
    )
    # Get and count unique bins 
    unique_bins, counts = np.unique(colors_packed, return_counts=True)  
    # Find the most popular bin
    most_popular_bin = unique_bins[np.argmax(counts)]  
    
    # Keep only the points with different colors: mask.shape = (npoints, )
    mask = colors_packed != most_popular_bin    # e.g. (0 1 0 1 1 1 0 0 0 ...)
    return trimesh.PointCloud(  
        vertices=cloud.vertices[mask],  # keep only points with mask set to 1 (numpy binary indexing)
        colors=colors[mask],  #  keep only colors for points with mask set to 1
    )

def remove_sparse_points(coord, colors, nb_neighbors=20, std_ratio=2.0):  
    """  
    Removes statically isolated points.  
      
    nb_neighbors: how many neighbours to consider per point  
    std_ratio: sdt dev threshold (lower = more aggressive)  
    """  
    # We use open3d function
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(coord)  
    pcd.colors = o3d.utility.Vector3dVector(colors)  
    pcd_clean, ind = pcd.remove_statistical_outlier(  
        nb_neighbors=nb_neighbors,  
        std_ratio=std_ratio  
    )  
      
    return np.asarray(pcd_clean.points), np.asarray(pcd_clean.colors), ind

def remove_radius_outliers(coord, colors, points_nb, radius):
    # We use open3d function
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(coord)  
    pcd.colors = o3d.utility.Vector3dVector(colors)  
    pcd_clean, ind = pcd.remove_radius_outlier(  
        nb_points=points_nb, 
        radius=radius
    )
    
    return np.asarray(pcd_clean.points), np.asarray(pcd_clean.colors), ind

# ------------ POINTCLOUD CONVERSION FUNCTIONS --------------------------
def scene_to_ndarr(scene: trimesh.Scene) -> tuple[np.ndarray, np.ndarray]:
    """Converts a trimesh scene pointcloud into arrays of points and colors"""
    # Collect points and color from scene, discard everyting else
    coords_list  = []  
    colors_list  = []  
    for name, geom in scene.geometry.items():  
        if not isinstance(geom, trimesh.PointCloud): continue  
        coords_list.append(np.asarray(geom.vertices, dtype=np.float32))  
        if geom.colors is not None:  
            color = np.asarray(geom.colors, dtype=np.float32)[:, :3]  # from trimesh RGBA uint8 to RGB float [0,255]
        else:  color = np.zeros((len(geom.vertices), 3), dtype=np.float32)  # or no color
        assert color.max() > 1.0, f"{txtcolors.FAIL}Error, expecting scene colors to be in range [0-225]{txtcolors.ENDC}"
        colors_list.append(color)

    # concatenate all points along axis
    coord = np.concatenate(coords_list, axis=0)  
    color = np.concatenate(colors_list, axis=0)  # [0, 255]  
    return coord, color

def numpy_to_trimesh(coord: np.ndarray, colors_float: np.ndarray) -> trimesh.Scene:  
    """  
    Convert a numpy array of points in a trimesh pointcloud
    
    coord:        (N, 3) float 
    colors_float: (N, 3) float in [0, 1]  
    """  
    assert colors_float[:, :3].max() <= 1.0, f"{txtcolors.FAIL}Expecting colors to be in float [0-1]{txtcolors.ENDC}"
    colors_rgba = np.ones((len(coord), 4), dtype=np.uint8) * 255  
    colors_rgba[:, :3] = (colors_float * 255).astype(np.uint8)
    pcd = trimesh.PointCloud(vertices=coord.astype(np.float64), colors=colors_rgba)
    return trimesh.Scene([pcd])

# ------------ POINTCLOUD COMPUTING FUNCTIONS ---------------------------
def estimate_normals(coords: np.ndarray) -> np.ndarray:
    """Returns an array estimating normal vector for each point"""
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(coords)
    o3d_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=O3D_NORMAL_RADIUS, max_nn=O3D_NORMAL_MAX_NN)
    )
    normal = np.asarray(o3d_pcd.normals, dtype=np.float32)  
    return normal

# ------------ TENSOR OPERATIONS ----------------------------------------
def dict_to_cuda(utonia_dict: dict):
    """Converts to cuda memory if possible"""
    # Convert each input tensor to cuda memory 
    for key in utonia_dict.keys():  
        if isinstance(utonia_dict[key], torch.Tensor) and device == "cuda":  
            utonia_dict[key] = utonia_dict[key].cuda(non_blocking=True)
    return utonia_dict

# ------------ DATASET TRAVERSAL ----------------------------------------
def get_tasks(dataset_path: str) -> list:
    assert os.path.exists(dataset_path)
    tasks = os.listdir(dataset_path)
    tasks.sort()
    return tasks

def get_variations(task_path: str) -> list:
    assert os.path.exists(task_path)
    folders = os.listdir(task_path)
    ids = []
    for folder in folders:
        if "variation" not in folder: continue
        ids.append(int(folder.replace("variation", "")))
    ids.sort()
    return ids

def get_episodes(variation_path: str) -> list:
    assert os.path.exists(variation_path)
    folders = os.listdir(variation_path)
    ids = []
    for folder in folders:
        if "episode" not in folder: continue
        ids.append(int(folder.replace("episode", "")))
    ids.sort()
    return ids

# ------------ Utility --------------------------------------------------
def visualize_cluster(coords: np.ndarray, labels: np.ndarray, K_clusters: Union[int, None] = None) -> trimesh.Scene:
    """
    Mappa qualsiasi numero di cluster (anche > 3) a colori distinti.
    Gestisce ID non consecutivi (es. 0, 1, 4) e assegna il grigio al rumore (-1).
    """
    # 1. Estrai gli ID dei soli cluster validi (escludendo il rumore -1)
    unique_valid_clusters = np.unique(labels[labels != -1])
    actual_n_clusters = len(unique_valid_clusters)
    
    n_colors = actual_n_clusters if K_clusters is None else K_clusters

    # 2. Inizializza la maschera colori con il grigio neutro per il rumore (RGB: 0.5, 0.5, 0.5)
    point_colors = np.ones((len(coords), 3), dtype=np.float32) * 0.5

    # 3. Assegna i colori ai cluster validi
    if n_colors > 0 and actual_n_clusters > 0:
        # Scegli la palette ideale in base al numero di cluster
        if n_colors <= 10:
            cmap = plt.get_cmap("tab10")
        elif n_colors <= 20:
            cmap = plt.get_cmap("tab20")
        else:
            cmap = plt.get_cmap("hsv")

        # Genera N colori ben distanziati
        colors = cmap(np.linspace(0, 1, n_colors))[:, :3]

        # Mappa ogni ID di cluster al suo colore corrispondente
        for idx, cluster_id in enumerate(unique_valid_clusters[:n_colors]):
            mask = (labels == cluster_id)
            point_colors[mask] = colors[idx]

    # 4. Crea e restituisce la scena Trimesh
    cluster_scene = numpy_to_trimesh(coords, point_colors)
    return cluster_scene

def save_trimesh_image(scene: Scene, path, filename: str):
    filepath = os.path.join(path, filename)
    if os.path.exists(filepath): print(f"{txtcolors.WARNING}Overwriting file {filepath}{txtcolors.ENDC}")
    image = scene.save_image(resolution=(1920, 1080), line_settings={'point_size': 1.0})
    with open(filepath, "wb") as f:
        f.write(image)

class txtcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == "__main__":
    main()
