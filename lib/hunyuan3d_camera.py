#!/usr/bin/env python3
"""
Utilitaires de caméra et rayons pour Hunyuan3D-2mv
Gère le positionnement sphérique des caméras et les calculs de rayons
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Union, Optional
import numpy as np


def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
    normalize: bool = True,
) -> torch.FloatTensor:
    """
    Get ray directions for all pixels in camera coordinate (basé sur TripoSR)

    Args:
        H: Image height
        W: Image width
        focal: Focal length (float or tuple of fx, fy)
        principal: Principal point (cx, cy), defaults to image center
        use_pixel_centers: Whether to use pixel centers (default True)
        normalize: Whether to normalize directions (default True)

    Returns:
        Ray directions tensor of shape (H, W, 3)
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)

    if normalize:
        directions = F.normalize(directions, dim=-1)

    return directions


def get_rays(
    directions,
    c2w,
    keepdim=False,
    normalize=False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Get rays origins and directions (basé sur TripoSR)

    Args:
        directions: Ray directions
        c2w: Camera-to-world transformation matrices
        keepdim: Whether to keep dimensions
        normalize: Whether to normalize ray directions

    Returns:
        Tuple of (ray_origins, ray_directions)
    """
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] *
                  c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_spherical_cameras(
    n_views: int,
    elevation_deg: float,
    camera_distance: float,
    fovy_deg: float,
    height: int,
    width: int,
):
    """
    Get spherical cameras for 360° rendering (basé sur TripoSR)

    Args:
        n_views: Number of views around the object
        elevation_deg: Camera elevation angle in degrees
        camera_distance: Distance from camera to object
        fovy_deg: Field of view in Y direction (degrees)
        height: Image height
        width: Image width

    Returns:
        Tuple of (ray_origins, ray_directions) for all views
    """
    azimuth_deg = torch.linspace(0, 360.0, n_views + 1)[:n_views]
    elevation_deg = torch.full_like(azimuth_deg, elevation_deg)
    camera_distances = torch.full_like(elevation_deg, camera_distance)

    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180

    # convert spherical coordinates to cartesian coordinates
    # right hand coordinate system, x back, y right, z up
    # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )

    # default scene center at origin
    center = torch.zeros_like(camera_positions)
    # default camera up direction as +z
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        None, :].repeat(n_views, 1)

    fovy = torch.full_like(elevation_deg, fovy_deg) * math.pi / 180

    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1),
         camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0

    # get directions by dividing directions_unit_focal by focal length
    focal_length = 0.5 * height / torch.tan(0.5 * fovy)
    directions_unit_focal = get_ray_directions(
        H=height,
        W=width,
        focal=1.0,
    )
    directions = directions_unit_focal[None, :, :, :].repeat(n_views, 1, 1, 1)
    directions[:, :, :, :2] = (
        directions[:, :, :, :2] / focal_length[:, None, None, None]
    )
    # must use normalize=True to normalize directions here
    rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=True)

    return rays_o, rays_d


def calculate_camera_position(
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    center: np.ndarray = None
) -> np.ndarray:
    """
    Calcule la position d'une caméra en coordonnées sphériques

    Args:
        azimuth_deg: Angle d'azimuth en degrés (rotation autour de l'axe Z)
        elevation_deg: Angle d'élévation en degrés (angle par rapport au plan XY)
        distance: Distance de la caméra par rapport au centre
        center: Position du centre (par défaut: origine)

    Returns:
        Position de la caméra en coordonnées cartésiennes
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])

    # Conversion en radians
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)

    # Coordonnées sphériques → cartésiennes
    x = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = distance * math.sin(elevation_rad)

    return center + np.array([x, y, z])


def calculate_camera_matrix(
    camera_pos: np.ndarray,
    target_pos: np.ndarray = None,
    up_vector: np.ndarray = None
) -> np.ndarray:
    """
    Calcule la matrice de transformation caméra-monde

    Args:
        camera_pos: Position de la caméra
        target_pos: Position du point visé (par défaut: origine)
        up_vector: Vecteur "up" de la caméra (par défaut: [0,0,1])

    Returns:
        Matrice 4x4 de transformation caméra vers monde
    """
    if target_pos is None:
        target_pos = np.array([0.0, 0.0, 0.0])
    if up_vector is None:
        up_vector = np.array([0.0, 0.0, 1.0])

    # Direction de vue (forward)
    forward = target_pos - camera_pos
    forward = forward / np.linalg.norm(forward)

    # Vecteur droit (right)
    right = np.cross(forward, up_vector)
    if np.linalg.norm(right) < 1e-6:
        # Cas dégénéré: forward parallèle à up
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right = right / np.linalg.norm(right)

    # Vecteur up ajusté
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Matrice de transformation
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward  # Convention OpenGL
    c2w[:3, 3] = camera_pos

    return c2w


def generate_camera_views(
    n_views: int,
    elevation_deg: float = 15.0,
    distance: float = 2.0,
    center: np.ndarray = None
) -> list:
    """
    Génère une liste de positions de caméras pour un rendu 360°

    Args:
        n_views: Nombre de vues autour de l'objet
        elevation_deg: Angle d'élévation constant
        distance: Distance constante par rapport au centre
        center: Centre de rotation (par défaut: origine)

    Returns:
        Liste de tuples (position_camera, matrice_transformation)
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])

    views = []
    for i in range(n_views):
        azimuth_deg = 360.0 * i / n_views

        # Position de la caméra
        camera_pos = calculate_camera_position(
            azimuth_deg, elevation_deg, distance, center)

        # Matrice de transformation
        c2w_matrix = calculate_camera_matrix(camera_pos, center)

        views.append((camera_pos, c2w_matrix, azimuth_deg))

    return views


def get_optimal_camera_distance(
    mesh_bounds: np.ndarray,
    fovy_deg: float = 40.0,
    margin_factor: float = 1.2
) -> float:
    """
    Calcule la distance optimale de caméra pour un mesh donné

    Args:
        mesh_bounds: Bounds du mesh (shape: (2, 3) - min/max pour x,y,z)
        fovy_deg: Angle de vue vertical en degrés
        margin_factor: Facteur de marge pour s'assurer que l'objet est entièrement visible

    Returns:
        Distance optimale de caméra
    """
    # Calculer la taille du mesh
    mesh_size = np.linalg.norm(mesh_bounds[1] - mesh_bounds[0])

    # Convertir le FOV en radians
    fovy_rad = math.radians(fovy_deg)

    # Distance nécessaire pour voir l'objet entier
    distance = (mesh_size / 2.0) / math.tan(fovy_rad / 2.0)

    # Appliquer la marge
    return distance * margin_factor


def create_perspective_camera(
    fovy_deg: float,
    aspect_ratio: float,
    near: float = 0.1,
    far: float = 100.0
) -> np.ndarray:
    """
    Crée une matrice de projection perspective

    Args:
        fovy_deg: Angle de vue vertical en degrés
        aspect_ratio: Ratio largeur/hauteur
        near: Plan de clipping proche
        far: Plan de clipping lointain

    Returns:
        Matrice de projection 4x4
    """
    fovy_rad = math.radians(fovy_deg)
    f = 1.0 / math.tan(fovy_rad / 2.0)

    projection = np.zeros((4, 4))
    projection[0, 0] = f / aspect_ratio
    projection[1, 1] = f
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = (2.0 * far * near) / (near - far)
    projection[3, 2] = -1.0

    return projection


def project_point_to_screen(
    point_3d: np.ndarray,
    view_matrix: np.ndarray,
    projection_matrix: np.ndarray,
    viewport_size: tuple
) -> tuple:
    """
    Projette un point 3D vers les coordonnées écran

    Args:
        point_3d: Point 3D en coordonnées monde
        view_matrix: Matrice de vue (world → camera)
        projection_matrix: Matrice de projection
        viewport_size: Taille du viewport (width, height)

    Returns:
        Coordonnées écran (x, y) et profondeur z
    """
    # Convertir en coordonnées homogènes
    point_homo = np.append(point_3d, 1.0)

    # Transformation vue
    point_view = view_matrix @ point_homo

    # Projection
    point_proj = projection_matrix @ point_view

    # Division par w (perspective)
    if point_proj[3] != 0:
        point_ndc = point_proj[:3] / point_proj[3]
    else:
        point_ndc = point_proj[:3]

    # Conversion vers coordonnées écran
    width, height = viewport_size
    screen_x = (point_ndc[0] + 1.0) * 0.5 * width
    screen_y = (1.0 - point_ndc[1]) * 0.5 * height  # Inverser Y

    return screen_x, screen_y, point_ndc[2]


def get_camera_info():
    """Retourne des informations sur le module caméra"""
    return {
        'name': 'Hunyuan3D Camera Utils',
        'description': 'Utilitaires de caméra et rayons pour rendu 3D',
        'functions': [
            'get_ray_directions: Génération des directions de rayons',
            'get_rays: Calcul des rayons de caméra',
            'get_spherical_cameras: Positionnement sphérique des caméras',
            'calculate_camera_position: Position caméra en coordonnées sphériques',
            'calculate_camera_matrix: Matrice de transformation caméra',
            'generate_camera_views: Génération de vues 360°',
            'get_optimal_camera_distance: Distance optimale selon le mesh',
            'create_perspective_camera: Matrice de projection perspective',
            'project_point_to_screen: Projection 3D → 2D'
        ],
        'features': [
            'Support coordonnées sphériques',
            'Calculs de rayons optimisés',
            'Positionnement automatique des caméras',
            'Projection perspective complète',
            'Compatible avec PyTorch et NumPy',
            'Optimisé pour rendu 360°'
        ]
    }
