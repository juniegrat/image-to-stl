#!/usr/bin/env python3
"""
Utilitaires modulaires pour Hunyuan3D-2mv
Complètement indépendant de TripoSR mais utilise les mêmes images de rendu
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional, Tuple, Any, Union
import trimesh
import imageio
import rembg
from pathlib import Path


def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
    normalize: bool = True,
) -> torch.FloatTensor:
    """
    Get ray directions for all pixels in camera coordinate (copied from TripoSR)

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
    Get rays origins and directions (copied from TripoSR)

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
    Get spherical cameras for 360° rendering (copied from TripoSR)

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


def save_video(
    frames: List[Image.Image],
    output_path: str,
    fps: int = 30,
):
    """
    Save a list of PIL images as a video (copied from TripoSR)

    Args:
        frames: List of PIL Images
        output_path: Path to save the video
        fps: Frames per second
    """
    # use imageio to save video
    frames = [np.array(frame) for frame in frames]
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def to_gradio_3d_orientation(mesh):
    """
    Apply standard 3D orientation transformation (copied from TripoSR)

    Args:
        mesh: Trimesh object

    Returns:
        Mesh with applied orientation transformation
    """
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    return mesh


def remove_background(
    image: Image.Image,
    rembg_session=None,
    force: bool = False,
    **rembg_kwargs,
) -> Image.Image:
    """
    Remove background from image using rembg (copied from TripoSR)

    Args:
        image: PIL Image
        rembg_session: rembg session (optional)
        force: Force background removal even if image has alpha
        **rembg_kwargs: Additional arguments for rembg.remove

    Returns:
        Image with background removed (RGBA format)
    """
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        # Avoid parameter conflict by being explicit about arguments
        if rembg_session is not None:
            image = rembg.remove(image, session=rembg_session)
        else:
            # Filter out 'session' from rembg_kwargs if present to avoid conflicts
            filtered_kwargs = {k: v for k,
                               v in rembg_kwargs.items() if k != 'session'}
            image = rembg.remove(image, **filtered_kwargs)
    return image


def resize_foreground(
    image: Image.Image,
    ratio: float,
) -> Image.Image:
    """
    Resize foreground object to specific ratio within image (copied from TripoSR)

    Args:
        image: PIL Image with alpha channel (RGBA)
        ratio: Desired ratio of foreground to image size

    Returns:
        Resized image with foreground at specified ratio
    """
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = Image.fromarray(new_image)
    return new_image


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Normalize mesh to unit scale and center at origin

    Args:
        mesh: Trimesh object to normalize

    Returns:
        Normalized mesh
    """
    # Center the mesh at origin
    center = mesh.bounds.mean(axis=0)
    mesh.vertices -= center

    # Scale to unit size
    scale = 1.0 / mesh.scale
    mesh.vertices *= scale

    return mesh


def debug_mesh_properties(mesh: trimesh.Trimesh) -> dict:
    """
    Debug function to examine mesh properties and colors

    Args:
        mesh: Trimesh object to examine

    Returns:
        Dictionary with mesh properties
    """
    props = {
        'vertices_count': len(mesh.vertices),
        'faces_count': len(mesh.faces),
        'bounds': mesh.bounds.tolist(),
        'scale': float(mesh.scale),
        'has_vertex_colors': hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None,
        'has_visual': hasattr(mesh, 'visual'),
        'has_visual_vertex_colors': False,
        'has_visual_face_colors': False,
        'vertex_colors_shape': None,
        'vertex_colors_dtype': None,
        'vertex_colors_range': None,
        'visual_type': None
    }

    # Check vertex_colors directly
    if props['has_vertex_colors']:
        props['vertex_colors_shape'] = mesh.vertex_colors.shape
        props['vertex_colors_dtype'] = str(mesh.vertex_colors.dtype)
        props['vertex_colors_range'] = (
            mesh.vertex_colors.min(), mesh.vertex_colors.max())

    # Check visual properties
    if props['has_visual']:
        props['visual_type'] = type(mesh.visual).__name__

        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            props['has_visual_vertex_colors'] = True
            props['visual_vertex_colors_shape'] = mesh.visual.vertex_colors.shape
            props['visual_vertex_colors_dtype'] = str(
                mesh.visual.vertex_colors.dtype)
            props['visual_vertex_colors_range'] = (
                mesh.visual.vertex_colors.min(), mesh.visual.vertex_colors.max())

        if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
            props['has_visual_face_colors'] = True
            props['visual_face_colors_shape'] = mesh.visual.face_colors.shape
            props['visual_face_colors_dtype'] = str(
                mesh.visual.face_colors.dtype)
            props['visual_face_colors_range'] = (
                mesh.visual.face_colors.min(), mesh.visual.face_colors.max())

    return props


def render_mesh_view(
    mesh: trimesh.Trimesh,
    azimuth_deg: float,
    elevation_deg: float,
    width: int,
    height: int,
    use_vertex_colors: bool = True
) -> Image.Image:
    """
    Rend une vue 3D du mesh à partir d'un angle donné

    Args:
        mesh: Trimesh object à rendre
        azimuth_deg: Azimuth angle in degrees  
        elevation_deg: Elevation angle in degrees
        width: Image width
        height: Image height
        use_vertex_colors: Whether to use vertex colors

    Returns:
        PIL Image rendered from the mesh
    """
    try:
        # Essayer d'utiliser pyrender pour un rendu de qualité
        try:
            import pyrender
            return render_mesh_view_pyrender(mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)
        except ImportError:
            # Fallback vers trimesh.Scene si pyrender n'est pas disponible
            return render_mesh_view_trimesh(mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)
    except Exception as e:
        print(f"⚠️  Erreur rendu mesh: {e}")
        # Fallback final: essayer d'utiliser les images existantes si le rendu échoue
        return render_mesh_view_fallback_images(mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)


def render_mesh_view_pyrender(
    mesh: trimesh.Trimesh,
    azimuth_deg: float,
    elevation_deg: float,
    width: int,
    height: int,
    use_vertex_colors: bool = True
) -> Image.Image:
    """
    Rend une vue 3D du mesh avec pyrender (haute qualité)
    """
    import pyrender

    # Créer la scène
    scene = pyrender.Scene()

    # Préparer le mesh pour pyrender
    if use_vertex_colors and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        # Utiliser les couleurs de vertices si disponibles
        mesh_render = pyrender.Mesh.from_trimesh(mesh)
    else:
        # Utiliser une couleur uniforme
        mesh_copy = mesh.copy()
        mesh_copy.visual.vertex_colors = [200, 200, 200, 255]  # Gris clair
        mesh_render = pyrender.Mesh.from_trimesh(mesh_copy)

    scene.add(mesh_render)

    # Calculer la position de la caméra
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)

    # Distance de la caméra basée sur la taille du mesh
    bounds = mesh.bounds
    mesh_size = np.linalg.norm(bounds[1] - bounds[0])
    camera_distance = mesh_size * 1.5

    # Position de la caméra en coordonnées sphériques
    cam_x = camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    cam_y = camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    cam_z = camera_distance * math.sin(elevation_rad)

    # Centre du mesh
    mesh_center = (bounds[0] + bounds[1]) / 2
    camera_pos = mesh_center + np.array([cam_x, cam_y, cam_z])

    # Créer la caméra
    camera = pyrender.PerspectiveCamera(yfov=math.radians(40.0))

    # Matrice de vue (regarder vers le centre du mesh)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_pos

    # Orienter la caméra vers le centre
    forward = mesh_center - camera_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, [0, 0, 1])
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, [0, 1, 0])
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = -forward

    scene.add(camera, pose=camera_pose)

    # Ajouter l'éclairage
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Rendu
    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(scene)
    renderer.delete()

    return Image.fromarray(color)


def render_mesh_view_trimesh(
    mesh: trimesh.Trimesh,
    azimuth_deg: float,
    elevation_deg: float,
    width: int,
    height: int,
    use_vertex_colors: bool = True
) -> Image.Image:
    """
    Rend une vue 3D du mesh avec trimesh.Scene (fallback)
    """
    scene = trimesh.Scene([mesh])

    # Calculer la transformation de caméra
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)

    # Distance de caméra basée sur la taille du mesh
    bounds = mesh.bounds
    mesh_size = np.linalg.norm(bounds[1] - bounds[0])
    camera_distance = mesh_size * 2.0

    # Position de la caméra
    cam_x = camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    cam_y = camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    cam_z = camera_distance * math.sin(elevation_rad)

    camera_pos = np.array([cam_x, cam_y, cam_z])

    # Essayer de faire un rendu avec trimesh
    try:
        png_data = scene.save_image(resolution=(width, height), visible=True)
        if png_data:
            from io import BytesIO
            return Image.open(BytesIO(png_data))
    except:
        pass

    # Si le rendu trimesh échoue, utiliser une approche simplifiée
    return render_mesh_view_simple(mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)


def render_mesh_view_simple(
    mesh: trimesh.Trimesh,
    azimuth_deg: float,
    elevation_deg: float,
    width: int,
    height: int,
    use_vertex_colors: bool = True
) -> Image.Image:
    """
    Rendu simple du mesh (projection orthographique)
    """
    # Rotation du mesh selon les angles
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)

    # Matrice de rotation
    cos_az, sin_az = math.cos(azimuth_rad), math.sin(azimuth_rad)
    cos_el, sin_el = math.cos(elevation_rad), math.sin(elevation_rad)

    # Rotation autour de Z puis X
    rot_z = np.array([
        [cos_az, -sin_az, 0],
        [sin_az, cos_az, 0],
        [0, 0, 1]
    ])

    rot_x = np.array([
        [1, 0, 0],
        [0, cos_el, -sin_el],
        [0, sin_el, cos_el]
    ])

    rotation_matrix = rot_z @ rot_x

    # Appliquer la rotation aux vertices
    vertices = mesh.vertices @ rotation_matrix.T

    # Projeter en 2D (projection orthographique)
    x_2d = vertices[:, 0]
    y_2d = vertices[:, 1]

    # Normaliser les coordonnées à l'image
    if len(x_2d) > 0:
        x_min, x_max = x_2d.min(), x_2d.max()
        y_min, y_max = y_2d.min(), y_2d.max()

        if x_max != x_min and y_max != y_min:
            margin = 0.1
            x_range = x_max - x_min
            y_range = y_max - y_min
            scale = min(width * (1 - 2 * margin) / x_range,
                        height * (1 - 2 * margin) / y_range)

            x_img = ((x_2d - x_min) / x_range *
                     (1 - 2 * margin) + margin) * width
            y_img = height - ((y_2d - y_min) / y_range *
                              (1 - 2 * margin) + margin) * height

            # Créer l'image
            from PIL import ImageDraw
            image = Image.new('RGB', (width, height), color=(240, 240, 240))
            draw = ImageDraw.Draw(image)

            # Dessiner les faces
            faces = mesh.faces
            for face in faces:
                points = [(int(x_img[face[i]]), int(y_img[face[i]]))
                          for i in range(3)]
                if all(0 <= p[0] < width and 0 <= p[1] < height for p in points):
                    draw.polygon(points, fill=(180, 180, 180),
                                 outline=(100, 100, 100))

            return image

    # Image vide si échec
    return Image.new('RGB', (width, height), color=(200, 200, 200))


def render_mesh_view_fallback_images(
    mesh: trimesh.Trimesh,
    azimuth_deg: float,
    elevation_deg: float,
    width: int,
    height: int,
    use_vertex_colors: bool = True
) -> Image.Image:
    """
    Fallback qui utilise les images existantes si le rendu échoue
    """
    # Priorité 1: Utiliser les images Hunyuan3D existantes
    hunyuan_images = list(Path("output_hunyuan3d").glob("render_*.png"))
    if hunyuan_images:
        hunyuan_images.sort()
        n_views = len(hunyuan_images)

        if n_views > 0:
            view_index = int((azimuth_deg % 360) / 360 * n_views)
            view_index = min(view_index, n_views - 1)

            selected_image = hunyuan_images[view_index]
            image = Image.open(selected_image)

            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            return image

    # Priorité 2: Utiliser les images TripoSR existantes
    n_views = 30
    view_index = int((azimuth_deg % 360) / 360 * n_views)
    render_path = f"output/0/render_{view_index:03d}.png"

    if Path(render_path).exists():
        image = Image.open(render_path)
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image

    # Image par défaut
    print(
        f"⚠️  Aucune image disponible pour azimuth={azimuth_deg}°, utilisation d'une image par défaut")
    return Image.new('RGB', (width, height), color=(128, 128, 128))


def render_mesh_view_fallback(
    mesh: trimesh.Trimesh,
    azimuth_deg: float,
    elevation_deg: float,
    width: int,
    height: int,
    use_vertex_colors: bool = True
) -> Image.Image:
    """
    Fallback qui utilise les images TripoSR existantes
    """
    return render_mesh_view_fallback_images(mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)


def use_generated_video_if_available(output_dir: str) -> Optional[str]:
    """
    Utilise la vidéo générée existante si disponible (Hunyuan3D prioritaire, puis TripoSR)

    Args:
        output_dir: Répertoire de sortie

    Returns:
        Chemin vers la vidéo existante ou None
    """
    target_video = Path(output_dir) / "render.mp4"

    # Priorité 1: Vidéo Hunyuan3D existante
    hunyuan_video = Path("output_hunyuan3d/render.mp4")
    if hunyuan_video.exists():
        import shutil
        shutil.copy2(hunyuan_video, target_video)
        print(f"✅ Vidéo Hunyuan3D copiée: {target_video}")
        return str(target_video)

    # Priorité 2: Vidéo TripoSR existante (fallback)
    triposr_video = Path("output/0/render.mp4")
    if triposr_video.exists():
        import shutil
        shutil.copy2(triposr_video, target_video)
        print(f"✅ Vidéo TripoSR copiée (fallback): {target_video}")
        return str(target_video)

    return None


def copy_generated_renders(output_dir: str, n_views: int = 30) -> List[str]:
    """
    Copie les images de rendu générées vers le répertoire de sortie (Hunyuan3D prioritaire, puis TripoSR)

    Args:
        output_dir: Répertoire de sortie
        n_views: Nombre de vues à copier

    Returns:
        Liste des chemins vers les images copiées
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_images = []

    # Priorité 1: Images Hunyuan3D existantes
    hunyuan_images = list(Path("output_hunyuan3d").glob("render_*.png"))
    if hunyuan_images:
        # Trier par nom pour avoir l'ordre correct
        hunyuan_images.sort()

        for i, hunyuan_image in enumerate(hunyuan_images):
            if i >= n_views:
                break
            target_image = output_dir / f"render_{i:03d}.png"
            import shutil
            shutil.copy2(hunyuan_image, target_image)
            copied_images.append(str(target_image))

        print(f"✅ {len(copied_images)} images Hunyuan3D copiées")
        return copied_images

    # Priorité 2: Images TripoSR existantes (fallback)
    for i in range(n_views):
        triposr_image = Path(f"output/0/render_{i:03d}.png")
        if triposr_image.exists():
            target_image = output_dir / f"render_{i:03d}.png"
            import shutil
            shutil.copy2(triposr_image, target_image)
            copied_images.append(str(target_image))

    if copied_images:
        print(f"✅ {len(copied_images)} images TripoSR copiées (fallback)")

    return copied_images


def get_hunyuan3d_info():
    """Retourne des informations sur Hunyuan3D-2"""
    info = {
        'name': 'Hunyuan3D-2mv (assets Hunyuan3D)',
        'description': 'Utilise directement les images et vidéos générées par Hunyuan3D, avec fallback TripoSR',
        'version': '2.0-hunyuan3d-assets',
        'features': [
            'Support multi-view (avers/revers)',
            'Génération haute résolution',
            'Texture réaliste avec détection automatique',
            'Utilise directement les rendus Hunyuan3D existants',
            'Fallback automatique vers TripoSR si nécessaire',
            'Loading bars de progression',
            'Suppression arrière-plan',
            'Optimisé pour pièces numismatiques',
            'Complètement indépendant de TripoSR',
            'Réutilise les assets Hunyuan3D (images + vidéo)',
            'Copie intelligente des renders existants',
            'Support des couleurs de vertices Hunyuan3D',
            'Priorité aux assets Hunyuan3D sur TripoSR'
        ],
        'requirements': [
            'CUDA 11.8+ (recommandé)',
            'GPU avec 16GB+ VRAM',
            'Python 3.8+',
            'Hunyuan3D-2 installé',
            'Images Hunyuan3D dans output_hunyuan3d/',
            'trimesh pour manipulation'
        ],
        'utils': [
            'get_ray_directions: Génération des directions de rayons',
            'get_rays: Calcul des rayons de caméra',
            'get_spherical_cameras: Positionnement des caméras sphériques',
            'save_video: Création de vidéos MP4',
            'to_gradio_3d_orientation: Orientation standard des modèles',
            'remove_background: Suppression d\'arrière-plan',
            'resize_foreground: Redimensionnement intelligent',
            'normalize_mesh: Normalisation des mesh',
            'render_mesh_view: Utilise les images Hunyuan3D existantes',
            'use_generated_video_if_available: Copie vidéo Hunyuan3D/TripoSR',
            'copy_generated_renders: Copie images Hunyuan3D/TripoSR',
            'debug_mesh_properties: Diagnostic des propriétés mesh'
        ]
    }
    return info
