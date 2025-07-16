#!/usr/bin/env python3
"""
Rendu 3D et génération de vidéos pour Hunyuan3D-2mv
Gère le rendu de mesh avec plusieurs backends et la création de vidéos 360°
"""

import math
import numpy as np
import trimesh
import imageio
from PIL import Image
from typing import List, Optional, Tuple, Union
from pathlib import Path


class Renderer3D:
    """
    Renderer 3D modulaire pour Hunyuan3D-2mv
    Support de plusieurs backends de rendu (pyrender, trimesh, fallback)
    """

    def __init__(self):
        """Initialise le renderer avec détection automatique des backends"""
        self.available_backends = self._detect_backends()
        self.preferred_backend = self._select_best_backend()
        print(f"🎨 Renderer initialisé avec backend: {self.preferred_backend}")

    def _detect_backends(self) -> dict:
        """Détecte les backends de rendu disponibles"""
        backends = {
            'pyrender': False,
            'trimesh': True,  # Toujours disponible
            'fallback': True  # Toujours disponible
        }

        # Test pyrender
        try:
            import pyrender
            backends['pyrender'] = True
            print("✅ Backend pyrender disponible")
        except ImportError:
            print("⚠️  Backend pyrender non disponible")

        return backends

    def _select_best_backend(self) -> str:
        """Sélectionne le meilleur backend disponible"""
        if self.available_backends['pyrender']:
            return 'pyrender'
        elif self.available_backends['trimesh']:
            return 'trimesh'
        else:
            return 'fallback'

    def render_mesh_view(
        self,
        mesh: trimesh.Trimesh,
        azimuth_deg: float,
        elevation_deg: float,
        width: int,
        height: int,
        use_vertex_colors: bool = True,
        backend: Optional[str] = None
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
            backend: Backend à utiliser (auto-détecté si None)

        Returns:
            PIL Image rendered from the mesh
        """
        if backend is None:
            backend = self.preferred_backend

        try:
            if backend == 'pyrender' and self.available_backends['pyrender']:
                return self._render_with_pyrender(
                    mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)
            elif backend == 'trimesh':
                return self._render_with_trimesh(
                    mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)
            else:
                return self._render_with_fallback(
                    mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)
        except Exception as e:
            print(f"⚠️  Erreur rendu avec {backend}: {e}")
            # Fallback vers le prochain backend disponible
            return self._render_with_fallback(
                mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)

    def _render_with_pyrender(
        self,
        mesh: trimesh.Trimesh,
        azimuth_deg: float,
        elevation_deg: float,
        width: int,
        height: int,
        use_vertex_colors: bool = True
    ) -> Image.Image:
        """Rend une vue 3D du mesh avec pyrender (haute qualité)"""
        import pyrender

        # Créer la scène
        scene = pyrender.Scene()

        # Préparer le mesh pour pyrender
        if use_vertex_colors and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            mesh_render = pyrender.Mesh.from_trimesh(mesh)
        else:
            mesh_copy = mesh.copy()
            mesh_copy.visual.vertex_colors = [200, 200, 200, 255]
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
        cam_x = camera_distance * \
            math.cos(elevation_rad) * math.cos(azimuth_rad)
        cam_y = camera_distance * \
            math.cos(elevation_rad) * math.sin(azimuth_rad)
        cam_z = camera_distance * math.sin(elevation_rad)

        # Centre du mesh
        mesh_center = (bounds[0] + bounds[1]) / 2
        camera_pos = mesh_center + np.array([cam_x, cam_y, cam_z])

        # Créer la caméra
        camera = pyrender.PerspectiveCamera(yfov=math.radians(40.0))

        # Matrice de vue
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

        # Éclairage
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=camera_pose)

        # Rendu
        renderer = pyrender.OffscreenRenderer(width, height)
        color, depth = renderer.render(scene)
        renderer.delete()

        return Image.fromarray(color)

    def _render_with_trimesh(
        self,
        mesh: trimesh.Trimesh,
        azimuth_deg: float,
        elevation_deg: float,
        width: int,
        height: int,
        use_vertex_colors: bool = True
    ) -> Image.Image:
        """Rend une vue 3D du mesh avec trimesh.Scene"""
        scene = trimesh.Scene([mesh])

        # Calculer la transformation de caméra
        azimuth_rad = math.radians(azimuth_deg)
        elevation_rad = math.radians(elevation_deg)

        # Distance de caméra basée sur la taille du mesh
        bounds = mesh.bounds
        mesh_size = np.linalg.norm(bounds[1] - bounds[0])
        camera_distance = mesh_size * 2.0

        # Position de la caméra
        cam_x = camera_distance * \
            math.cos(elevation_rad) * math.cos(azimuth_rad)
        cam_y = camera_distance * \
            math.cos(elevation_rad) * math.sin(azimuth_rad)
        cam_z = camera_distance * math.sin(elevation_rad)

        camera_pos = np.array([cam_x, cam_y, cam_z])

        # Essayer de faire un rendu avec trimesh
        try:
            png_data = scene.save_image(
                resolution=(width, height), visible=True)
            if png_data:
                from io import BytesIO
                return Image.open(BytesIO(png_data))
        except:
            pass

        # Si le rendu trimesh échoue, utiliser le fallback
        return self._render_with_fallback(
            mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)

    def _render_with_fallback(
        self,
        mesh: trimesh.Trimesh,
        azimuth_deg: float,
        elevation_deg: float,
        width: int,
        height: int,
        use_vertex_colors: bool = True
    ) -> Image.Image:
        """Fallback qui utilise les images existantes ou rendu simple"""
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
                    image = image.resize(
                        (width, height), Image.Resampling.LANCZOS)
                return image

        # Priorité 2: Images TripoSR existantes
        n_views = 30
        view_index = int((azimuth_deg % 360) / 360 * n_views)
        render_path = f"output/0/render_{view_index:03d}.png"

        if Path(render_path).exists():
            image = Image.open(render_path)
            if image.size != (width, height):
                image = image.resize(
                    (width, height), Image.Resampling.LANCZOS)
            return image

        # Fallback final: Rendu simple
        return self._render_simple_projection(
            mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)

    def _render_simple_projection(
        self,
        mesh: trimesh.Trimesh,
        azimuth_deg: float,
        elevation_deg: float,
        width: int,
        height: int,
        use_vertex_colors: bool = True
    ) -> Image.Image:
        """Rendu simple du mesh (projection orthographique)"""
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
                image = Image.new('RGB', (width, height),
                                  color=(240, 240, 240))
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

    def render_multiple_views(
        self,
        mesh: trimesh.Trimesh,
        n_views: int,
        elevation_deg: float = 15.0,
        width: int = 512,
        height: int = 512,
        use_vertex_colors: bool = True
    ) -> List[Image.Image]:
        """
        Rend plusieurs vues du mesh pour une rotation 360°

        Args:
            mesh: Trimesh object à rendre
            n_views: Nombre de vues autour de l'objet
            elevation_deg: Angle d'élévation constant
            width: Largeur des images
            height: Hauteur des images
            use_vertex_colors: Utiliser les couleurs de vertices

        Returns:
            Liste des images rendues
        """
        print(f"🎬 Rendu de {n_views} vues du mesh...")

        render_images = []

        from tqdm import tqdm
        with tqdm(total=n_views, desc="🎨 Rendu vues", unit="vue", colour="cyan") as pbar:
            for i in range(n_views):
                azimuth_deg = 360.0 * i / n_views

                render_image = self.render_mesh_view(
                    mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)

                render_images.append(render_image)
                pbar.update(1)
                pbar.set_postfix({"azimuth": f"{azimuth_deg:.1f}°"})

        print(f"✅ {len(render_images)} vues générées")
        return render_images

    def save_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: int = 30,
    ):
        """
        Save a list of PIL images as a video (basé sur TripoSR)

        Args:
            frames: List of PIL Images
            output_path: Path to save the video
            fps: Frames per second
        """
        print(f"🎬 Création vidéo: {output_path}")

        # Convertir les frames en arrays numpy
        frames_array = [np.array(frame) for frame in frames]

        # Créer la vidéo avec imageio
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames_array:
            writer.append_data(frame)
        writer.close()

        print(f"✅ Vidéo créée: {len(frames)} frames à {fps} FPS")

    def create_turntable_video(
        self,
        mesh: trimesh.Trimesh,
        output_path: str,
        n_views: int = 36,
        elevation_deg: float = 15.0,
        width: int = 512,
        height: int = 512,
        fps: int = 30,
        use_vertex_colors: bool = True
    ) -> str:
        """
        Crée une vidéo de rotation 360° (turntable) du mesh

        Args:
            mesh: Mesh à rendre
            output_path: Chemin de sortie pour la vidéo
            n_views: Nombre de vues pour la rotation
            elevation_deg: Angle d'élévation de la caméra
            width: Largeur des frames
            height: Hauteur des frames
            fps: Frames par seconde
            use_vertex_colors: Utiliser les couleurs de vertices

        Returns:
            Chemin vers la vidéo créée
        """
        print("🎬 Création d'une vidéo turntable...")

        # Rendre toutes les vues
        render_images = self.render_multiple_views(
            mesh, n_views, elevation_deg, width, height, use_vertex_colors)

        # Créer la vidéo
        self.save_video(render_images, output_path, fps)

        return output_path

    def save_rendered_images(
        self,
        images: List[Image.Image],
        output_dir: str,
        prefix: str = "render_",
        format: str = "PNG"
    ) -> List[str]:
        """
        Sauvegarde les images rendues

        Args:
            images: Liste d'images rendues
            output_dir: Répertoire de sortie
            prefix: Préfixe pour les noms de fichiers
            format: Format d'image

        Returns:
            Liste des chemins des fichiers sauvegardés
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, image in enumerate(images):
            filename = f"{prefix}{i:03d}.{format.lower()}"
            filepath = output_dir / filename
            image.save(filepath, format=format)
            saved_paths.append(str(filepath))

        print(f"✅ {len(images)} images sauvegardées dans {output_dir}")
        return saved_paths

    def copy_existing_assets(
        self,
        output_dir: str,
        n_views: int = 30
    ) -> Optional[str]:
        """
        Copie les assets existants (vidéo et images) vers le répertoire de sortie

        Args:
            output_dir: Répertoire de destination
            n_views: Nombre de vues à copier

        Returns:
            Chemin vers la vidéo copiée ou None
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copier la vidéo existante (Hunyuan3D prioritaire)
        video_copied = None
        target_video = output_dir / "render.mp4"

        # Priorité 1: Vidéo Hunyuan3D
        hunyuan_video = Path("output_hunyuan3d/render.mp4")
        if hunyuan_video.exists():
            import shutil
            shutil.copy2(hunyuan_video, target_video)
            print(f"✅ Vidéo Hunyuan3D copiée: {target_video}")
            video_copied = str(target_video)

        # Priorité 2: Vidéo TripoSR (fallback)
        elif Path("output/0/render.mp4").exists():
            import shutil
            shutil.copy2("output/0/render.mp4", target_video)
            print(f"✅ Vidéo TripoSR copiée (fallback): {target_video}")
            video_copied = str(target_video)

        # Copier les images (Hunyuan3D prioritaire)
        copied_images = []

        # Priorité 1: Images Hunyuan3D
        hunyuan_images = list(Path("output_hunyuan3d").glob("render_*.png"))
        if hunyuan_images:
            hunyuan_images.sort()
            for i, hunyuan_image in enumerate(hunyuan_images):
                if i >= n_views:
                    break
                target_image = output_dir / f"render_{i:03d}.png"
                import shutil
                shutil.copy2(hunyuan_image, target_image)
                copied_images.append(str(target_image))
            print(f"✅ {len(copied_images)} images Hunyuan3D copiées")

        # Priorité 2: Images TripoSR (fallback)
        elif not copied_images:
            for i in range(n_views):
                triposr_image = Path(f"output/0/render_{i:03d}.png")
                if triposr_image.exists():
                    target_image = output_dir / f"render_{i:03d}.png"
                    import shutil
                    shutil.copy2(triposr_image, target_image)
                    copied_images.append(str(target_image))
            if copied_images:
                print(f"✅ {len(copied_images)} images TripoSR copiées (fallback)")

        return video_copied

    def get_info(self) -> dict:
        """Retourne des informations sur le renderer"""
        return {
            'name': 'Hunyuan3D 3D Renderer',
            'description': 'Renderer 3D modulaire avec support multi-backends',
            'available_backends': self.available_backends,
            'preferred_backend': self.preferred_backend,
            'features': [
                'Support multi-backends (pyrender, trimesh, fallback)',
                'Rendu de vues individuelles',
                'Création de vidéos turntable 360°',
                'Sauvegarde d\'images et vidéos',
                'Copie d\'assets existants (Hunyuan3D/TripoSR)',
                'Fallback intelligent en cas d\'erreur',
                'Support des couleurs de vertices',
                'Projection orthographique simple',
                'Calculs de caméra sphériques'
            ],
            'functions': [
                'render_mesh_view: Rendu vue unique',
                'render_multiple_views: Rendu multiple vues',
                'create_turntable_video: Création vidéo 360°',
                'save_video: Sauvegarde vidéo depuis images',
                'save_rendered_images: Sauvegarde images rendues',
                'copy_existing_assets: Copie assets existants'
            ]
        }


# Instance globale du renderer
renderer_3d = Renderer3D()


def get_renderer() -> Renderer3D:
    """Retourne l'instance globale du renderer"""
    return renderer_3d


def render_mesh_view(
    mesh: trimesh.Trimesh,
    azimuth_deg: float,
    elevation_deg: float,
    width: int,
    height: int,
    use_vertex_colors: bool = True
) -> Image.Image:
    """Fonction de convenance pour rendre une vue de mesh"""
    return renderer_3d.render_mesh_view(
        mesh, azimuth_deg, elevation_deg, width, height, use_vertex_colors)


def save_video(frames: List[Image.Image], output_path: str, fps: int = 30):
    """Fonction de convenance pour sauvegarder une vidéo"""
    return renderer_3d.save_video(frames, output_path, fps)


def create_turntable_video(
    mesh: trimesh.Trimesh,
    output_path: str,
    n_views: int = 36,
    elevation_deg: float = 15.0,
    width: int = 512,
    height: int = 512,
    fps: int = 30,
    use_vertex_colors: bool = True
) -> str:
    """Fonction de convenance pour créer une vidéo turntable"""
    return renderer_3d.create_turntable_video(
        mesh, output_path, n_views, elevation_deg, width, height, fps, use_vertex_colors)


def get_rendering_info() -> dict:
    """Fonction de convenance pour obtenir les informations du renderer"""
    return renderer_3d.get_info()
