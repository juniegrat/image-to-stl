#!/usr/bin/env python3
"""
Utilitaires de compatibilité pour Hunyuan3D-2mv (version modulaire)
Fournit des fonctions de compatibilité pour l'ancien code qui utilise les utilitaires
Ce module fait maintenant le pont vers les nouveaux modules spécialisés
"""

import warnings
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import trimesh

# Import des nouveaux modules spécialisés
try:
    # Import relatif
    from .hunyuan3d_camera import (
        get_ray_directions,
        get_rays,
        get_spherical_cameras
    )
    from .hunyuan3d_rendering import save_video
    from .hunyuan3d_mesh_processing import (
        normalize_mesh,
        to_gradio_3d_orientation,
        debug_mesh_properties
    )
    from .hunyuan3d_image_processing import (
        remove_background_from_image as remove_background,
        get_image_processor
    )
    from .hunyuan3d_rendering import (
        render_mesh_view,
        get_renderer
    )
except ImportError:
    # Import absolu
    from hunyuan3d_camera import (
        get_ray_directions,
        get_rays,
        get_spherical_cameras
    )
    from hunyuan3d_rendering import save_video
    from hunyuan3d_mesh_processing import (
        normalize_mesh,
        to_gradio_3d_orientation,
        debug_mesh_properties
    )
    from hunyuan3d_image_processing import (
        remove_background_from_image as remove_background,
        get_image_processor
    )
    from hunyuan3d_rendering import (
        render_mesh_view,
        get_renderer
    )

# Supprimer les warnings de dépréciation pour ce module de compatibilité
warnings.filterwarnings("ignore", category=DeprecationWarning)


def resize_foreground(image: Image.Image, ratio: float) -> Image.Image:
    """
    Fonction de compatibilité pour resize_foreground

    Args:
        image: PIL Image with alpha channel (RGBA)
        ratio: Desired ratio of foreground to image size

    Returns:
        Resized image with foreground at specified ratio
    """
    warnings.warn(
        "resize_foreground depuis hunyuan3d_utils est dépréciée. "
        "Utilisez hunyuan3d_image_processing.ImageProcessor.resize_foreground",
        DeprecationWarning,
        stacklevel=2
    )

    processor = get_image_processor()
    return processor.resize_foreground(image, ratio)


def use_generated_video_if_available(output_dir: str) -> Optional[str]:
    """
    Fonction de compatibilité pour use_generated_video_if_available

    Args:
        output_dir: Répertoire de sortie

    Returns:
        Chemin vers la vidéo existante ou None
    """
    warnings.warn(
        "use_generated_video_if_available depuis hunyuan3d_utils est dépréciée. "
        "Utilisez hunyuan3d_rendering.Renderer3D.copy_existing_assets",
        DeprecationWarning,
        stacklevel=2
    )

    renderer = get_renderer()
    return renderer.copy_existing_assets(output_dir)


def copy_generated_renders(output_dir: str, n_views: int = 30) -> List[str]:
    """
    Fonction de compatibilité pour copy_generated_renders

    Args:
        output_dir: Répertoire de sortie
        n_views: Nombre de vues à copier

    Returns:
        Liste des chemins vers les images copiées
    """
    warnings.warn(
        "copy_generated_renders depuis hunyuan3d_utils est dépréciée. "
        "Utilisez hunyuan3d_rendering.Renderer3D.copy_existing_assets",
        DeprecationWarning,
        stacklevel=2
    )

    renderer = get_renderer()
    renderer.copy_existing_assets(output_dir, n_views)

    # Retourner une liste vide pour compatibilité
    # (la nouvelle fonction ne retourne que le chemin vidéo)
    return []


def get_hunyuan3d_info():
    """
    Fonction de compatibilité pour get_hunyuan3d_info

    Returns:
        Informations sur Hunyuan3D modulaire
    """
    warnings.warn(
        "get_hunyuan3d_info depuis hunyuan3d_utils est dépréciée. "
        "Utilisez hunyuan3d_converter.get_hunyuan3d_info",
        DeprecationWarning,
        stacklevel=2
    )

    return {
        'name': 'Hunyuan3D-2mv (Modulaire - Compatibilité)',
        'description': 'Module de compatibilité vers la nouvelle architecture modulaire',
        'version': '3.0-compat',
        'status': 'DÉPRÉCIÉ - Utilisez les nouveaux modules spécialisés',
        'migration': {
            'camera': 'hunyuan3d_camera',
            'rendering': 'hunyuan3d_rendering',
            'mesh_processing': 'hunyuan3d_mesh_processing',
            'image_processing': 'hunyuan3d_image_processing',
            'models': 'hunyuan3d_models',
            'config': 'hunyuan3d_config'
        },
        'deprecated_functions': [
            'resize_foreground → hunyuan3d_image_processing.ImageProcessor.resize_foreground',
            'use_generated_video_if_available → hunyuan3d_rendering.Renderer3D.copy_existing_assets',
            'copy_generated_renders → hunyuan3d_rendering.Renderer3D.copy_existing_assets',
            'get_hunyuan3d_info → hunyuan3d_converter.get_hunyuan3d_info'
        ],
        'active_functions': [
            'get_ray_directions → hunyuan3d_camera.get_ray_directions',
            'get_rays → hunyuan3d_camera.get_rays',
            'get_spherical_cameras → hunyuan3d_camera.get_spherical_cameras',
            'save_video → hunyuan3d_rendering.save_video',
            'normalize_mesh → hunyuan3d_mesh_processing.normalize_mesh',
            'to_gradio_3d_orientation → hunyuan3d_mesh_processing.to_gradio_3d_orientation',
            'debug_mesh_properties → hunyuan3d_mesh_processing.debug_mesh_properties',
            'remove_background → hunyuan3d_image_processing.remove_background_from_image',
            'render_mesh_view → hunyuan3d_rendering.render_mesh_view'
        ]
    }


# Fonctions re-exportées depuis les nouveaux modules (sans dépréciation)
# Ces fonctions sont directement importées depuis les modules spécialisés

# Camera utilities (depuis hunyuan3d_camera)
__all__ = [
    'get_ray_directions',
    'get_rays',
    'get_spherical_cameras',

    # Rendering utilities (depuis hunyuan3d_rendering)
    'save_video',
    'render_mesh_view',

    # Mesh processing utilities (depuis hunyuan3d_mesh_processing)
    'normalize_mesh',
    'to_gradio_3d_orientation',
    'debug_mesh_properties',

    # Image processing utilities (depuis hunyuan3d_image_processing)
    'remove_background',

    # Fonctions de compatibilité (dépréciées)
    'resize_foreground',
    'use_generated_video_if_available',
    'copy_generated_renders',
    'get_hunyuan3d_info'
]


def print_migration_guide():
    """Affiche un guide de migration vers les nouveaux modules"""
    print("📋 Guide de migration Hunyuan3D-2mv vers architecture modulaire")
    print("=" * 70)
    print()
    print("🏗️  NOUVELLE ARCHITECTURE:")
    print("   • hunyuan3d_config.py      - Configuration et modes de qualité")
    print("   • hunyuan3d_models.py      - Gestion des modèles et pipelines")
    print("   • hunyuan3d_camera.py      - Utilitaires de caméra et rayons")
    print("   • hunyuan3d_rendering.py   - Rendu 3D et génération vidéos")
    print("   • hunyuan3d_mesh_processing.py - Traitement et optimisation mesh")
    print("   • hunyuan3d_image_processing.py - Traitement d'images")
    print()
    print("🔄 MIGRATIONS PRINCIPALES:")
    print("   ANCIEN: from hunyuan3d_utils import normalize_mesh")
    print("   NOUVEAU: from hunyuan3d_mesh_processing import normalize_mesh")
    print()
    print("   ANCIEN: from hunyuan3d_utils import render_mesh_view")
    print("   NOUVEAU: from hunyuan3d_rendering import render_mesh_view")
    print()
    print("   ANCIEN: from hunyuan3d_utils import remove_background")
    print("   NOUVEAU: from hunyuan3d_image_processing import remove_background_from_image")
    print()
    print("💡 Vous pouvez continuer à utiliser hunyuan3d_utils (avec warnings)")
    print("   mais nous recommandons de migrer vers les nouveaux modules")
    print()
    print("📖 Documentation complète: voir les docstrings des nouveaux modules")


if __name__ == "__main__":
    print_migration_guide()
