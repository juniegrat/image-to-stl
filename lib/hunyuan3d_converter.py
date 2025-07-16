#!/usr/bin/env python3
"""
Module de conversion STL pour Hunyuan3D-2mv (version modulaire)
Gère la conversion d'images de pièces (avers/revers) en modèles STL 3D haute fidélité
Utilise des modules spécialisés pour une architecture modulaire
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image
import trimesh
from tqdm import tqdm

# Import des modules spécialisés
try:
    # Import relatif quand utilisé comme module
    from .hunyuan3d_config import get_config, set_quality_mode, QualityMode
    from .hunyuan3d_models import get_model_manager
    from .hunyuan3d_image_processing import get_image_processor
    from .hunyuan3d_mesh_processing import get_mesh_processor
    from .hunyuan3d_rendering import get_renderer
    from .hunyuan3d_camera import get_camera_info
except ImportError:
    # Import absolu quand utilisé directement
    from hunyuan3d_config import get_config, set_quality_mode, QualityMode
    from hunyuan3d_models import get_model_manager
    from hunyuan3d_image_processing import get_image_processor
    from hunyuan3d_mesh_processing import get_mesh_processor
    from hunyuan3d_rendering import get_renderer
    from hunyuan3d_camera import get_camera_info

# Supprimer les warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Hunyuan3DConverter:
    """
    Convertisseur principal pour Hunyuan3D-2mv (version modulaire)
    Optimisé pour les pièces numismatiques avec support multi-view
    Architecture modulaire avec composants spécialisés
    """

    def __init__(self, model_path="tencent/Hunyuan3D-2",
                 texture_model_path="tencent/Hunyuan3D-2",
                 device=None,
                 disable_texture=False):
        """
        Initialise le convertisseur Hunyuan3D-2mv modulaire

        Args:
            model_path: Chemin vers le modèle de forme (défaut: Hunyuan3D-2mv)
            texture_model_path: Chemin vers le modèle de texture
            device: Device à utiliser (auto-détecté si None)
            disable_texture: Désactiver complètement le chargement du modèle de texture
        """
        # Initialiser les gestionnaires modulaires
        self.config_manager = get_config()
        self.model_manager = get_model_manager()
        self.image_processor = get_image_processor()
        self.mesh_processor = get_mesh_processor()
        self.renderer = get_renderer()

        # Configuration du gestionnaire de modèles
        self.model_manager.model_path = model_path
        self.model_manager.texture_model_path = texture_model_path
        if device:
            self.model_manager.device = device
        self.model_manager.disable_texture = disable_texture

        print(f"🔧 Convertisseur Hunyuan3D-2mv modulaire initialisé")
        print(f"   Device: {self.model_manager.device}")
        print(f"   Modèle forme: {self.model_manager.model_path}")
        print(f"   Modèle texture: {self.model_manager.texture_model_path}")
        print(f"   Architecture: modulaire (6 composants)")

    def enable_test_mode(self):
        """Active le mode test ultra-rapide pour les tests et développement"""
        print("⚡ Activation du mode TEST ultra-rapide")
        set_quality_mode(QualityMode.TEST)

    def enable_debug_mode(self):
        """Active le mode debug ultra-minimal pour tests instantanés"""
        print("⚡ Activation du mode DEBUG (modèle lisse et simple)")
        set_quality_mode(QualityMode.DEBUG)

    def enable_fast_mode(self):
        """Active le mode rapide (compromis qualité/vitesse)"""
        print("🏃 Activation du mode RAPIDE (compromis qualité/vitesse)")
        set_quality_mode(QualityMode.FAST)

    def enable_ultra_mode(self):
        """Active le mode ultra qualité (paramètres maximaux)"""
        print("🌟 Activation du mode ULTRA qualité")
        set_quality_mode(QualityMode.ULTRA)

    def check_environment(self):
        """Vérifie l'environnement et les dépendances"""
        return self.model_manager.check_environment()

    def load_models(self):
        """Charge les modèles Hunyuan3D-2"""
        return self.model_manager.load_models()

    def prepare_images(self, image_paths: List[str], remove_bg: bool = False) -> List[Image.Image]:
        """
        Prépare les images pour le traitement

        Args:
            image_paths: Liste des chemins vers les images
            remove_bg: Si True, supprime l'arrière-plan

        Returns:
            Liste des images préparées
        """
        config = self.config_manager.get_config()
        return self.image_processor.prepare_multiple_images(
            image_paths,
            target_size=config['image_size'],
            remove_bg=remove_bg,
            foreground_ratio=config['foreground_ratio']
        )

    def generate_3d_mesh(self, images: List[Image.Image],
                         output_dir: str = "output") -> Optional[trimesh.Trimesh]:
        """
        Génère un mesh 3D à partir des images avec loading bar

        Args:
            images: Liste des images préparées
            output_dir: Répertoire de sortie

        Returns:
            Mesh 3D généré ou None en cas d'erreur
        """
        print("🏗️  Génération du mesh 3D...")

        config = self.config_manager.get_config()

        # Callback de progression avec tqdm
        with tqdm(total=config['num_inference_steps'], desc="🔄 Génération mesh",
                  unit="step", colour="green") as pbar:

            def callback(step, timestep, latents):
                pbar.update(1)
                pbar.set_postfix({"timestep": f"{timestep:.1f}"})

            mesh = self.model_manager.generate_3d_mesh(
                images, config, progress_callback=callback)

        if mesh:
            # Sauvegarder le mesh temporaire
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            temp_mesh_path = output_path / "temp_mesh.obj"
            mesh.export(str(temp_mesh_path))

        return mesh

    def apply_texture(self, mesh: trimesh.Trimesh,
                      reference_image: Image.Image) -> trimesh.Trimesh:
        """
        Applique une texture au mesh avec timer

        Args:
            mesh: Mesh 3D de base
            reference_image: Image de référence pour la texture

        Returns:
            Mesh texturé
        """
        print("🎨 Application de la texture...")
        config = self.config_manager.get_config()

        start_time = time.time()
        textured_mesh = self.model_manager.apply_texture(
            mesh, reference_image, config)
        elapsed_time = time.time() - start_time

        print(
            f"   ⏱️  Texture appliquée en {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
        return textured_mesh

    def apply_vertex_colors(self, mesh: trimesh.Trimesh, reference_image: Image.Image) -> trimesh.Trimesh:
        """
        Applique des couleurs de vertices rapides

        Args:
            mesh: Mesh 3D de base
            reference_image: Image de référence pour les couleurs

        Returns:
            Mesh avec couleurs de vertices (rapide, sans texture)
        """
        return self.mesh_processor.apply_vertex_colors(mesh, reference_image)

    def render_video(self, mesh: trimesh.Trimesh, output_dir: str = "output") -> Optional[str]:
        """
        Génère une vidéo de rotation 360° avec les paramètres de configuration

        Args:
            mesh: Mesh 3D à rendre
            output_dir: Répertoire de sortie

        Returns:
            Chemin vers la vidéo générée ou None en cas d'erreur
        """
        config = self.config_manager.get_config()
        render_params = self.config_manager.get_render_params()

        # Préparer le mesh avec l'orientation standard
        oriented_mesh = mesh.copy()
        oriented_mesh = self.mesh_processor.to_gradio_3d_orientation(
            oriented_mesh)

        # Normaliser seulement si pas en mode debug
        if not self.config_manager.is_debug_mode():
            oriented_mesh = self.mesh_processor.normalize_mesh(oriented_mesh)
        else:
            print("   ⚡ Mode DEBUG - normalisation désactivée pour vitesse")

        # Utiliser le renderer modulaire
        video_path = Path(output_dir) / "render.mp4"
        return self.renderer.create_turntable_video(
            oriented_mesh,
            str(video_path),
            n_views=render_params['n_views'],
            elevation_deg=render_params['elevation_deg'],
            width=render_params['width'],
            height=render_params['height'],
            fps=render_params['fps'],
            use_vertex_colors=True
        )

    def post_process_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Post-traite le mesh selon la configuration

        Args:
            mesh: Mesh d'entrée

        Returns:
            Mesh post-traité
        """
        preserve_details = not self.config_manager.should_skip_post_processing()
        return self.mesh_processor.post_process_mesh(mesh, preserve_details)

    def convert_to_stl(self, mesh: trimesh.Trimesh, output_path: str) -> bool:
        """
        Convertit le mesh en STL

        Args:
            mesh: Mesh à convertir
            output_path: Chemin de sortie

        Returns:
            True si succès, False sinon
        """
        print("📦 Conversion en STL...")

        try:
            # Post-traiter le mesh selon la configuration
            if self.config_manager.should_skip_post_processing():
                print("   ⚡ Mode rapide - export direct sans post-processing")
                processed_mesh = mesh
            else:
                processed_mesh = self.post_process_mesh(mesh)

            # Exporter en STL via le mesh processor
            return self.mesh_processor.export_mesh(processed_mesh, output_path, 'stl')

        except Exception as e:
            print(f"   ❌ Erreur conversion STL: {e}")
            return False

    def convert_coin_to_stl(self, front_image: str, back_image: str = None,
                            output_dir: str = "output_hunyuan3d",
                            remove_background: bool = False,
                            render_video: bool = True,
                            enable_post_processing: bool = False,
                            use_vertex_colors: bool = False) -> Optional[str]:
        """
        Convertit une pièce (avers/revers) en STL avec architecture modulaire

        Args:
            front_image: Chemin vers l'image de face
            back_image: Chemin vers l'image de dos (optionnel)
            output_dir: Répertoire de sortie
            remove_background: Supprimer l'arrière-plan
            render_video: Générer une vidéo de rotation
            enable_post_processing: Activer le post-processing du mesh
            use_vertex_colors: Utiliser des couleurs de vertices rapides

        Returns:
            Chemin vers le fichier STL ou None en cas d'erreur
        """
        print("🪙 Conversion de pièce avec Hunyuan3D-2mv (architecture modulaire)")
        print("=" * 70)

        start_time = time.time()

        # Préparer les images avec le processeur d'images
        image_paths = [front_image]
        if back_image:
            image_paths.append(back_image)

        images = self.prepare_images(image_paths, remove_background)
        if not images:
            print("❌ Aucune image préparée")
            return None

        # Générer le mesh 3D avec le gestionnaire de modèles
        mesh = self.generate_3d_mesh(images, output_dir)
        if not mesh:
            print("❌ Échec génération mesh")
            return None

        # Appliquer la texture ou les couleurs de vertices selon le mode
        if use_vertex_colors:
            # Mode vertex colors rapide (quelques secondes)
            colored_mesh = self.apply_vertex_colors(mesh, images[0])
            print(
                f"   📊 Mesh avec vertex colors: {len(colored_mesh.vertices)} vertices, {len(colored_mesh.faces)} faces")
        elif not self.model_manager.disable_texture and self.model_manager.texture_pipeline:
            # Mode texture complet (8+ minutes)
            colored_mesh = self.apply_texture(mesh, images[0])
            print(
                f"   📊 Mesh avec texture: {len(colored_mesh.vertices)} vertices, {len(colored_mesh.faces)} faces")
        else:
            # Mode sans couleur ni texture (ultra-rapide)
            colored_mesh = mesh
            print(
                f"   📊 Mesh sans couleur: {len(colored_mesh.vertices)} vertices, {len(colored_mesh.faces)} faces")

        # Post-traiter le mesh selon la configuration
        if self.config_manager.is_debug_mode():
            print("🔄 Mode DEBUG - aucun post-processing (mesh brut instantané)")
            final_mesh = colored_mesh
        elif enable_post_processing and not self.config_manager.should_skip_post_processing():
            print("🔄 Post-processing activé")
            final_mesh = self.post_process_mesh(colored_mesh)
        else:
            print("🔄 Post-processing désactivé - préservation maximale des détails")
            final_mesh = colored_mesh

        # Convertir en STL
        stl_path = Path(output_dir) / "coin_model.stl"
        if self.convert_to_stl(final_mesh, str(stl_path)):
            elapsed_time = time.time() - start_time

            print(f"\n✅ Conversion terminée en {elapsed_time:.1f}s")
            print(f"📁 Fichier STL: {stl_path}")

            # Générer la vidéo si demandé
            if render_video:
                video_path = self.render_video(final_mesh, output_dir)
                if video_path:
                    print(f"🎬 Vidéo générée (modulaire): {video_path}")

            if back_image:
                print("🔄 Modèle généré avec avers et revers")
            else:
                print("🔄 Modèle généré avec vue unique")

            return str(stl_path)

        return None

    def get_system_info(self) -> dict:
        """Retourne des informations complètes sur le système modulaire"""
        return {
            'converter': {
                'name': 'Hunyuan3D-2mv Converter (Modulaire)',
                'version': '3.0-modular',
                'architecture': 'modulaire'
            },
            'modules': {
                'config': self.config_manager.get_info() if hasattr(self.config_manager, 'get_info') else get_config().get_info(),
                'models': self.model_manager.get_info(),
                'image_processing': self.image_processor.get_info(),
                'mesh_processing': self.mesh_processor.get_info(),
                'rendering': self.renderer.get_info(),
                'camera': get_camera_info()
            },
            'current_config': self.config_manager.get_config()
        }


def convert_coin_hunyuan3d(front_image: str, back_image: str = None,
                           output_dir: str = "output_hunyuan3d",
                           model_path: str = "tencent/Hunyuan3D-2",
                           remove_background: bool = False,
                           render_video: bool = True,
                           enable_post_processing: bool = False,
                           use_vertex_colors: bool = False) -> Optional[str]:
    """
    Fonction de convenance pour convertir une pièce avec Hunyuan3D-2mv modulaire

    Args:
        front_image: Chemin vers l'image de face
        back_image: Chemin vers l'image de dos (optionnel)
        output_dir: Répertoire de sortie
        model_path: Chemin vers le modèle Hunyuan3D
        remove_background: Supprimer l'arrière-plan
        render_video: Générer une vidéo de rotation
        enable_post_processing: Activer le post-processing du mesh
        use_vertex_colors: Utiliser des couleurs de vertices rapides

    Returns:
        Chemin vers le fichier STL ou None en cas d'erreur
    """
    # Créer le convertisseur modulaire
    converter = Hunyuan3DConverter(model_path)

    # Vérifier l'environnement
    if not converter.check_environment():
        return None

    # Charger les modèles
    if not converter.load_models():
        return None

    # Convertir
    return converter.convert_coin_to_stl(
        front_image, back_image, output_dir, remove_background,
        render_video, enable_post_processing, use_vertex_colors
    )


def get_hunyuan3d_info():
    """Retourne des informations sur Hunyuan3D-2 modulaire"""
    converter = Hunyuan3DConverter()
    return converter.get_system_info()
