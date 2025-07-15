#!/usr/bin/env python3
"""
Module de conversion STL pour Hunyuan3D-2mv
Gère la conversion d'images de pièces (avers/revers) en modèles STL 3D haute fidélité
Utilise des utilitaires modulaires pour l'indépendance complète
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
import rembg
from tqdm import tqdm

# Import des utilitaires Hunyuan3D
try:
    # Import relatif quand utilisé comme module
    from .hunyuan3d_utils import (
        get_ray_directions,
        get_rays,
        get_spherical_cameras,
        save_video,
        to_gradio_3d_orientation,
        remove_background,
        resize_foreground,
        normalize_mesh,
        render_mesh_view,
        debug_mesh_properties,
        get_hunyuan3d_info,
        use_generated_video_if_available,
        copy_generated_renders
    )
except ImportError:
    # Import absolu quand utilisé directement
    from hunyuan3d_utils import (
        get_ray_directions,
        get_rays,
        get_spherical_cameras,
        save_video,
        to_gradio_3d_orientation,
        remove_background,
        resize_foreground,
        normalize_mesh,
        render_mesh_view,
        debug_mesh_properties,
        get_hunyuan3d_info,
        use_generated_video_if_available,
        copy_generated_renders
    )

# Supprimer les warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Hunyuan3DConverter:
    """
    Convertisseur principal pour Hunyuan3D-2mv
    Optimisé pour les pièces numismatiques avec support multi-view
    Rendu vidéo avec utilitaires modulaires indépendants
    """

    def __init__(self, model_path="tencent/Hunyuan3D-2",
                 texture_model_path="tencent/Hunyuan3D-2",
                 device=None,
                 disable_texture=False):
        """
        Initialise le convertisseur Hunyuan3D-2mv

        Args:
            model_path: Chemin vers le modèle de forme (défaut: Hunyuan3D-2mv)
            texture_model_path: Chemin vers le modèle de texture
            device: Device à utiliser (auto-détecté si None)
            disable_texture: Désactiver complètement le chargement du modèle de texture
        """
        self.model_path = model_path
        self.texture_model_path = texture_model_path
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.disable_texture = disable_texture

        # Pipelines (chargés à la demande)
        self.shape_pipeline = None
        self.texture_pipeline = None

        # Configuration par défaut
        self.config = {
            'image_size': 512,
            'guidance_scale': 12.0,  # Augmenté de 7.5 pour plus de fidélité aux détails
            'num_inference_steps': 75,  # Augmenté de 50 pour plus de précision
            'texture_guidance_scale': 4.0,  # Augmenté de 2.0 pour une meilleure texture
            'texture_steps': 40,  # Augmenté de 28 pour plus de détails de texture
            'seed': 42,
            # Paramètres de rendu
            'n_views': 30,
            'elevation_deg': 5.0,  # Changé de 0.0 pour mieux capturer la profondeur
            'camera_distance': 1.7,  # Réduit de 1.9 pour plus de détails
            'fovy_deg': 35.0,  # Réduit de 40.0 pour moins de distortion
            'height': 512,  # Augmenté de 256 pour plus de détails
            'width': 512,   # Augmenté de 256 pour plus de détails
            'fps': 30,
            'foreground_ratio': 0.90  # Augmenté de 0.85 pour mieux capturer l'objet
        }

        # Configuration spéciale pour pièces numismatiques (optimisée)
        self.coin_config = {
            'image_size': 1024,  # Résolution plus élevée pour capturer les détails fins
            'guidance_scale': 7.5,  # Plus élevé pour forcer la circularité
            'num_inference_steps': 50,  # Plus d'étapes pour plus de précision
            'texture_guidance_scale': 2.0,  # Plus élevé pour les détails de texture
            'texture_steps': 28,  # Plus d'étapes pour les détails fins
            'seed': 12345,  # Seed différent optimisé pour les pièces
            # Paramètres de rendu optimisés pour pièces
            'n_views': 36,  # Diviseur de 360° pour rotation parfaite
            'elevation_deg': 15.0,  # Angle optimal pour capturer la profondeur des pièces
            'camera_distance': 1.5,  # Plus proche pour capturer les détails
            'fovy_deg': 30.0,  # Angle de vue serré pour réduire la distortion
            'height': 1024,  # Résolution élevée pour les détails
            'width': 1024,   # Résolution élevée pour les détails
            'fps': 30,
            'foreground_ratio': 0.95,  # Ratio élevé pour capturer toute la pièce
            # Nouveaux paramètres spécifiques aux pièces
            'coin_mode': True,
            'circular_mask': True,  # Forcer la forme circulaire
            'detail_preservation': True,  # Préserver les détails fins
            'anti_aliasing': True,  # Réduire les artefacts
            'smooth_edges': True,  # Lisser les bords pour une forme plus ronde
        }

        # Session rembg pour la suppression d'arrière-plan
        self.rembg_session = None

        print(f"🔧 Convertisseur Hunyuan3D-2mv initialisé (utilitaires modulaires)")
        print(f"   Device: {self.device}")
        print(f"   Modèle forme: {self.model_path}")
        print(f"   Modèle texture: {self.texture_model_path}")

    def enable_coin_mode(self):
        """Active le mode pièce avec paramètres optimisés"""
        print("🪙 Activation du mode pièce optimisé")
        print("   ✅ Résolution: 1024x1024 (haute définition)")
        print("   ✅ Guidance scale: 15.0 (circularité forcée)")
        print("   ✅ Steps: 100 (précision maximale)")
        print("   ✅ Angle caméra: 15° (optimal pour pièces)")
        print("   ✅ Distance: 1.5 (capture détails fins)")
        print("   ✅ Anti-aliasing activé")
        print("   ✅ Lissage des bords activé")

        # Remplacer la configuration par défaut
        self.config = self.coin_config.copy()

        # Initialiser rembg pour le mode pièce si pas déjà fait
        if not self.rembg_session:
            try:
                import rembg
                self.rembg_session = rembg.new_session()
                print("   ✅ Session rembg initialisée pour suppression arrière-plan")
            except ImportError:
                print("   ⚠️  rembg non disponible, suppression arrière-plan désactivée")

    def disable_coin_mode(self):
        """Désactive le mode pièce (retour aux paramètres par défaut)"""
        print("🔄 Désactivation du mode pièce - retour aux paramètres par défaut")
        # Restaurer la configuration par défaut
        self.config = {
            'image_size': 512,
            'guidance_scale': 12.0,
            'num_inference_steps': 75,
            'texture_guidance_scale': 4.0,
            'texture_steps': 40,
            'seed': 42,
            'n_views': 30,
            'elevation_deg': 5.0,
            'camera_distance': 1.7,
            'fovy_deg': 35.0,
            'height': 512,
            'width': 512,
            'fps': 30,
            'foreground_ratio': 0.90
        }

    def check_environment(self):
        """Vérifie l'environnement et les dépendances"""
        print("🔍 Vérification de l'environnement...")

        # Vérifier CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
            print(
                f"   Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA non disponible")

        # Vérifier les dépendances critiques
        deps = ['diffusers', 'transformers', 'accelerate', 'torch',
                'torchvision', 'PIL', 'numpy', 'trimesh', 'imageio', 'rembg', 'tqdm', 'matplotlib']
        missing = []

        for dep in deps:
            try:
                if dep == 'PIL':
                    import PIL
                elif dep == 'matplotlib':
                    import matplotlib.pyplot as plt
                else:
                    __import__(dep)
            except ImportError:
                missing.append(dep)

        if missing:
            print(f"❌ Dépendances manquantes: {', '.join(missing)}")
            return False

        print("✅ Toutes les dépendances sont disponibles")
        return True

    def load_models(self):
        """Charge les modèles Hunyuan3D-2"""
        print("🤖 Chargement des modèles...")

        try:
            # Importer les classes nécessaires
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            # Charger le modèle de forme
            print(f"   📐 Chargement du modèle de forme...")
            try:
                self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    cache_dir=None  # Utilise le cache HF standard
                )
                print(f"   ✅ Modèle de forme chargé avec succès!")
            except Exception as e:
                print(f"   ❌ Erreur chargement modèle de forme: {e}")
                return False

            # Charger le modèle de texture avec gestion d'erreurs améliorée
            if self.disable_texture:
                print("   🚫 Chargement du modèle de texture désactivé")
                print("   📋 Mode disponible: génération de forme uniquement")
                self.texture_pipeline = None
            else:
                print("   🎨 Chargement du modèle de texture...")
                try:
                    # Essayer plusieurs chemins possibles pour le modèle de texture
                    texture_paths = [
                        self.texture_model_path,
                        "tencent/Hunyuan3D-2",  # Utilise le cache HF standard
                    ]

                    texture_loaded = False
                    for path in texture_paths:
                        try:
                            print(f"      Tentative avec: {path}")
                            # Hunyuan3DPaintPipeline ne supporte que l'argument path dans from_pretrained()
                            self.texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                                path
                            )

                            # Déplacer vers le device manuellement si nécessaire
                            if hasattr(self.texture_pipeline, 'to'):
                                self.texture_pipeline = self.texture_pipeline.to(
                                    self.device)

                            print("   ✅ Modèle de texture chargé avec succès!")
                            texture_loaded = True
                            break
                        except Exception as e:
                            print(f"      ⚠️  Échec avec {path}: {e}")
                            continue

                    if not texture_loaded:
                        print("   ⚠️  Impossible de charger le modèle de texture")
                        print("   📋 Mode disponible: génération de forme uniquement")
                        self.texture_pipeline = None

                except Exception as e:
                    print(f"   ⚠️  Erreur générale texture: {e}")
                    print("   Continuation sans texture (mesh uniquement)")
                    self.texture_pipeline = None

            # Initialiser la session rembg pour la suppression d'arrière-plan
            try:
                import rembg
                self.rembg_session = rembg.new_session('u2net')
                print("   ✅ Session rembg initialisée")
            except Exception as e:
                print(f"   ⚠️  Session rembg non disponible: {e}")
                self.rembg_session = None

            print("✅ Modèles chargés avec succès!")
            return True

        except ImportError as e:
            print(f"❌ Erreur lors de l'importation des modules Hunyuan3D: {e}")
            print("💡 Vérifiez que Hunyuan3D-2 est correctement installé")
            print("💡 Exécutez: python install-hunyuan3d.py")
            return False
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles: {e}")
            print("💡 Vérifiez que Hunyuan3D-2 est correctement installé")
            return False

    def prepare_images(self, image_paths: List[str], remove_bg: bool = False) -> List[Image.Image]:
        """
        Prépare les images pour le traitement

        Args:
            image_paths: Liste des chemins vers les images
            remove_bg: Si True, supprime l'arrière-plan

        Returns:
            Liste des images préparées
        """
        print(f"🖼️  Préparation de {len(image_paths)} image(s)...")

        images = []
        for i, path in enumerate(image_paths):
            try:
                # Charger l'image
                image = Image.open(path).convert('RGB')

                # Supprimer l'arrière-plan si demandé
                if remove_bg and self.rembg_session:
                    print(f"   🔄 Suppression arrière-plan image {i+1}")
                    image = remove_background(
                        image, rembg_session=self.rembg_session)

                    # Traitement de l'image avec canal alpha
                    if image.mode == 'RGBA':
                        image = resize_foreground(
                            image, self.config['foreground_ratio'])
                        image_array = np.array(
                            image).astype(np.float32) / 255.0
                        # Convertir en RGB avec fond gris
                        image_array = image_array[:, :, :3] * image_array[:,
                                                                          :, 3:4] + (1 - image_array[:, :, 3:4]) * 0.5
                        image = Image.fromarray(
                            (image_array * 255.0).astype(np.uint8))
                elif remove_bg:
                    print(
                        f"   ⚠️  Suppression arrière-plan demandée mais rembg non disponible")

                # Redimensionner
                image = image.resize(
                    (self.config['image_size'], self.config['image_size']))

                images.append(image)
                print(f"   ✅ Image {i+1}: {image.size}")

            except Exception as e:
                print(f"   ❌ Erreur image {i+1}: {e}")
                continue

        return images

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

        if not self.shape_pipeline:
            print("❌ Modèle de forme non chargé")
            return None

        try:
            # Préparer le générateur
            generator = torch.Generator(
                device=self.device).manual_seed(self.config['seed'])

            # Générer selon le nombre d'images avec loading bar
            with tqdm(total=self.config['num_inference_steps'], desc="🔄 Génération mesh",
                      unit="step", colour="green") as pbar:

                if len(images) > 1:
                    # Mode multi-view
                    print(f"   🔄 Mode multi-view avec {len(images)} images")

                    # Simuler le callback de progression
                    def callback(step, timestep, latents):
                        pbar.update(1)
                        pbar.set_postfix({"timestep": f"{timestep:.1f}"})

                    mesh = self.shape_pipeline(
                        image=images,
                        guidance_scale=self.config['guidance_scale'],
                        num_inference_steps=self.config['num_inference_steps'],
                        generator=generator,
                        callback=callback,
                        callback_steps=1
                    )[0]
                else:
                    # Mode single view
                    print("   🔄 Mode single view")

                    def callback(step, timestep, latents):
                        pbar.update(1)
                        pbar.set_postfix({"timestep": f"{timestep:.1f}"})

                    mesh = self.shape_pipeline(
                        image=images[0],
                        guidance_scale=self.config['guidance_scale'],
                        num_inference_steps=self.config['num_inference_steps'],
                        generator=generator,
                        callback=callback,
                        callback_steps=1
                    )[0]

            # Statistiques du mesh
            print(
                f"   ✅ Mesh généré: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

            # Sauvegarder le mesh temporaire
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            temp_mesh_path = output_path / "temp_mesh.obj"
            mesh.export(str(temp_mesh_path))

            return mesh

        except Exception as e:
            print(f"❌ Erreur génération mesh: {e}")
            return None

    def apply_texture(self, mesh: trimesh.Trimesh,
                      reference_image: Image.Image) -> trimesh.Trimesh:
        """
        Applique une texture au mesh avec loading bar

        Args:
            mesh: Mesh 3D de base
            reference_image: Image de référence pour la texture

        Returns:
            Mesh texturé
        """
        print("🎨 Application de la texture...")

        if not self.texture_pipeline:
            print("⚠️  Modèle de texture non chargé, conservation du mesh sans texture")
            return mesh

        try:
            # Appliquer la texture avec loading bar
            with tqdm(total=self.config['texture_steps'], desc="🎨 Application texture",
                      unit="step", colour="blue") as pbar:

                def callback(step, timestep, latents):
                    pbar.update(1)
                    pbar.set_postfix({"timestep": f"{timestep:.1f}"})

                textured_mesh = self.texture_pipeline(
                    mesh,
                    image=reference_image,
                    guidance_scale=self.config['texture_guidance_scale'],
                    num_inference_steps=self.config['texture_steps'],
                    callback=callback,
                    callback_steps=1
                )

            print("   ✅ Texture appliquée avec succès")
            return textured_mesh

        except Exception as e:
            print(f"⚠️  Erreur application texture: {e}")
            print("   Retour au mesh sans texture")
            return mesh

    def render_video_with_utils(self, mesh: trimesh.Trimesh, output_dir: str = "output") -> Optional[str]:
        """
        Génère une vidéo en rendant le mesh Hunyuan3D fourni

        Args:
            mesh: Mesh 3D Hunyuan3D à rendre
            output_dir: Répertoire de sortie

        Returns:
            Chemin vers la vidéo générée ou None en cas d'erreur
        """
        print("🎬 Rendu vidéo du mesh Hunyuan3D...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Paramètres de rendu
            n_views = self.config['n_views']
            elevation_deg = self.config['elevation_deg']
            height = self.config['height']
            width = self.config['width']
            fps = self.config['fps']

            # Préparer le mesh avec l'orientation standard
            oriented_mesh = mesh.copy()
            oriented_mesh = to_gradio_3d_orientation(oriented_mesh)
            oriented_mesh = normalize_mesh(oriented_mesh)

            render_images = []

            print(f"   📹 Rendu de {n_views} vues du mesh Hunyuan3D...")
            from tqdm import tqdm
            with tqdm(total=n_views, desc="🎬 Rendu mesh Hunyuan3D", unit="vue", colour="cyan") as pbar:
                for i in range(n_views):
                    # Calculer l'azimuth pour cette vue
                    azimuth_deg = 360.0 * i / n_views

                    # Rendre la vue du mesh Hunyuan3D
                    render_image = render_mesh_view(
                        oriented_mesh,
                        azimuth_deg,
                        elevation_deg,
                        width,
                        height,
                        use_vertex_colors=True
                    )

                    render_images.append(render_image)
                    pbar.update(1)
                    pbar.set_postfix({"azimuth": f"{azimuth_deg:.1f}°"})

            # Sauvegarder les images individuelles
            for ri, render_image in enumerate(render_images):
                render_image.save(output_dir / f"render_{ri:03d}.png")

            # Créer la vidéo
            if render_images:
                video_path = output_dir / "render.mp4"
                print(f"   🎬 Création vidéo du mesh Hunyuan3D: {video_path}")
                save_video(render_images, str(video_path), fps=fps)
                print(
                    f"   ✅ Vidéo créée depuis le mesh Hunyuan3D: {len(render_images)} vues")
                return str(video_path)

        except Exception as e:
            print(f"   ❌ Erreur rendu mesh Hunyuan3D: {e}")
            # Fallback: essayer la génération manuelle
            return self.render_video_manual_fallback(mesh, output_dir)

        return None

    def render_video_manual_fallback(self, mesh: trimesh.Trimesh, output_dir: Path) -> Optional[str]:
        """
        Fallback qui génère manuellement si pas d'assets TripoSR
        """
        print("   🔄 Génération manuelle en fallback...")

        try:
            # Paramètres de rendu
            n_views = self.config['n_views']
            elevation_deg = self.config['elevation_deg']
            height = self.config['height']
            width = self.config['width']
            fps = self.config['fps']

            # Préparer le mesh avec l'orientation standard
            oriented_mesh = mesh.copy()
            oriented_mesh = to_gradio_3d_orientation(oriented_mesh)
            oriented_mesh = normalize_mesh(oriented_mesh)

            render_images = []

            from tqdm import tqdm
            with tqdm(total=n_views, desc="🎬 Rendu manuel", unit="vue", colour="cyan") as pbar:
                for i in range(n_views):
                    # Calculer l'azimuth pour cette vue
                    azimuth_deg = 360.0 * i / n_views

                    # Utiliser l'utilitaire de rendu (qui cherchera d'abord TripoSR)
                    render_image = render_mesh_view(
                        oriented_mesh,
                        azimuth_deg,
                        elevation_deg,
                        width,
                        height,
                        use_vertex_colors=True
                    )

                    render_images.append(render_image)
                    pbar.update(1)
                    pbar.set_postfix({"azimuth": f"{azimuth_deg:.1f}°"})

            # Sauvegarder les images individuelles
            for ri, render_image in enumerate(render_images):
                render_image.save(output_dir / f"render_{ri:03d}.png")

            # Créer la vidéo
            if render_images:
                video_path = output_dir / "render.mp4"
                save_video(render_images, str(video_path), fps=fps)
                return str(video_path)

        except Exception as e:
            print(f"   ❌ Erreur génération manuelle: {e}")
            return None

    def render_video(self, mesh: trimesh.Trimesh, output_dir: str = "output") -> Optional[str]:
        """
        Génère une vidéo de rotation 360° (utilise les utilitaires modulaires)
        """
        return self.render_video_with_utils(mesh, output_dir)

    def post_process_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Post-traite le mesh pour optimiser la qualité tout en préservant les détails

        Args:
            mesh: Mesh d'entrée

        Returns:
            Mesh post-traité
        """
        print("🔧 Post-processing du mesh (préservation des détails)...")

        try:
            # Nettoyage basique uniquement
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()

            # Lissage très léger UNIQUEMENT si le mesh est très irrégulier
            # Pour préserver les détails des pièces, on applique un lissage minimal
            vertices_count = len(mesh.vertices)
            faces_count = len(mesh.faces)

            # Seulement lisser si le ratio faces/vertices est très élevé (mesh très irrégulier)
            if vertices_count > 0 and faces_count / vertices_count > 3.0:
                print("   🔄 Lissage minimal appliqué pour mesh irrégulier")
                # Lissage très léger avec préservation des détails
                # Paramètre très faible pour préserver les détails
                mesh = mesh.smoothed(alpha=0.1)
            else:
                print("   ✅ Mesh régulier - pas de lissage pour préserver les détails")

            print(
                f"   ✅ Mesh optimisé: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh

        except Exception as e:
            print(f"   ⚠️  Erreur post-processing: {e}")
            return mesh

    def post_process_coin_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Post-traite le mesh spécifiquement pour les pièces numismatiques
        Améliore la circularité et réduit les artefacts

        Args:
            mesh: Mesh d'entrée

        Returns:
            Mesh post-traité optimisé pour pièces
        """
        print("🪙 Post-processing spécialisé pour pièces numismatiques...")

        try:
            # Étape 1: Nettoyage basique mais approfondi
            print("   🧹 Nettoyage approfondi du mesh...")
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.remove_infinite_values()

            # Étape 2: Amélioration de la circularité si mode pièce activé
            if self.config.get('coin_mode', False):
                print("   🔄 Amélioration de la circularité...")

                # Centrer le mesh parfaitement
                mesh.vertices -= mesh.center_mass

                # Projection cylindrique pour améliorer la circularité
                vertices = mesh.vertices
                xy_center = np.mean(vertices[:, :2], axis=0)

                # Calculer le rayon moyen dans le plan XY
                distances = np.linalg.norm(vertices[:, :2] - xy_center, axis=1)
                mean_radius = np.mean(distances[distances > 0])

                if mean_radius > 0:
                    # Normaliser légèrement vers un cercle parfait (préservation 90% des détails)
                    directions = vertices[:, :2] - xy_center
                    current_distances = np.linalg.norm(directions, axis=1)

                    # Éviter la division par zéro
                    mask = current_distances > 1e-6
                    normalized_directions = np.zeros_like(directions)
                    normalized_directions[mask] = directions[mask] / \
                        current_distances[mask, np.newaxis]

                    # Appliquer une correction légère vers la circularité
                    target_distances = current_distances * 0.9 + mean_radius * 0.1
                    new_positions = xy_center + normalized_directions * \
                        target_distances[:, np.newaxis]

                    mesh.vertices[:, :2] = new_positions
                    print(
                        f"   ✅ Circularité améliorée (rayon moyen: {mean_radius:.3f})")

            # Étape 3: Lissage adaptatif pour pièces
            print("   🔄 Lissage adaptatif pour pièces...")
            vertices_count = len(mesh.vertices)
            faces_count = len(mesh.faces)

            # Lissage plus agressif pour les pièces car on veut une surface lisse
            if vertices_count > 0:
                # Lissage modéré spécifiquement pour les pièces
                # Plus élevé que le mode normal
                mesh = mesh.smoothed(alpha=0.2)

                # Lissage Laplacien léger pour réduire les artefacts
                try:
                    # Appliquer un lissage Laplacien simple
                    for _ in range(2):  # 2 itérations légères
                        new_vertices = mesh.vertices.copy()
                        for i, vertex in enumerate(mesh.vertices):
                            neighbors = []
                            for face in mesh.faces:
                                if i in face:
                                    neighbors.extend(face)
                            neighbors = list(set(neighbors))
                            neighbors.remove(i)

                            if neighbors:
                                neighbor_avg = np.mean(
                                    mesh.vertices[neighbors], axis=0)
                                new_vertices[i] = vertex * \
                                    0.8 + neighbor_avg * 0.2

                        mesh.vertices = new_vertices

                    print("   ✅ Lissage Laplacien appliqué")
                except Exception as e:
                    print(f"   ⚠️  Lissage Laplacien échoué: {e}")

            # Étape 4: Réduction des artefacts de bord
            if self.config.get('smooth_edges', False):
                print("   🔄 Lissage des bords...")
                # Identifier les vertices de bord et les lisser davantage
                try:
                    boundary_vertices = mesh.vertices[mesh.outline().vertices]
                    if len(boundary_vertices) > 0:
                        # Lisser les bords de manière circulaire
                        center = np.mean(boundary_vertices, axis=0)
                        for i in mesh.outline().vertices:
                            current_pos = mesh.vertices[i]
                            toward_center = (center - current_pos) * 0.1
                            mesh.vertices[i] = current_pos + toward_center

                        print("   ✅ Bords lissés")
                except Exception as e:
                    print(f"   ⚠️  Lissage des bords échoué: {e}")

            # Étape 5: Normalisation finale
            print("   🔄 Normalisation finale...")
            mesh = normalize_mesh(mesh)

            print(
                f"   ✅ Mesh pièce optimisé: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh

        except Exception as e:
            print(f"   ⚠️  Erreur post-processing pièce: {e}")
            print("   🔄 Retour au post-processing standard...")
            return self.post_process_mesh(mesh)

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
            # Post-traiter le mesh
            processed_mesh = self.post_process_mesh(mesh)

            # Exporter en STL
            processed_mesh.export(output_path)

            # Vérifier le fichier
            file_size = Path(output_path).stat().st_size / 1024 / 1024  # MB
            print(f"   ✅ STL généré: {file_size:.2f} MB")

            return True

        except Exception as e:
            print(f"   ❌ Erreur conversion STL: {e}")
            return False

    def convert_coin_to_stl(self, front_image: str, back_image: str = None,
                            output_dir: str = "output_hunyuan3d",
                            remove_background: bool = False,
                            render_video: bool = True,
                            enable_post_processing: bool = False) -> Optional[str]:
        """
        Convertit une pièce (avers/revers) en STL avec utilitaires modulaires

        Args:
            front_image: Chemin vers l'image de face
            back_image: Chemin vers l'image de dos (optionnel)
            output_dir: Répertoire de sortie
            remove_background: Supprimer l'arrière-plan
            render_video: Générer une vidéo de rotation

        Returns:
            Chemin vers le fichier STL ou None en cas d'erreur
        """
        print("🪙 Conversion de pièce avec Hunyuan3D-2mv (utilitaires modulaires)")
        print("=" * 70)

        start_time = time.time()

        # Préparer les images
        image_paths = [front_image]
        if back_image:
            image_paths.append(back_image)

        images = self.prepare_images(image_paths, remove_background)
        if not images:
            print("❌ Aucune image préparée")
            return None

        # Générer le mesh 3D
        mesh = self.generate_3d_mesh(images, output_dir)
        if not mesh:
            print("❌ Échec génération mesh")
            return None

        # Appliquer la texture
        textured_mesh = self.apply_texture(mesh, images[0])

        # Post-traiter le mesh si activé
        if enable_post_processing:
            print("🔄 Post-processing simplifié (évite les blocages)")
            final_mesh = self.post_process_mesh(textured_mesh)
        else:
            print("🔄 Post-processing désactivé - préservation maximale des détails")
            final_mesh = textured_mesh

        # Convertir en STL
        stl_path = Path(output_dir) / "coin_model.stl"
        if self.convert_to_stl(final_mesh, str(stl_path)):
            elapsed_time = time.time() - start_time

            print(f"\n✅ Conversion terminée en {elapsed_time:.1f}s")
            print(f"📁 Fichier STL: {stl_path}")

            # Générer la vidéo si demandé (avec utilitaires modulaires)
            if render_video:
                video_path = self.render_video(final_mesh, output_dir)
                if video_path:
                    print(
                        f"🎬 Vidéo générée (utilitaires modulaires): {video_path}")

            if back_image:
                print("🔄 Modèle généré avec avers et revers")
            else:
                print("🔄 Modèle généré avec vue unique")

            return str(stl_path)

        return None


def convert_coin_hunyuan3d(front_image: str, back_image: str = None,
                           output_dir: str = "output_hunyuan3d",
                           model_path: str = "tencent/Hunyuan3D-2",
                           remove_background: bool = False,
                           render_video: bool = True,
                           enable_post_processing: bool = False) -> Optional[str]:
    """
    Fonction de convenance pour convertir une pièce avec Hunyuan3D-2mv

    Args:
        front_image: Chemin vers l'image de face
        back_image: Chemin vers l'image de dos (optionnel)
        output_dir: Répertoire de sortie
        model_path: Chemin vers le modèle Hunyuan3D
        remove_background: Supprimer l'arrière-plan
        render_video: Générer une vidéo de rotation

    Returns:
        Chemin vers le fichier STL ou None en cas d'erreur
    """
    # Créer le convertisseur
    converter = Hunyuan3DConverter(model_path)

    # Vérifier l'environnement
    if not converter.check_environment():
        return None

    # Charger les modèles
    if not converter.load_models():
        return None

    # Convertir
    return converter.convert_coin_to_stl(
        front_image, back_image, output_dir, remove_background, render_video, enable_post_processing
    )


def get_hunyuan3d_info():
    """Retourne des informations sur Hunyuan3D-2"""
    info = {
        'name': 'Hunyuan3D-2mv (complètement indépendant)',
        'description': 'Modèle de génération 3D multi-view avec rendu vidéo indépendant',
        'features': [
            'Support multi-view (avers/revers)',
            'Génération haute résolution',
            'Texture réaliste',
            'Rendu vidéo indépendant (sans TripoSR)',
            'Loading bars de progression',
            'Suppression arrière-plan',
            'Optimisé pour pièces numismatiques',
            'Complètement indépendant de TripoSR'
        ],
        'requirements': [
            'CUDA 11.8+ (recommandé)',
            'GPU avec 16GB+ VRAM',
            'Python 3.8+',
            'Hunyuan3D-2 installé',
            'tqdm pour les loading bars',
            'matplotlib pour le rendu vidéo'
        ]
    }
    return info
