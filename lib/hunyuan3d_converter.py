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

        # Configuration par défaut (niveau "high" - optimisé pour pièces)
        self.config = {
            'image_size': 1024,  # Résolution élevée pour capturer les détails fins
            'guidance_scale': 15.0,  # Élevé pour une meilleure forme
            'num_inference_steps': 100,  # Plus d'étapes pour plus de précision
            'octree_resolution': 380,  # Résolution mesh élevée pour détails
            'num_chunks': 20000,  # Complexité élevée
            'texture_guidance_scale': 6.0,  # Élevé pour les détails de texture
            'texture_steps': 60,  # Plus d'étapes pour les détails fins
            'seed': 12345,
            # Paramètres de rendu optimisés
            'n_views': 36,  # Diviseur de 360° pour rotation parfaite
            'elevation_deg': 15.0,  # Angle optimal pour capturer la profondeur
            'camera_distance': 1.5,  # Plus proche pour capturer les détails
            'fovy_deg': 30.0,  # Angle de vue serré pour réduire la distortion
            'height': 1024,  # Résolution élevée pour les détails
            'width': 1024,   # Résolution élevée pour les détails
            'fps': 30,
            'foreground_ratio': 0.95  # Ratio élevé pour capturer tout l'objet
        }

        # Session rembg pour la suppression d'arrière-plan
        self.rembg_session = None

        print(f"🔧 Convertisseur Hunyuan3D-2mv initialisé (utilitaires modulaires)")
        print(f"   Device: {self.device}")
        print(f"   Modèle forme: {self.model_path}")
        print(f"   Modèle texture: {self.texture_model_path}")

    def enable_test_mode(self):
        """Active le mode test ultra-rapide pour les tests et développement"""
        print("⚡ Activation du mode TEST ultra-rapide")
        print("   🚀 Résolution: 256x256 (vitesse maximale)")
        print("   🚀 Guidance scale: 2.0 (très minimal)")
        print("   🚀 Steps: 10 (ultra-rapide)")
        print("   🚀 Octree resolution: 64 (mesh très simple)")
        print("   🚀 Chunks: 1000 (complexité minimale)")
        print("   🚀 Texture steps: 8 (minimal)")
        print("   🚀 Rendus: 8 vues (au lieu de 36)")
        print("   ⚡ OPTIMISÉ POUR TESTS - PAS POUR PRODUCTION")

        # Configuration test ultra-rapide et agressive
        self.config = {
            # Paramètres de génération (ultra-rapides)
            'image_size': 256,  # Très petite résolution pour vitesse
            'guidance_scale': 2.0,  # Plus bas que 3.0
            'num_inference_steps': 10,  # Très peu d'étapes
            'octree_resolution': 64,  # NOUVEAU: Résolution mesh très basse
            'num_chunks': 1000,  # NOUVEAU: Complexité minimale
            'texture_guidance_scale': 1.5,  # Minimal pour texture
            'texture_steps': 8,  # Très peu d'étapes texture
            'seed': 42,
            # Paramètres de rendu (simplifiés)
            'n_views': 8,  # Seulement 8 vues au lieu de 36
            'elevation_deg': 0.0,  # Angle simple
            'camera_distance': 2.5,  # Distance normale
            'fovy_deg': 45.0,  # Angle standard
            'height': 256,  # Petite résolution rendu
            'width': 256,   # Petite résolution rendu
            'fps': 12,  # Moins de FPS
            'foreground_ratio': 0.8,
            # Mode test agressif
            'test_mode': True,
            'quick_render': True,
            'skip_post_processing': True,  # Éviter les traitements longs
            'low_precision': True,  # Utiliser une précision réduite
        }

        # Initialiser rembg rapidement si pas déjà fait
        if not self.rembg_session:
            try:
                import rembg
                self.rembg_session = rembg.new_session(
                    'u2net')  # Plus rapide que le défaut
                print("   ✅ Session rembg rapide initialisée")
            except ImportError:
                print("   ⚠️  rembg non disponible, suppression arrière-plan désactivée")

    def enable_debug_mode(self):
        """Active le mode debug ultra-minimal pour tests instantanés"""
        print("⚡ Activation du mode DEBUG (modèle lisse et simple)")
        print("   🚀 Résolution: 256x256 (minimal mais cohérent)")
        print("   🚀 Guidance scale: 3.0 (minimal mais forme reconnaissable)")
        print("   🚀 Steps: 15 (rapide mais évite les artefacts)")
        print("   🚀 Octree resolution: 96 (mesh simple mais lisse)")
        print("   🚀 Chunks: 1500 (complexité simple)")
        print("   🚀 Texture steps: 8 (minimal)")
        print("   🚀 Rendus: 8 vues seulement")
        print("   🚀 Mode flat: mesh lisse avec minimum de vertices")
        print("   ⚡ OPTIMISÉ POUR TESTS RAPIDES AVEC MODÈLE COHÉRENT")

        # Configuration debug équilibrée : rapide mais pas d'artefacts
        self.config = {
            # Paramètres de génération (rapides mais cohérents)
            'image_size': 256,  # Petite résolution mais pas trop
            'guidance_scale': 3.0,  # Assez pour une forme reconnaissable
            'num_inference_steps': 15,  # Suffisant pour éviter les artefacts
            'octree_resolution': 96,  # Résolution mesh simple mais lisse
            'num_chunks': 1500,  # Complexité simple mais suffisante
            'texture_guidance_scale': 2.0,  # Minimal mais fonctionnel
            'texture_steps': 8,  # Peu d'étapes texture
            'seed': 42,
            # Paramètres de rendu (simplifiés mais corrects)
            'n_views': 8,  # 8 vues suffisantes pour debug
            'elevation_deg': 5.0,  # Léger angle pour voir la forme
            'camera_distance': 2.0,  # Distance raisonnable
            'fovy_deg': 40.0,  # Angle standard
            'height': 256,  # Petite résolution rendu
            'width': 256,   # Petite résolution rendu
            'fps': 12,  # FPS réduit
            'foreground_ratio': 0.8,
            # Mode debug équilibré
            'debug_mode': True,
            'quick_render': True,
            'skip_post_processing': True,  # Éviter les traitements longs
            'simple_mesh': True,  # Mesh simple mais lisse
            'preserve_shape': True,  # Préserver la forme de base
            'minimal_vertices': True,  # Nombre minimal de vertices
        }

        # Pas besoin de rembg en mode debug
        print("   🚀 Suppression arrière-plan désactivée en mode debug")

    def enable_fast_mode(self):
        """Active le mode rapide (compromis qualité/vitesse)"""
        print("🏃 Activation du mode RAPIDE (compromis qualité/vitesse)")
        print("   ⚡ Résolution: 512x512 (qualité correcte)")
        print("   ⚡ Guidance scale: 7.0 (équilibré)")
        print("   ⚡ Steps: 50 (raisonnable)")
        print("   ⚡ Texture steps: 25 (équilibré)")
        print("   ⚡ Rendus: 24 vues")

        # Configuration rapide mais qualité correcte
        self.config = {
            'image_size': 512,  # Résolution intermédiaire
            'guidance_scale': 7.0,  # Équilibré
            'num_inference_steps': 50,  # Moitié du mode pièce
            'octree_resolution': 192,  # Résolution mesh réduite
            'num_chunks': 5000,  # Complexité réduite
            'texture_guidance_scale': 3.0,  # Équilibré
            'texture_steps': 25,  # Moitié du mode pièce
            'seed': 42,
            # Paramètres de rendu équilibrés
            'n_views': 24,  # 24 vues suffisantes
            'elevation_deg': 12.0,
            'camera_distance': 1.6,
            'fovy_deg': 35.0,
            'height': 512,
            'width': 512,
            'fps': 24,
            'foreground_ratio': 0.90,
            # Optimisations
            'fast_mode': True,
            'moderate_post_processing': True
        }

    def enable_ultra_mode(self):
        """Active le mode ultra qualité (paramètres maximaux)"""
        print("🌟 Activation du mode ULTRA qualité")
        print("   🎯 Résolution: 1024x1024 (haute définition)")
        print("   🎯 Guidance scale: 20.0 (précision maximale)")
        print("   🎯 Steps: 150 (qualité ultime)")
        print("   🎯 Texture steps: 80 (détails fins)")
        print("   🎯 Rendus: 48 vues (rendu premium)")
        print("   🌟 QUALITÉ MAXIMALE - TEMPS DE RENDU ÉLEVÉ")

        # Configuration ultra qualité
        self.config = {
            'image_size': 1024,  # Haute résolution
            'guidance_scale': 20.0,  # Très élevé pour précision maximale
            'num_inference_steps': 150,  # Beaucoup d'étapes
            'octree_resolution': 512,  # Résolution mesh très élevée
            'num_chunks': 30000,  # Complexité maximale
            'texture_guidance_scale': 8.0,  # Très élevé pour texture
            'texture_steps': 80,  # Beaucoup d'étapes texture
            'seed': 12345,
            # Paramètres de rendu premium
            'n_views': 48,  # Plus de vues pour plus de détails
            'elevation_deg': 20.0,  # Angle optimal
            'camera_distance': 1.4,  # Très proche pour détails
            'fovy_deg': 25.0,  # Angle serré
            'height': 1024,
            'width': 1024,
            'fps': 30,
            'foreground_ratio': 0.98,  # Ratio maximum
            # Optimisations qualité
            'ultra_mode': True,
            'max_post_processing': True,
            'anti_aliasing': True,
            'detail_preservation': True
        }

        # Initialiser rembg avec le meilleur modèle
        if not self.rembg_session:
            try:
                import rembg
                self.rembg_session = rembg.new_session(
                    'u2net')  # Meilleur modèle
                print("   ✅ Session rembg premium initialisée")
            except ImportError:
                print("   ⚠️  rembg non disponible, suppression arrière-plan désactivée")

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
                        octree_resolution=self.config['octree_resolution'],
                        num_chunks=self.config['num_chunks'],
                        generator=generator,
                        callback=callback,
                        callback_steps=1,
                        output_type='trimesh'
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
                        octree_resolution=self.config['octree_resolution'],
                        num_chunks=self.config['num_chunks'],
                        generator=generator,
                        callback=callback,
                        callback_steps=1,
                        output_type='trimesh'
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
        Applique une texture au mesh avec timer

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
            # Préparer les paramètres de texture
            texture_steps = self.config.get('texture_steps', 40)
            guidance_scale = self.config.get('texture_guidance_scale', 2.0)

            print(
                f"   🔄 Application de texture ({texture_steps} steps, guidance={guidance_scale})...")
            print("   ⏱️  Démarrage du timer...")

            # Démarrer le timer
            start_time = time.time()

            # Appeler le pipeline de texture sans callback
            textured_mesh = self.texture_pipeline(
                mesh,
                image=reference_image,
                guidance_scale=guidance_scale,
                num_inference_steps=texture_steps
            )

            # Calculer le temps écoulé
            elapsed_time = time.time() - start_time
            print(
                f"   ⏱️  Texture appliquée en {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
            print("   ✅ Texture appliquée avec succès")
            return textured_mesh

        except Exception as e:
            print(f"⚠️  Erreur application texture: {e}")
            print("   Retour au mesh sans texture")
            return mesh

    def apply_vertex_colors(self, mesh: trimesh.Trimesh, reference_image: Image.Image) -> trimesh.Trimesh:
        """
        Applique des couleurs de vertices rapides en échantillonnant les vraies couleurs de l'image

        Args:
            mesh: Mesh 3D de base
            reference_image: Image de référence pour les couleurs

        Returns:
            Mesh avec couleurs de vertices (rapide, sans texture)
        """
        print("🎨 Application de couleurs de vertices (mode rapide)...")

        try:
            start_time = time.time()

            # Convertir l'image en array numpy
            if reference_image.mode != 'RGB':
                reference_image = reference_image.convert('RGB')

            img_array = np.array(reference_image).astype(np.float32) / 255.0
            img_height, img_width = img_array.shape[:2]

            # Calculer les normales des vertices si pas disponibles
            if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
                mesh.compute_vertex_normals()

            # Projeter les vertices sur l'image 2D (vue frontale)
            vertices = mesh.vertices

            # Normaliser les coordonnées X,Y des vertices vers l'espace image [0,1]
            # On utilise X,Y pour projeter sur l'image (Z = profondeur)
            x_coords = vertices[:, 0]
            y_coords = vertices[:, 1]

            # Normaliser vers [0,1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            # Éviter division par zéro
            x_range = x_max - x_min if x_max != x_min else 1.0
            y_range = y_max - y_min if y_max != y_min else 1.0

            u_coords = (x_coords - x_min) / x_range
            v_coords = (y_coords - y_min) / y_range

            # Convertir vers coordonnées de pixels
            pixel_x = np.clip(u_coords * (img_width - 1),
                              0, img_width - 1).astype(int)
            pixel_y = np.clip(v_coords * (img_height - 1),
                              0, img_height - 1).astype(int)

            # Échantillonner les couleurs directement de l'image
            sampled_colors = img_array[pixel_y, pixel_x]

            # Ajouter un très léger effet de relief basé sur les normales (10% max)
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                # Calculer un facteur de relief basé sur la normale Z (face avant)
                relief_factor = np.abs(
                    mesh.vertex_normals[:, 2])  # 0=profil, 1=face
                # Pour broadcasting
                relief_factor = relief_factor.reshape(-1, 1)

                # Ajuster légèrement la luminosité selon le relief (±10%)
                brightness_adjustment = 1.0 + (relief_factor - 0.5) * 0.2
                final_colors = sampled_colors * brightness_adjustment
            else:
                final_colors = sampled_colors

            # S'assurer que les couleurs sont dans la plage [0,1]
            final_colors = np.clip(final_colors, 0.0, 1.0)

            # Appliquer les couleurs au mesh (format 0-255)
            mesh.visual.vertex_colors = (final_colors * 255).astype(np.uint8)

            elapsed_time = time.time() - start_time
            print(
                f"   ⏱️  Couleurs de vertices appliquées en {elapsed_time:.1f}s")
            print(
                f"   ✅ {len(mesh.vertices)} vertices colorés avec vraies couleurs de l'image")
            print(
                f"   🎯 Projection: {img_width}x{img_height} → {len(mesh.vertices)} vertices")

            return mesh

        except Exception as e:
            print(f"⚠️  Erreur application couleurs vertices: {e}")
            print("   Retour au mesh sans couleurs")
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
            debug_mode = self.config.get('debug_mode', False)

            # Préparer le mesh avec l'orientation standard
            oriented_mesh = mesh.copy()
            oriented_mesh = to_gradio_3d_orientation(oriented_mesh)

            # Normaliser seulement si pas en mode debug (pour économiser du temps)
            if not debug_mode:
                oriented_mesh = normalize_mesh(oriented_mesh)
            else:
                print("   ⚡ Mode DEBUG - normalisation désactivée pour vitesse")

            render_images = []

            if debug_mode:
                print(
                    f"   📹 Rendu DEBUG ultra-rapide: {n_views} vues minimalistes...")
            else:
                print(f"   📹 Rendu de {n_views} vues du mesh Hunyuan3D...")

            from tqdm import tqdm
            desc = "⚡ DEBUG ultra-rapide" if debug_mode else "🎬 Rendu mesh Hunyuan3D"
            with tqdm(total=n_views, desc=desc, unit="vue", colour="cyan") as pbar:
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
            # Post-traiter le mesh seulement si autorisé
            debug_mode = self.config.get('debug_mode', False)
            skip_post_processing = self.config.get(
                'skip_post_processing', False)

            if debug_mode:
                print("   ⚡ Mode DEBUG - export direct sans post-processing")
                processed_mesh = mesh
            elif skip_post_processing:
                print("   ⚡ Mode TEST - export direct sans post-processing")
                processed_mesh = mesh
            else:
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
                            enable_post_processing: bool = False,
                            use_vertex_colors: bool = False) -> Optional[str]:
        """
        Convertit une pièce (avers/revers) en STL avec utilitaires modulaires

        Args:
            front_image: Chemin vers l'image de face
            back_image: Chemin vers l'image de dos (optionnel)
            output_dir: Répertoire de sortie
            remove_background: Supprimer l'arrière-plan
            render_video: Générer une vidéo de rotation
            enable_post_processing: Activer le post-processing du mesh
            use_vertex_colors: Utiliser des couleurs de vertices rapides (2-5s au lieu de 8+ min)

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

        # Appliquer la texture ou les couleurs de vertices selon le mode
        if use_vertex_colors:
            # Mode vertex colors rapide (quelques secondes)
            colored_mesh = self.apply_vertex_colors(mesh, images[0])
            print(
                f"   📊 Mesh avec vertex colors: {len(colored_mesh.vertices)} vertices, {len(colored_mesh.faces)} faces")
        elif not self.disable_texture and self.texture_pipeline:
            # Mode texture complet (8+ minutes)
            colored_mesh = self.apply_texture(mesh, images[0])
            print(
                f"   📊 Mesh avec texture: {len(colored_mesh.vertices)} vertices, {len(colored_mesh.faces)} faces")
        else:
            # Mode sans couleur ni texture (ultra-rapide)
            colored_mesh = mesh
            print(
                f"   📊 Mesh sans couleur: {len(colored_mesh.vertices)} vertices, {len(colored_mesh.faces)} faces")

        # Post-traiter le mesh si activé
        skip_post_processing = self.config.get('skip_post_processing', False)
        debug_mode = self.config.get('debug_mode', False)

        if debug_mode:
            print("🔄 Mode DEBUG - aucun post-processing (mesh brut instantané)")
            print(
                f"   ⚡ Économie maximale: {len(colored_mesh.vertices)} vertices préservés")
            final_mesh = colored_mesh
        elif enable_post_processing and not skip_post_processing:
            print("🔄 Post-processing activé (peut ajouter des vertices)")
            final_mesh = self.post_process_mesh(colored_mesh)
        elif skip_post_processing:
            print("🔄 Post-processing désactivé en mode test (préserve le mesh)")
            print(f"   💡 Économie: pas d'ajout de vertices supplémentaires")
            final_mesh = colored_mesh
        else:
            print("🔄 Post-processing désactivé - préservation maximale des détails")
            final_mesh = colored_mesh

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
                           enable_post_processing: bool = False,
                           use_vertex_colors: bool = False) -> Optional[str]:
    """
    Fonction de convenance pour convertir une pièce avec Hunyuan3D-2mv

    Args:
        front_image: Chemin vers l'image de face
        back_image: Chemin vers l'image de dos (optionnel)
        output_dir: Répertoire de sortie
        model_path: Chemin vers le modèle Hunyuan3D
        remove_background: Supprimer l'arrière-plan
        render_video: Générer une vidéo de rotation
        enable_post_processing: Activer le post-processing du mesh
        use_vertex_colors: Utiliser des couleurs de vertices rapides (2-5s au lieu de 8+ min)

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
        front_image, back_image, output_dir, remove_background, render_video, enable_post_processing, use_vertex_colors
    )


def get_hunyuan3d_info():
    """Retourne des informations sur Hunyuan3D-2"""
    info = {
        'name': 'Hunyuan3D-2mv (complètement indépendant)',
        'description': 'Modèle de génération 3D multi-view avec rendu vidéo indépendant',
        'version': '2.0 (nettoyé)',
        'features': [
            'Support multi-view (avers/revers)',
            'Génération haute résolution',
            'Texture réaliste',
            'Rendu vidéo indépendant (sans TripoSR)',
            'Loading bars de progression',
            'Suppression arrière-plan',
            'Modes qualité: debug, low, medium, high, ultra',
            'Complètement indépendant de TripoSR'
        ],
        'utils': [
            'Utilitaires modulaires indépendants',
            'Rendu vidéo sans TripoSR',
            'Couleurs de vertices rapides',
            'Post-processing optimisé',
            'Configuration par niveaux de qualité'
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
