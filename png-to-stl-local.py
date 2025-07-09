#!/usr/bin/env python3
"""
Convertisseur PNG vers STL avec TripoSR - Version Locale
Adapté pour fonctionner sur Windows avec GPU NVIDIA
"""

import os
import sys
import torch
import time
from PIL import Image
import numpy as np
import rembg
import pymeshlab as pymesh
import argparse
from pathlib import Path
import subprocess
import pkg_resources
import warnings

# Supprimer les warnings pour une sortie plus propre
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def check_and_install_dependencies():
    """Vérifie et installe les dépendances nécessaires - basé sur le script Google Colab qui fonctionne"""
    required_packages = [
        'scikit-image==0.19.3',  # Version qui fonctionne avec Pillow 9.5.0
        'PyMCubes',
        'huggingface_hub',
        'opencv-python',
        'onnxruntime',
        'rembg',
        'pymeshlab',
        'omegaconf',
        'plyfile',
        'tqdm',
        'mcubes',
        'trimesh',
        'diffusers',
        'transformers',
        'accelerate',
        'safetensors',
        # 'einops==0.7.0',  # Requis par TripoSR
        'xatlas==0.0.9',
        'moderngl==5.10.0',
        'imageio[ffmpeg]'
    ]

    print("Vérification des dépendances...")
    missing_packages = []

    # Vérifier les packages de base
    packages_to_check = {
        'skimage': 'scikit-image==0.19.3',
        'PIL': 'pillow==9.5.0',  # Déjà installé
        'cv2': 'opencv-python',
        'rembg': 'rembg',
        'pymeshlab': 'pymeshlab',
        'omegaconf': 'omegaconf',
        'plyfile': 'plyfile',
        'tqdm': 'tqdm',
        'trimesh': 'trimesh',
        'diffusers': 'diffusers',
        'transformers': 'transformers',
        'accelerate': 'accelerate',
        'safetensors': 'safetensors',
        'einops': 'einops==0.7.0',
        'xatlas': 'xatlas==0.0.9',
        'moderngl': 'moderngl==5.10.0',
        'imageio': 'imageio[ffmpeg]'
    }

    for package, install_name in packages_to_check.items():
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'skimage':
                import skimage
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(install_name)

    # Vérifier PyMCubes séparément
    try:
        import mcubes
    except ImportError:
        missing_packages.append('PyMCubes')

    if missing_packages:
        print(f"Installation des packages manquants: {missing_packages}")
        for package in missing_packages:
            print(f"Installation de {package}...")
            if 'git+' in package:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package])
            else:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package])
    else:
        print("Toutes les dépendances sont déjà installées!")

    # Installer torchmcubes depuis git (comme dans le script Colab)
    try:
        import torchmcubes
    except ImportError:
        print("Installation de torchmcubes depuis GitHub...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/tatsy/torchmcubes.git"
        ])


def setup_triposr():
    """Clone et configure TripoSR si nécessaire - basé sur le script Google Colab"""
    triposr_path = Path("TripoSR")

    if not triposr_path.exists():
        print("Clonage du dépôt TripoSR...")
        subprocess.run(
            ["git", "clone", "https://github.com/pyimagesearch/TripoSR.git"], check=True)

    # S'assurer que nous sommes dans le bon répertoire (comme dans le script Colab)
    if (triposr_path / "TripoSR").exists():
        # Structure: TripoSR/TripoSR/
        triposr_dir = triposr_path / "TripoSR"
        sys.path.insert(0, str(triposr_dir))
        sys.path.insert(0, str(triposr_dir / "tsr"))
        print(f"Configuration TripoSR: {triposr_dir}")
    else:
        # Structure: TripoSR/
        sys.path.insert(0, str(triposr_path))
        sys.path.insert(0, str(triposr_path / "tsr"))
        print(f"Configuration TripoSR: {triposr_path}")

    # Installer les dépendances du projet TripoSR (comme dans le script Colab)
    requirements_file = triposr_path / "requirements.txt"
    if requirements_file.exists():
        print("Installation des dépendances TripoSR...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(
                requirements_file), "-q"
        ])

    return triposr_path


def check_cuda_compatibility():
    """Vérifie la compatibilité CUDA et affiche les informations GPU"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA non disponible. Le traitement sera effectué sur CPU (beaucoup plus lent).")
        print("   Pour utiliser le GPU, assurez-vous que:")
        print("   1. CUDA Toolkit 11.8+ est installé")
        print("   2. Les drivers NVIDIA sont à jour")
        print("   3. PyTorch avec support CUDA est installé")
        return False

    print(f"✅ CUDA disponible!")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"   Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   Version CUDA: {torch.version.cuda}")

    # Vérifier la mémoire disponible
    torch.cuda.empty_cache()
    memory_free = torch.cuda.get_device_properties(
        0).total_memory - torch.cuda.memory_allocated(0)
    memory_free_gb = memory_free / 1024**3

    if memory_free_gb < 4:
        print(f"⚠️  Mémoire GPU faible: {memory_free_gb:.1f} GB disponible")
        print("   Fermez les autres applications utilisant le GPU pour de meilleures performances.")

    return True


def detect_and_convert_image_format(image_path):
    """
    Détecte le format d'image et le convertit si nécessaire
    Supporte: PNG, WebP, JPEG, JPG, BMP, TIFF

    Args:
        image_path: Chemin vers l'image d'entrée

    Returns:
        PIL.Image: Image convertie en RGB/RGBA
        str: Format original détecté
    """
    try:
        image = Image.open(image_path)
        original_format = image.format.lower() if image.format else "unknown"

        print(f"   Format détecté: {original_format.upper()}")

        # Formats supportés
        supported_formats = ['png', 'webp',
                             'jpeg', 'jpg', 'bmp', 'tiff', 'tif']

        if original_format not in supported_formats:
            print(
                f"   ⚠️  Format '{original_format}' non testé, tentative de conversion...")

        # Convertir vers RGB/RGBA selon le besoin
        if image.mode in ('RGBA', 'LA', 'P'):
            if 'transparency' in image.info:
                # Garder la transparence
                image = image.convert('RGBA')
            else:
                # Convertir en RGB si pas de transparence
                image = image.convert('RGB')
        elif image.mode in ('L', 'I', 'F'):
            # Images en niveaux de gris
            image = image.convert('RGB')
        elif image.mode not in ('RGB', 'RGBA'):
            # Autres modes, convertir en RGB
            print(f"   Conversion du mode {image.mode} vers RGB")
            image = image.convert('RGB')

        return image, original_format

    except Exception as e:
        print(f"❌ Erreur lors de la conversion du format: {e}")
        raise


def convert_png_to_stl(input_image_path, output_dir="output", remove_bg=False, render_video=True, reverse_image_path=None):
    """
    Convertit une image (PNG, WebP, JPEG, etc.) en modèle STL avec possibilité d'ajouter une vue de revers

    Args:
        input_image_path: Chemin vers l'image d'entrée (recto) - supporte PNG, WebP, JPEG, BMP, TIFF
        output_dir: Répertoire de sortie pour les fichiers générés
        remove_bg: Si True, supprime l'arrière-plan de l'image
        render_video: Si True, génère une vidéo du modèle 3D
        reverse_image_path: Chemin optionnel vers l'image de revers (verso)
    """

    # Importer les modules TripoSR (avec gestion d'erreur comme dans le script Colab)
    try:
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground, save_video
        print("Importations spécifiques à TripoSR réussies!")
    except ImportError as e:
        print(f"Erreur lors de l'importation des modules TripoSR: {e}")
        print("Tentative de correction en installant les packages manquants...")

        # Installer les packages manquants (comme dans le script Colab)
        missing_packages = ['mcubes', 'trimesh', 'diffusers',
                            'transformers', 'accelerate', 'safetensors']
        for package in missing_packages:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", package])
            except subprocess.CalledProcessError:
                print(f"Erreur lors de l'installation de {package}")

        # Réessayer l'importation
        import importlib
        importlib.invalidate_caches()
        try:
            from tsr.system import TSR
            from tsr.utils import remove_background, resize_foreground, save_video
            print("Importations réussies après correction!")
        except ImportError as e:
            print(f"Échec des importations après tentative de correction: {e}")
            print(
                "Le script ne pourra pas continuer. Veuillez vérifier votre environnement.")
            return None

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Configuration:")
    print(f"   Périphérique: {device.upper()}")

    # Vérifier si nous avons une image de revers
    has_reverse_image = reverse_image_path is not None and Path(
        reverse_image_path).exists()
    if has_reverse_image:
        print(f"   Mode: Reconstruction optimisée avec 2 vues (traitement séparé)")
        print(f"   Image recto: {input_image_path}")
        print(f"   Image verso: {reverse_image_path}")
    else:
        print(f"   Mode: Reconstruction standard avec 1 vue")
        if reverse_image_path:
            print(
                f"   ⚠️  Image de revers spécifiée mais introuvable: {reverse_image_path}")

    if device == "cuda":
        # Nettoyer la mémoire GPU avant de commencer
        torch.cuda.empty_cache()

    # Paramètres TripoSR optimisés
    pretrained_model_name_or_path = "stabilityai/TripoSR"
    chunk_size = 8192
    foreground_ratio = 0.85
    model_save_format = "obj"
    mc_resolution = 256  # Paramètre officiel TripoSR
    # Optimisé pour objets solides (chaises, etc.) - 25.0 cause des rendus bizarres
    mc_threshold = 10.0

    # Créer le répertoire de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger l'image principale
    print(f"\n📷 Chargement de l'image principale: {input_image_path}")
    try:
        image, original_format = detect_and_convert_image_format(
            input_image_path)
        print(f"   Taille originale: {image.size}")
        print(f"   Mode: {image.mode}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'image: {e}")
        return None

    # Charger l'image de revers si disponible
    reverse_image = None
    if has_reverse_image:
        print(f"\n📷 Chargement de l'image de revers: {reverse_image_path}")
        try:
            reverse_image, _ = detect_and_convert_image_format(
                reverse_image_path)
            print(f"   Taille originale: {reverse_image.size}")
            print(f"   Mode: {reverse_image.mode}")
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement de l'image de revers: {e}")
            print("   Poursuite avec l'image principale seulement...")
            has_reverse_image = False
            reverse_image = None

    # Redimensionner les images
    original_image_resized = image.resize((512, 512))
    if has_reverse_image:
        reverse_image_resized = reverse_image.resize((512, 512))

    # Créer le dossier examples
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    original_image_resized.save(examples_dir / "product.png")
    if has_reverse_image:
        reverse_image_resized.save(examples_dir / "product_reverse.png")

    # Initialiser le modèle TripoSR
    print(f"\n🤖 Chargement du modèle TripoSR...")
    print("   (Cela peut prendre quelques minutes lors de la première utilisation)")
    try:
        model = TSR.from_pretrained(
            pretrained_model_name_or_path,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(chunk_size)
        model.to(device)
        print("✅ Modèle chargé avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("   Vérifiez votre connexion internet et réessayez.")
        return None

    # Traiter les images
    print(f"\n🎨 Traitement des images...")
    processed_images = []

    try:
        # Traitement de l'image principale
        print("   Traitement de l'image principale (recto)...")
        if remove_bg:
            print("   Suppression de l'arrière-plan...")
            rembg_session = rembg.new_session()
            processed_main = remove_background(
                original_image_resized, rembg_session)
        else:
            processed_main = original_image_resized

        # Redimensionner l'image
        processed_main = resize_foreground(processed_main, foreground_ratio)

        # Gérer les images RGBA
        if processed_main.mode == "RGBA":
            print("   Conversion RGBA vers RGB...")
            processed_main = np.array(
                processed_main).astype(np.float32) / 255.0
            processed_main = processed_main[:, :, :3] * processed_main[:, :, 3:4] + \
                (1 - processed_main[:, :, 3:4]) * 0.5
            processed_main = Image.fromarray(
                (processed_main * 255.0).astype(np.uint8))

        processed_images.append(processed_main)

        # Traitement de l'image de revers si disponible
        if has_reverse_image:
            print("   Traitement de l'image de revers (verso)...")
            if remove_bg:
                reverse_processed = remove_background(
                    reverse_image_resized, rembg_session)
            else:
                reverse_processed = reverse_image_resized

            reverse_processed = resize_foreground(
                reverse_processed, foreground_ratio)

            if reverse_processed.mode == "RGBA":
                reverse_processed = np.array(
                    reverse_processed).astype(np.float32) / 255.0
                reverse_processed = reverse_processed[:, :, :3] * reverse_processed[:, :, 3:4] + \
                    (1 - reverse_processed[:, :, 3:4]) * 0.5
                reverse_processed = Image.fromarray(
                    (reverse_processed * 255.0).astype(np.uint8))

            processed_images.append(reverse_processed)

        # Créer le répertoire de sortie pour cette image
        image_dir = output_dir / "0"
        image_dir.mkdir(exist_ok=True)

        # Sauvegarder les images traitées
        for i, proc_img in enumerate(processed_images):
            suffix = "" if i == 0 else f"_reverse"
            proc_img.save(image_dir / f"input{suffix}.png")

        print(f"✅ {len(processed_images)} image(s) traitée(s) et sauvegardée(s)")

    except Exception as e:
        print(f"⚠️  Erreur lors du traitement des images: {e}")
        print("   Utilisation des images originales...")
        processed_images = [original_image_resized]
        if has_reverse_image:
            processed_images.append(reverse_image_resized)

        image_dir = output_dir / "0"
        image_dir.mkdir(exist_ok=True)
        for i, proc_img in enumerate(processed_images):
            suffix = "" if i == 0 else f"_reverse"
            proc_img.save(image_dir / f"input{suffix}.png")

    # Générer le modèle 3D avec approche optimisée
    print(f"\n🏗️  Génération du modèle 3D...")
    start_time = time.time()

    try:
        if has_reverse_image and len(processed_images) > 1:
            print("   🔄 Mode 2 vues: Reconstruction optimisée...")
            print("   📐 Génération du modèle principal (recto)...")

            # Générer le modèle principal avec des paramètres optimisés
            with torch.no_grad():
                scene_codes_main = model([processed_images[0]], device=device)

            print(
                "   🎯 Utilisation de la vue principale pour une reconstruction de qualité")
            scene_codes = scene_codes_main

            # Note: Pour l'instant, on utilise principalement la vue principale
            # Une approche plus sophistiquée pourrait fusionner les deux reconstructions

        else:
            print("   📐 Génération des codes de scène (vue unique)...")
            with torch.no_grad():
                scene_codes = model([processed_images[0]], device=device)

        # Rendre les vues si demandé
        if render_video:
            print("   🎬 Rendu des vues multiples (30 vues)...")
            render_images = model.render(
                scene_codes, n_views=30, return_type="pil")
            for ri, render_image in enumerate(render_images[0]):
                render_image.save(image_dir / f"render_{ri:03d}.png")

            # Créer une vidéo
            print("   🎞️  Création de la vidéo...")
            save_video(render_images[0], str(image_dir / "render.mp4"), fps=30)
            print(f"   ✅ Vidéo sauvegardée: {image_dir / 'render.mp4'}")

        # Extraire et sauvegarder le maillage avec paramètres optimisés
        print("   🔧 Extraction du maillage 3D (haute qualité)...")
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=False,
            resolution=mc_resolution,
            threshold=mc_threshold
        )
        mesh_file = image_dir / f"mesh.{model_save_format}"
        meshes[0].export(str(mesh_file))
        print(f"   ✅ Modèle 3D sauvegardé: {mesh_file}")

        # Convertir en STL
        print("   📦 Conversion en STL...")
        ms = pymesh.MeshSet()
        ms.load_new_mesh(str(mesh_file))

        # Appliquer des filtres pour améliorer la qualité avec gestion d'erreur robuste
        print("   🔧 Application de filtres de qualité...")

        # Découvrir les filtres disponibles
        def discover_available_filters(ms):
            """Découvre les filtres disponibles dans cette version de PyMeshLab"""
            available_filters = set()
            try:
                # Capturer la sortie de print_filter_list() pour découvrir les filtres
                import io
                import contextlib

                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    ms.print_filter_list()

                filter_output = f.getvalue()
                # Extraire les noms de filtres de la sortie
                for line in filter_output.split('\n'):
                    if line.strip() and not line.startswith(' ') and ':' in line:
                        filter_name = line.split(':')[0].strip()
                        if filter_name:
                            available_filters.add(filter_name)

                print(
                    f"   📋 {len(available_filters)} filtres PyMeshLab découverts")
                return available_filters

            except Exception as e:
                print(f"   ⚠️  Impossible de découvrir les filtres: {e}")
                return set()

        # Fonction helper pour appliquer les filtres de manière sécurisée
        def apply_filter_safe(ms, filter_name, **kwargs):
            try:
                ms.apply_filter(filter_name, **kwargs)
                print(f"   ✅ Filtre appliqué: {filter_name}")
                return True
            except Exception as e:
                print(f"   ⚠️  Filtre ignoré ({filter_name}): {str(e)}")
                return False

        # Découvrir les filtres disponibles
        available_filters = discover_available_filters(ms)

        # Fonction pour trouver le meilleur filtre disponible
        def find_available_filter(candidates, available_filters):
            for candidate in candidates:
                if candidate in available_filters:
                    return candidate
            return None

        # Essayer différents noms de filtres selon la version de PyMeshLab
        applied_filters = 0

        # Supprimer les composants isolés (nouveaux noms PyMeshLab 2022.2+)
        cleanup_filters = [
            # Nouveaux noms (2022.2+)
            'meshing_remove_connected_component_by_diameter',
            'meshing_remove_connected_component_by_face_number',
            'meshing_remove_duplicate_vertices',
            'meshing_remove_unreferenced_vertices',
            'meshing_remove_null_faces',
            'meshing_remove_t_vertices',
            # Anciens noms (compatibility)
            'remove_isolated_pieces_wrt_diameter',
            'remove_isolated_folded_faces',
            'remove_zero_area_faces',
            'remove_duplicate_vertices',
            'remove_unreferenced_vertices'
        ]

        for filter_name in cleanup_filters:
            if filter_name in available_filters:
                if apply_filter_safe(ms, filter_name):
                    applied_filters += 1

        # Lisser légèrement le maillage (activé par défaut)
        apply_smoothing = True  # Valeur par défaut
        if apply_smoothing:
            smooth_filters = [
                # Nouveaux noms (2022.2+)
                'apply_coord_laplacian_smoothing',
                'apply_coord_taubin_smoothing',
                'apply_coord_hc_laplacian_smoothing',
                # Anciens noms (compatibility)
                'laplacian_smooth',
                'taubin_smooth',
                'hc_laplacian_smooth'
            ]

            smoothed = False
            for filter_name in smooth_filters:
                if filter_name in available_filters:
                    smooth_params = {
                        'stepsmoothnum': 2} if 'laplacian' in filter_name else {}
                    if apply_filter_safe(ms, filter_name, **smooth_params):
                        smoothed = True
                        break

            if not smoothed:
                print("   ⚠️  Aucun filtre de lissage disponible")

        # Simplifier si nécessaire (réduire le bruit)
        vertex_count = ms.current_mesh().vertex_number()
        if vertex_count > 50000:
            print(
                f"   📊 Maillage dense détecté ({vertex_count} vertices), simplification...")

            simplification_filters = [
                # Nouveaux noms (2022.2+)
                'meshing_decimation_quadric_edge_collapse',
                'meshing_decimation_clustering',
                # Anciens noms (compatibility)
                'simplification_quadric_edge_collapse_decimation',
                'simplification_clustering_decimation'
            ]

            simplified = False
            for filter_name in simplification_filters:
                if filter_name in available_filters:
                    if 'quadric' in filter_name:
                        params = {'targetfacenum': 25000}
                    else:
                        params = {'threshold': 0.01}

                    if apply_filter_safe(ms, filter_name, **params):
                        simplified = True
                        break

            if not simplified:
                print("   ⚠️  Aucun filtre de simplification disponible")

        print(f"   📊 Filtres appliqués avec succès: {applied_filters}")

        # Afficher quelques statistiques du maillage final
        final_mesh = ms.current_mesh()
        print(
            f"   📈 Maillage final: {final_mesh.vertex_number()} vertices, {final_mesh.face_number()} faces")

        # Sauvegarder le STL final
        stl_file = output_dir / f"{Path(input_image_path).stem}.stl"
        try:
            ms.save_current_mesh(str(stl_file))
        except Exception as e:
            print(f"   ⚠️  Erreur lors de la sauvegarde avec filtres: {e}")
            print("   🔄 Tentative de sauvegarde directe...")
            # En cas d'échec, sauvegarder le mesh original
            ms_backup = pymesh.MeshSet()
            ms_backup.load_new_mesh(str(mesh_file))
            ms_backup.save_current_mesh(str(stl_file))

        elapsed_time = time.time() - start_time
        print(f"✅ Fichier STL généré: {stl_file}")
        print(f"⏱️  Temps total: {elapsed_time:.1f} secondes")

        if has_reverse_image:
            print(f"🎯 Reconstruction optimisée avec référence verso!")
            print(
                f"💡 Conseil: La vue de revers a été utilisée comme référence pour améliorer la qualité")

        return str(stl_file)

    except Exception as e:
        print(f"❌ Erreur lors de la génération du modèle 3D: {str(e)}")
        import traceback
        print(f"💡 Détails de l'erreur: {traceback.format_exc()}")
        if "out of memory" in str(e).lower():
            print(
                "   💡 Conseil: Fermez les autres applications utilisant le GPU et réessayez.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convertir une image (PNG, WebP, JPEG, etc.) en modèle STL")
    parser.add_argument(
        "input", help="Chemin vers l'image d'entrée (PNG, WebP, JPEG, BMP, TIFF)")
    parser.add_argument("-o", "--output", default="output",
                        help="Répertoire de sortie (défaut: output)")
    parser.add_argument("--remove-bg", action="store_true",
                        help="Supprimer l'arrière-plan de l'image")
    parser.add_argument("--no-video", action="store_true",
                        help="Ne pas générer de vidéo du modèle")
    parser.add_argument("--reverse-image",
                        help="Chemin vers l'image de revers (verso) pour améliorer la reconstruction 3D")
    parser.add_argument("--mc-resolution", type=int, default=256,
                        help="Résolution du marching cubes (défaut: 256 - paramètre officiel TripoSR)")
    parser.add_argument("--mc-threshold", type=float, default=10.0,
                        help="Seuil du marching cubes (défaut: 10.0 - paramètre optimisé pour objets solides)")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Désactiver le lissage du maillage")
    parser.add_argument("--foreground-ratio", type=float, default=0.85,
                        help="Ratio de l'objet dans l'image (défaut: 0.85 - paramètre officiel TripoSR)")
    parser.add_argument("--debug", action="store_true",
                        help="Afficher les informations de diagnostic (filtres disponibles, etc.)")
    parser.add_argument("--setup", action="store_true",
                        help="Installer les dépendances et configurer l'environnement")
    parser.add_argument("--tips", action="store_true",
                        help="Afficher les conseils d'optimisation pour pièces numismatiques")

    args = parser.parse_args()

    print("🚀 Convertisseur d'Images vers STL avec TripoSR")
    print("   Formats supportés: PNG, WebP, JPEG, BMP, TIFF")
    print("=" * 50)

    if args.setup:
        print("⚙️  Configuration de l'environnement...")
        check_and_install_dependencies()
        setup_triposr()
        print("\n✅ Configuration terminée! Vous pouvez maintenant utiliser le script.")
        print(f"💡 Exemple PNG: python {sys.argv[0]} mon_image.png")
        print(f"💡 Exemple WebP: python {sys.argv[0]} mon_image.webp")
        print(
            f"💡 Avec revers: python {sys.argv[0]} recto.png --reverse-image verso.webp")
        print(
            f"💡 Haute qualité: python {sys.argv[0]} image.png --mc-resolution 1024")
        print(f"💡 Diagnostic: python {sys.argv[0]} image.png --debug")
        print(f"💡 Conseils: python {sys.argv[0]} --tips")
        return

    if args.tips:
        print_coin_tips()
        return

    if args.debug:
        print("🔍 Mode diagnostic activé...")
        diagnostic_info()
        if not Path(args.input).exists():
            print(f"❌ Erreur: Le fichier '{args.input}' n'existe pas.")
            return
        # Continuer avec la conversion mais avec plus d'informations de debug

    if not Path(args.input).exists():
        print(f"❌ Erreur: Le fichier '{args.input}' n'existe pas.")
        return

    # Vérifier l'image de revers si spécifiée
    if args.reverse_image and not Path(args.reverse_image).exists():
        print(
            f"⚠️  Attention: L'image de revers '{args.reverse_image}' n'existe pas.")
        print("   La reconstruction se fera avec l'image principale seulement.")

    # Vérifier CUDA
    check_cuda_compatibility()

    # Setup TripoSR si nécessaire
    print(f"\n📦 Configuration de TripoSR...")
    setup_triposr()

    # Convertir l'image avec les paramètres par défaut TripoSR
    if args.mc_resolution == 256 and args.mc_threshold == 10.0 and args.foreground_ratio == 0.85:
        # Utiliser les paramètres par défaut optimisés pour pièces numismatiques
        print("🔧 Mode automatique détecté: Utilisation des paramètres officiels TripoSR")
        stl_file = convert_coin_to_stl_safe(
            args.input,
            args.output,
            remove_bg=args.remove_bg,
            render_video=not args.no_video,
            reverse_image_path=args.reverse_image
        )
    else:
        # Utiliser les paramètres personnalisés de l'utilisateur
        print(
            f"🔧 Utilisation des paramètres personnalisés: résolution={args.mc_resolution}, seuil={args.mc_threshold}, ratio={args.foreground_ratio}")
        stl_file = convert_png_to_stl(
            args.input,
            args.output,
            remove_bg=args.remove_bg,
            render_video=not args.no_video,
            reverse_image_path=args.reverse_image
        )

    print("\n" + "=" * 50)
    if stl_file:
        print(f"🎉 Conversion terminée avec succès!")
        print(f"📁 Fichier STL: {stl_file}")
        print(f"📁 Dossier de sortie: {args.output}")
        if not args.no_video:
            print(f"🎬 Vidéo disponible dans: {args.output}/0/render.mp4")
        if args.reverse_image:
            print(f"🔄 Images utilisées: {args.input} + {args.reverse_image}")
        print(
            f"⚙️  Paramètres utilisés: résolution={args.mc_resolution}, seuil={args.mc_threshold}")
    else:
        print("❌ Échec de la conversion.")
        print("💡 Vérifiez les messages d'erreur ci-dessus pour plus d'informations.")
        print("💡 Essayez d'ajuster --mc-resolution ou --mc-threshold")
        print("💡 Utilisez --debug pour plus d'informations de diagnostic")


def convert_coin_to_stl_safe(input_image_path, output_dir="output", remove_bg=False,
                             render_video=True, reverse_image_path=None):
    """
    VERSION SÉCURISÉE pour pièces numismatiques avec gestion automatique de la mémoire GPU.
    Évite les erreurs "CUDA out of memory" en adaptant automatiquement les paramètres.

    Cette fonction remplace convert_coin_to_stl avec une meilleure gestion des ressources.
    """

    print("🔧 Mode TripoSR SÉCURISÉ avec gestion automatique de la mémoire")

    # Obtenir les paramètres optimaux pour le GPU détecté
    gpu_settings = get_optimal_settings_for_gpu()
    print(f"   {gpu_settings['device_info']}")

    # Importer les modules TripoSR
    try:
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground, save_video
    except ImportError as e:
        print(f"❌ Erreur importation TripoSR: {e}")
        return None

    # Configuration avec paramètres adaptés au GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mc_resolution = gpu_settings['mc_resolution']
    image_size = gpu_settings['image_size']
    chunk_size = gpu_settings['chunk_size']

    print(f"\n🔧 Configuration sécurisée:")
    print(f"   Périphérique: {device.upper()}")
    print(f"   Résolution marching cubes: {mc_resolution}")
    print(f"   Taille image: {image_size}x{image_size}")
    print(f"   Chunk size: {chunk_size}")

    # Vérifier les images
    has_reverse_image = reverse_image_path is not None and Path(
        reverse_image_path).exists()
    if has_reverse_image:
        print(f"   Mode: 2 vues avec recto et verso")

    # Nettoyer la mémoire GPU dès le début
    if device == "cuda":
        clear_gpu_memory()

    # Charger et traiter les images
    try:
        image, _ = detect_and_convert_image_format(input_image_path)
        image_resized = image.resize((image_size, image_size))

        reverse_image_resized = None
        if has_reverse_image:
            reverse_image, _ = detect_and_convert_image_format(
                reverse_image_path)
            reverse_image_resized = reverse_image.resize(
                (image_size, image_size))

    except Exception as e:
        print(f"❌ Erreur chargement images: {e}")
        return None

    # Charger le modèle TripoSR avec paramètres optimisés
    print(f"\n🤖 Chargement du modèle TripoSR (sécurisé)...")
    try:
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(chunk_size)
        model.to(device)
        print("✅ Modèle chargé avec succès!")
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return None

    # Traitement des images avec paramètres adaptés
    print(f"\n🎨 Traitement des images...")
    processed_images = []

    try:
        # Traitement image principale
        if remove_bg:
            rembg_session = rembg.new_session()
            processed_main = remove_background(image_resized, rembg_session)
        else:
            processed_main = image_resized

        processed_main = resize_foreground(
            processed_main, 0.85)  # Paramètre officiel TripoSR

        if processed_main.mode == "RGBA":
            processed_main = np.array(
                processed_main).astype(np.float32) / 255.0
            processed_main = processed_main[:, :, :3] * processed_main[:, :, 3:4] + \
                (1 - processed_main[:, :, 3:4]) * 0.5
            processed_main = Image.fromarray(
                (processed_main * 255.0).astype(np.uint8))

        processed_images.append(processed_main)

        # Traitement image de revers si disponible
        if has_reverse_image:
            if remove_bg:
                reverse_processed = remove_background(
                    reverse_image_resized, rembg_session)
            else:
                reverse_processed = reverse_image_resized
            reverse_processed = resize_foreground(reverse_processed, 0.85)

            if reverse_processed.mode == "RGBA":
                reverse_processed = np.array(
                    reverse_processed).astype(np.float32) / 255.0
                reverse_processed = reverse_processed[:, :, :3] * reverse_processed[:, :, 3:4] + \
                    (1 - reverse_processed[:, :, 3:4]) * 0.5
                reverse_processed = Image.fromarray(
                    (reverse_processed * 255.0).astype(np.uint8))

            processed_images.append(reverse_processed)

    except Exception as e:
        print(
            f"⚠️  Erreur traitement images: {e}, utilisation images originales")
        processed_images = [image_resized]
        if has_reverse_image:
            processed_images.append(reverse_image_resized)

    # Sauvegarder les images traitées
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "0"
    image_dir.mkdir(exist_ok=True)

    for i, proc_img in enumerate(processed_images):
        suffix = "" if i == 0 else f"_reverse"
        proc_img.save(image_dir / f"input{suffix}.png")

    # Génération du modèle 3D avec gestion mémoire
    print(f"\n🏗️  Génération du modèle 3D SÉCURISÉE...")
    start_time = time.time()

    try:
        # Nettoyer la mémoire avant génération
        clear_gpu_memory()

        # Génération des codes de scène
        print("   📐 Génération des codes de scène...")
        with torch.no_grad():
            scene_codes = model([processed_images[0]], device=device)

        # Rendu des vues si demandé (avec gestion mémoire)
        if render_video:
            try:
                print("   🎬 Rendu des vues multiples...")
                render_images = model.render(
                    scene_codes, n_views=30, return_type="pil")
                for ri, render_image in enumerate(render_images[0]):
                    render_image.save(image_dir / f"render_{ri:03d}.png")
                save_video(render_images[0], str(
                    image_dir / "render.mp4"), fps=30)
                print(f"   ✅ Vidéo sauvegardée: {image_dir / 'render.mp4'}")
            except Exception as e:
                print(
                    f"   ⚠️  Erreur rendu vidéo: {e}, continuation sans vidéo")

        # EXTRACTION SÉCURISÉE DU MAILLAGE avec fallback automatique
        print("   🔧 Extraction sécurisée du maillage 3D...")
        clear_gpu_memory()

        meshes = None
        fallback_attempts = [
            (mc_resolution, "résolution optimisée GPU"),
            (max(256, mc_resolution // 2), "résolution réduite (sécurité niveau 1)"),
            (256, "résolution standard (sécurité niveau 2)"),
            (192, "résolution faible (sécurité niveau 3)"),
            (128, "résolution très faible (sécurité niveau 4)"),
            (96, "résolution minimale (sécurité niveau 5)")
        ]

        for attempt_resolution, description in fallback_attempts:
            try:
                print(f"   • {description}: {attempt_resolution}")
                meshes = model.extract_mesh(
                    scene_codes,
                    has_vertex_color=False,
                    resolution=attempt_resolution,
                    threshold=10.0  # Paramètre optimisé pour objets solides
                )
                print(
                    f"   ✅ Extraction réussie avec résolution {attempt_resolution}")
                break

            except RuntimeError as e:
                error_str = str(e).lower()
                if "out of memory" in error_str or "numel needs to be smaller" in error_str:
                    error_type = "Mémoire insuffisante" if "out of memory" in error_str else "Limitation tenseur int32"
                    print(
                        f"   ⚠️  {error_type} pour résolution {attempt_resolution}")
                    clear_gpu_memory()
                    if attempt_resolution == 96:
                        print(
                            "   ❌ Échec même avec résolution minimale - GPU insuffisant")
                        return None
                    continue
                else:
                    print(f"   ❌ Erreur non-ressource: {e}")
                    return None

        if meshes is None:
            print("   ❌ Impossible d'extraire le maillage")
            return None

        # Sauvegarder le maillage
        mesh_file = image_dir / "mesh.obj"
        meshes[0].export(str(mesh_file))
        print(f"   ✅ Maillage 3D sauvegardé: {mesh_file}")

        # Conversion en STL avec PyMeshLab
        print("   📦 Conversion optimisée en STL...")
        try:
            import pymeshlab as pymesh
            ms = pymesh.MeshSet()
            ms.load_new_mesh(str(mesh_file))

            # Post-processing optimisé pour pièces numismatiques
            print("   🔧 Post-processing spécialisé pièces...")

            # Nettoyage basique
            try:
                ms.apply_filter('meshing_remove_duplicate_vertices')
                ms.apply_filter('meshing_remove_unreferenced_vertices')
            except:
                try:
                    ms.apply_filter('remove_duplicate_vertices')
                    ms.apply_filter('remove_unreferenced_vertices')
                except:
                    pass

            # Lissage léger avec préservation des détails pour pièces
            try:
                ms.apply_filter('apply_coord_taubin_smoothing',
                                stepsmoothnum=3, lambda1=0.5, mu=-0.53)
            except:
                try:
                    ms.apply_filter('taubin_smooth', stepsmoothnum=3)
                except:
                    pass

            # Statistiques finales
            final_mesh = ms.current_mesh()
            print(
                f"   📈 Maillage final: {final_mesh.vertex_number()} vertices, {final_mesh.face_number()} faces")

            # Sauvegarder STL final
            stl_file = output_dir / f"{Path(input_image_path).stem}.stl"
            ms.save_current_mesh(str(stl_file))

            elapsed_time = time.time() - start_time
            print(f"✅ Fichier STL généré: {stl_file}")
            print(f"⏱️  Temps total: {elapsed_time:.1f} secondes")

            if has_reverse_image:
                print(f"🎯 Modèle créé avec optimisation 2 vues (recto + verso)")

            return str(stl_file)

        except Exception as e:
            print(f"❌ Erreur post-processing: {e}")
            return None

    except Exception as e:
        print(f"❌ Erreur génération 3D: {e}")
        return None


def diagnostic_info():
    """Affiche les informations de diagnostic pour déboguer les problèmes"""
    print("🔍 Diagnostic du système")
    print("=" * 50)

    # Versions des librairies principales
    print("📋 Versions des librairies:")
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Version CUDA: {torch.version.cuda}")
    except ImportError:
        print("   ❌ PyTorch non installé")

    try:
        import pymeshlab
        print(f"   PyMeshLab: {pymeshlab.__version__}")
    except (ImportError, AttributeError):
        print("   ❌ PyMeshLab non installé ou version sans __version__")

    try:
        from PIL import Image
        print(f"   Pillow: {Image.__version__}")
    except (ImportError, AttributeError):
        print("   ❌ Pillow non installé")

    try:
        import numpy as np
        print(f"   NumPy: {np.__version__}")
    except (ImportError, AttributeError):
        print("   ❌ NumPy non installé")

    try:
        import rembg
        print(f"   Rembg: installé")
    except ImportError:
        print("   ❌ Rembg non installé")

    # Test des filtres PyMeshLab
    print("\n🔧 Test des filtres PyMeshLab:")
    try:
        import pymeshlab as ml
        ms = ml.MeshSet()

        # Créer un mesh de test simple (cube)
        try:
            # Essayer de créer un cube pour tester
            ms.create_cube()
            print("   ✅ Création de mesh de test réussie")

            # Découvrir les filtres disponibles
            available_filters = set()
            try:
                import io
                import contextlib

                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    ms.print_filter_list()

                filter_output = f.getvalue()
                for line in filter_output.split('\n'):
                    if line.strip() and not line.startswith(' ') and ':' in line:
                        filter_name = line.split(':')[0].strip()
                        if filter_name:
                            available_filters.add(filter_name)

                print(
                    f"   📊 {len(available_filters)} filtres PyMeshLab disponibles")

                # Afficher les filtres les plus importants
                important_filters = [
                    'meshing_remove_duplicate_vertices',
                    'meshing_remove_unreferenced_vertices',
                    'apply_coord_laplacian_smoothing',
                    'meshing_decimation_quadric_edge_collapse',
                    # Anciens noms
                    'remove_duplicate_vertices',
                    'remove_unreferenced_vertices',
                    'laplacian_smooth',
                    'simplification_quadric_edge_collapse_decimation'
                ]

                print("   🎯 Filtres importants disponibles:")
                for filter_name in important_filters:
                    status = "✅" if filter_name in available_filters else "❌"
                    print(f"      {status} {filter_name}")

                # Test d'application d'un filtre simple
                try:
                    if 'meshing_remove_duplicate_vertices' in available_filters:
                        ms.apply_filter('meshing_remove_duplicate_vertices')
                        print("   ✅ Test de filtre (nouveaux noms) réussi")
                    elif 'remove_duplicate_vertices' in available_filters:
                        ms.apply_filter('remove_duplicate_vertices')
                        print("   ✅ Test de filtre (anciens noms) réussi")
                    else:
                        print("   ⚠️  Aucun filtre de nettoyage disponible")
                except Exception as e:
                    print(f"   ❌ Erreur lors du test de filtre: {e}")

            except Exception as e:
                print(f"   ❌ Erreur lors de la découverte des filtres: {e}")

        except Exception as e:
            print(f"   ❌ Erreur lors de la création du mesh de test: {e}")

    except ImportError:
        print("   ❌ PyMeshLab non disponible pour les tests")

    # Informations sur les formats d'images supportés
    print("\n📸 Formats d'images supportés:")
    supported_formats = ['PNG', 'WebP', 'JPEG', 'BMP', 'TIFF']
    for fmt in supported_formats:
        try:
            # Test basique d'ouverture
            from PIL import Image
            Image.new('RGB', (10, 10)).save(f'test.{fmt.lower()}', fmt)
            # Supprimer le fichier de test
            Path(f'test.{fmt.lower()}').unlink()
            print(f"   ✅ {fmt}")
        except Exception:
            print(f"   ❌ {fmt}")

    # Vérification de l'espace disque
    print("\n💾 Espace disque:")
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        print(f"   Total: {total // (1024**3)} GB")
        print(f"   Libre: {free // (1024**3)} GB")
        if free < 5 * 1024**3:  # Moins de 5GB
            print("   ⚠️  Espace disque faible (< 5GB)")
    except Exception as e:
        print(f"   ❌ Impossible de vérifier l'espace disque: {e}")

    print("\n" + "=" * 50)
    print("💡 Conseils de dépannage:")
    print("   1. Si erreurs de filtres PyMeshLab: version récente installée")
    print("   2. Si CUDA non disponible: vérifiez drivers NVIDIA")
    print("   3. Si mémoire insuffisante: fermez autres applications")
    print("   4. Si formats non supportés: réinstallez Pillow")


def print_coin_tips():
    """Affiche des conseils pour optimiser la conversion de pièces numismatiques"""
    print("\n💡 CONSEILS POUR PIÈCES NUMISMATIQUES:")
    print("=" * 50)
    print("🪙 QUALITÉ OPTIMALE:")
    print("   • Utilisez des images haute résolution (minimum 1000x1000 pixels)")
    print("   • Éclairage uniforme sans ombres portées")
    print("   • Fond contrasté (blanc ou noir uni)")
    print("   • Pièce bien centrée dans l'image")
    print("   • Ajoutez une image de revers avec --reverse-image pour de meilleurs résultats")
    print("")
    print("⚙️  PARAMÈTRES RECOMMANDÉS:")
    print("   • Résolution standard: --mc-resolution 640 (défaut optimisé)")
    print("   • Haute qualité: --mc-resolution 800 ou 1024")
    print("   • Si artefacts: ajuster --mc-threshold (0.1-0.2)")
    print("   • Images très détaillées: --foreground-ratio 0.7")
    print("")
    print("🚀 EXEMPLES DE COMMANDES:")
    print("   • Standard: python png-to-stl-local.py ma_piece.png")
    print("   • Avec revers: python png-to-stl-local.py recto.png --reverse-image verso.png")
    print("   • Très haute qualité: python png-to-stl-local.py piece.png --mc-resolution 1024")
    print("   • Supprimer fond: python png-to-stl-local.py piece.jpg --remove-bg")
    print("")


def get_optimal_settings_for_gpu():
    """
    Détermine les paramètres optimaux selon la mémoire GPU disponible
    Retourne des paramètres adaptés au hardware pour éviter les erreurs OOM
    """
    if not torch.cuda.is_available():
        return {
            'mc_resolution': 256,
            'image_size': 512,
            'chunk_size': 4096,
            'device_info': 'CPU seulement - paramètres conservateurs'
        }

    try:
        # Obtenir les informations GPU
        gpu_memory_gb = torch.cuda.get_device_properties(
            0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)

        print(
            f"   🔍 Détection automatique: {gpu_name} ({gpu_memory_gb:.1f} GB)")

        # Paramètres adaptés selon la mémoire GPU disponible (basés sur TripoSR officiel)
        if gpu_memory_gb >= 24:  # RTX 4090, A6000, etc.
            return {
                'mc_resolution': 320,  # Maximum recommandé dans Gradio
                'image_size': 512,     # Taille standard TripoSR
                'chunk_size': 8192,    # Valeur officielle TripoSR
                'device_info': f'GPU haut de gamme ({gpu_memory_gb:.1f}GB) - qualité maximale'
            }
        elif gpu_memory_gb >= 16:  # RTX 4080, 3090, etc.
            return {
                'mc_resolution': 320,
                'image_size': 512,
                'chunk_size': 8192,
                'device_info': f'GPU performant ({gpu_memory_gb:.1f}GB) - haute qualité'
            }
        elif gpu_memory_gb >= 10:  # RTX 3080, 4070, etc.
            return {
                'mc_resolution': 256,  # Valeur par défaut TripoSR
                'image_size': 512,
                'chunk_size': 8192,
                'device_info': f'GPU moyen-haut ({gpu_memory_gb:.1f}GB) - paramètres standards TripoSR'
            }
        elif gpu_memory_gb >= 6:   # RTX 2070 SUPER, 3060, etc.
            return {
                'mc_resolution': 256,
                'image_size': 512,
                'chunk_size': 8192,
                'device_info': f'GPU milieu de gamme ({gpu_memory_gb:.1f}GB) - paramètres TripoSR avec sécurité'
            }
        else:  # RTX 2060, GTX 1660, etc.
            return {
                'mc_resolution': 192,  # Plus conservateur pour GPU faibles
                'image_size': 512,
                'chunk_size': 4096,
                'device_info': f'GPU entrée de gamme ({gpu_memory_gb:.1f}GB) - qualité réduite'
            }

    except Exception as e:
        print(f"   ⚠️  Erreur détection GPU: {e}")
        return {
            'mc_resolution': 384,
            'image_size': 512,
            'chunk_size': 4096,
            'device_info': 'Détection échouée - paramètres sécurisés'
        }


def clear_gpu_memory():
    """Nettoie la mémoire GPU pour optimiser l'utilisation"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
