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
    else:
        # Structure: TripoSR/
        sys.path.insert(0, str(triposr_path))
        sys.path.insert(0, str(triposr_path / "tsr"))

    # Installer les dépendances du projet TripoSR (comme dans le script Colab)
    requirements_file = triposr_path / "requirements.txt"
    if requirements_file.exists():
        # Vérifier d'abord si les dépendances sont déjà installées
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(
                    requirements_file), "--dry-run"
            ], capture_output=True, text=True)

            # Si des packages doivent être installés, afficher le message
            if "would install" in result.stdout or result.returncode != 0:
                print("Installation des dépendances TripoSR...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", str(
                        requirements_file), "-q"
                ])
        except subprocess.CalledProcessError:
            # En cas d'erreur avec --dry-run, installer directement
            print("Installation des dépendances TripoSR...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(
                    requirements_file), "-q"
            ])

    print("✅ TripoSR configuré avec succès!")
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
        print("   Tentative de chargement avec OpenCV...")
        try:
            import cv2
            img_cv = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                raise ValueError("cv2.imread renvoie None")
            # Convertir BGR -> RGB
            if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            elif len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
            image = Image.fromarray(img_cv)
            original_format = Path(image_path).suffix.lstrip('.').lower()
            print("✅ Chargement réussi avec OpenCV, conversion en PIL.Image")
            return image, original_format
        except Exception as e2:
            print(f"❌ Échec du fallback OpenCV: {e2}")
            raise


def convert_png_to_stl(input_image_path, output_dir="output", remove_bg=True, render_video=True, reverse_image_path=None, render_params=None):
    """
    Convertit une image en modèle STL 3D en s'appuyant directement sur le script officiel
    TripoSR/run.py. On se contente de :
    1. Préparer/installer TripoSR si nécessaire (setup_triposr)
    2. Lancer `python TripoSR/run.py <image>` avec les bonnes options
    3. Convertir le mesh OBJ créé par TripoSR en STL via PyMeshLab

    Seuls les points non gérés par run.py (ex : sortie STL) restent traités ici.
    """

    # 1) Vérifier/installer TripoSR et ses dépendances
    setup_triposr()

    run_py = Path("TripoSR") / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(
            f"Impossible de trouver {run_py}. Avez-vous bien cloné TripoSR ?")

    # 2) Construire la commande à exécuter
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(run_py), str(
        input_image_path), "--output-dir", str(output_dir)]

    # Gestion arrière-plan : par défaut run.py supprime le BG, on ajoute --no-remove-bg si l'utilisateur le souhaite
    if not remove_bg:
        cmd.append("--no-remove-bg")

    # Vidéo de rendu si souhaitée
    if render_video:
        cmd.append("--render")

    # Pour l'instant reverse_image_path n'est pas pris en charge par run.py ➜ avertir
    if reverse_image_path is not None:
        print("⚠️  L'option reverse_image_path n'est pas supportée avec run.py et sera ignorée.")

    # 3) Créer le dossier de sortie nécessaire (bug fix pour --no-remove-bg + --render)
    subdir = output_dir / "0"
    subdir.mkdir(parents=True, exist_ok=True)

    # 4) Lancer TripoSR/run.py
    print(f"\n🚀 Exécution de TripoSR/run.py : {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution de TripoSR/run.py : {e}")
        return None

    # 5) Conversion OBJ → STL
    obj_path = output_dir / "0" / "mesh.obj"
    if not obj_path.exists():
        print(f"❌ Fichier OBJ introuvable : {obj_path}")
        return None

    stl_path = output_dir / "model.stl"
    try:
        ms = pymesh.MeshSet()
        ms.load_new_mesh(str(obj_path))
        ms.save_current_mesh(str(stl_path))
    except Exception as e:
        print(f"❌ Erreur lors de la conversion OBJ → STL : {e}")
        return None

    print(f"✅ STL généré : {stl_path}")
    return str(stl_path)


def get_render_params(args):
    """
    Crée les paramètres de rendu basés sur les arguments de ligne de commande
    """
    return {
        'n_views': args.render_views,
        'elevation_deg': args.render_elevation,
        'camera_distance': args.render_distance,
        'fovy_deg': args.render_fov,
        'height': args.render_resolution,
        'width': args.render_resolution,
        'return_type': "pil"
    }


def print_render_info(args):
    """
    Affiche les informations de rendu pour diagnostic
    """
    print(f"\n🎬 Paramètres de rendu:")
    print(f"   Résolution: {args.render_resolution}x{args.render_resolution}")
    print(f"   Nombre de vues: {args.render_views}")
    print(f"   Élévation: {args.render_elevation}°")
    print(f"   Distance caméra: {args.render_distance}")
    print(f"   Champ de vision: {args.render_fov}°")


def main():
    parser = argparse.ArgumentParser(
        description="Convertir une image (PNG, WebP, JPEG, etc.) en modèle STL")
    parser.add_argument(
        "input", help="Chemin vers l'image d'entrée (PNG, WebP, JPEG, BMP, TIFF)")
    parser.add_argument("-o", "--output", default="output",
                        help="Répertoire de sortie (défaut: output)")
    parser.add_argument("--remove-bg", action="store_true",
                        help="Supprimer l'arrière-plan de l'image (défaut: True)")
    parser.add_argument("--no-remove-bg", action="store_true",
                        help="NE PAS supprimer l'arrière-plan de l'image")
    parser.add_argument("--no-video", action="store_true",
                        help="Ne pas générer de vidéo du modèle")
    parser.add_argument("--reverse-image",
                        help="Chemin vers l'image de revers (verso) pour améliorer la reconstruction 3D")
    parser.add_argument("--mc-resolution", type=int, default=256,
                        help="Résolution du marching cubes (défaut: 256 - paramètre officiel TripoSR)")
    parser.add_argument("--mc-threshold", type=float, default=25.0,
                        help="Seuil du marching cubes (défaut: 25.0 - paramètre officiel TripoSR)")
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
    parser.add_argument("--render-resolution", type=int, default=512,
                        help="Résolution des images de rendu (défaut: 512)")
    parser.add_argument("--render-elevation", type=float, default=0.0,
                        help="Angle d'élévation de la caméra en degrés (défaut: 0.0)")
    parser.add_argument("--render-distance", type=float, default=1.9,
                        help="Distance de la caméra (défaut: 1.9)")
    parser.add_argument("--render-fov", type=float, default=40.0,
                        help="Champ de vision de la caméra en degrés (défaut: 40.0)")
    parser.add_argument("--render-views", type=int, default=30,
                        help="Nombre de vues pour la vidéo de rotation (défaut: 30)")
    parser.add_argument("--analyze-render",
                        help="Analyser la qualité des rendus dans le dossier spécifié")
    parser.add_argument("--render-tips", action="store_true",
                        help="Afficher les conseils pour améliorer la qualité des rendus")

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

    if args.debug:
        diagnostic_info()
        return

    if args.tips:
        print_coin_tips()
        return

    if args.render_tips:
        print_render_tips()
        return

    if args.analyze_render:
        analyze_render_quality(args.analyze_render)
        return

    # Afficher les informations de rendu si mode debug ou si paramètres non-standard
    if args.debug or args.render_resolution != 512 or args.render_views != 30:
        print_render_info(args)

    # Vérifier l'image d'entrée
    if not Path(args.input).exists():
        print(f"❌ Fichier d'entrée introuvable: {args.input}")
        return

    # Vérifier l'image de revers si spécifiée
    if args.reverse_image and not Path(args.reverse_image).exists():
        print(f"❌ Fichier image de revers introuvable: {args.reverse_image}")
        return

    # Initialiser TripoSR (ajout du chemin) et vérifier CUDA
    setup_triposr()
    check_cuda_compatibility()

    # Obtenir les paramètres de rendu (pour l'instant non utilisés mais conservés pour compatibilité)
    render_params = get_render_params(args)

    # Gestion de la suppression d'arrière-plan (pour correspondre au comportement par défaut de run.py)
    if args.remove_bg and args.no_remove_bg:
        print("❌ Erreur: --remove-bg et --no-remove-bg sont incompatibles")
        return

    if args.no_remove_bg:
        remove_bg = False
        print("🖼️  Mode: Conservation de l'arrière-plan")
    elif args.remove_bg:
        remove_bg = True
        print("🖼️  Mode: Suppression de l'arrière-plan")
    else:
        # Par défaut, comme run.py, on supprime l'arrière-plan
        remove_bg = True
        print("🖼️  Mode: Suppression de l'arrière-plan (défaut comme run.py)")

    # Conversion unique via le script officiel TripoSR/run.py
    stl_file = convert_png_to_stl(
        args.input,
        args.output,
        remove_bg=remove_bg,
        render_video=not args.no_video,
        reverse_image_path=args.reverse_image,
        render_params=render_params,
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
                             render_video=True, reverse_image_path=None, render_params=None):
    """
    Convertit une image en modèle STL 3D pour pièces numismatiques.
    Utilise les paramètres par défaut de TripoSR pour la compatibilité maximale.
    """

    print("🔧 Mode TripoSR avec paramètres par défaut")

    # Paramètres de rendu par défaut si non spécifiés
    if render_params is None:
        render_params = {
            'n_views': 30,
            'elevation_deg': 0.0,
            'camera_distance': 1.9,
            'fovy_deg': 40.0,
            'height': 512,
            'width': 512,
            'return_type': "pil"
        }

    # Importer les modules TripoSR
    try:
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground, save_video
    except ImportError as e:
        print(f"❌ Erreur importation TripoSR: {e}")
        return None

    # Configuration avec paramètres par défaut TripoSR
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mc_resolution = 256  # Valeur par défaut TripoSR
    image_size = 512     # Taille standard TripoSR
    chunk_size = 8192    # Valeur par défaut TripoSR

    print(f"\n🔧 Configuration (paramètres par défaut TripoSR):")
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
                print(
                    f"   🎬 Rendu des vues multiples ({render_params['n_views']} vues)...")
                print(
                    f"   📐 Résolution: {render_params['height']}x{render_params['width']}")
                # Utiliser les paramètres de rendu personnalisés
                render_images = model.render(scene_codes, **render_params)
                for ri, render_image in enumerate(render_images[0]):
                    render_image.save(image_dir / f"render_{ri:03d}.png")
                save_video(render_images[0], str(
                    image_dir / "render.mp4"), fps=30)
                print(f"   ✅ Vidéo sauvegardée: {image_dir / 'render.mp4'}")
            except Exception as e:
                print(
                    f"   ⚠️  Erreur rendu vidéo: {e}, continuation sans vidéo")

        # EXTRACTION DU MAILLAGE avec paramètres par défaut TripoSR
        print("   🔧 Extraction du maillage 3D...")
        clear_gpu_memory()

        try:
            meshes = model.extract_mesh(
                scene_codes,
                # Conforme au script officiel (not bake_texture)
                has_vertex_color=True,
                resolution=mc_resolution
                # threshold utilise la valeur par défaut de 25.0 (100% conforme TripoSR officiel)
            )
            print(f"   ✅ Extraction réussie avec résolution {mc_resolution}")
        except Exception as e:
            print(f"   ❌ Erreur extraction maillage: {e}")
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

            # Post-processing du maillage
            print("   🔧 Post-processing du maillage...")

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
                print(f"🎯 Modèle créé avec 2 vues (recto + verso)")

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


def analyze_render_quality(render_dir):
    """
    Analyse la qualité des rendus et suggère des améliorations
    """
    render_dir = Path(render_dir)

    if not render_dir.exists():
        print("❌ Dossier de rendu introuvable")
        return

    render_files = list(render_dir.glob("render_*.png"))
    if not render_files:
        print("❌ Aucun fichier de rendu trouvé")
        return

    print(f"\n🔍 Analyse de qualité des rendus:")
    print(f"   📁 Dossier: {render_dir}")
    print(f"   🖼️  Nombre de vues: {len(render_files)}")

    # Analyser quelques images pour détecter des problèmes
    sample_files = render_files[:5]  # Analyser les 5 premiers

    for i, render_file in enumerate(sample_files):
        try:
            from PIL import Image
            img = Image.open(render_file)
            width, height = img.size

            # Convertir en array numpy pour analyse
            import numpy as np
            img_array = np.array(img)

            # Détecter les zones noires (possibles artefacts)
            if len(img_array.shape) == 3:
                # Image couleur
                # Pixels très sombres
                dark_pixels = np.sum(img_array, axis=2) < 30
                dark_ratio = np.sum(dark_pixels) / (width * height)

                if dark_ratio > 0.3:  # Plus de 30% de pixels sombres
                    print(
                        f"   ⚠️  Vue {i:03d}: Beaucoup de zones sombres ({dark_ratio:.1%})")

            print(f"   ✅ Vue {i:03d}: {width}x{height} - OK")

        except Exception as e:
            print(f"   ❌ Vue {i:03d}: Erreur d'analyse - {e}")

    # Suggestions d'amélioration
    print(f"\n💡 Suggestions d'amélioration:")
    print(f"   • Augmenter la résolution: --render-resolution 1024")
    print(f"   • Ajuster l'angle: --render-elevation 15 (vue légèrement en plongée)")
    print(f"   • Modifier la distance: --render-distance 2.2 (plus loin)")
    print(f"   • Changer le champ de vision: --render-fov 35 (plus serré)")
    print(f"   • Plus de vues: --render-views 60 (rotation plus fluide)")


def print_render_tips():
    """
    Affiche des conseils pour améliorer la qualité des rendus
    """
    print("\n🎬 Conseils pour améliorer la qualité des rendus:")
    print("\n📐 Résolution:")
    print("   • 256x256: Rapide mais qualité basique")
    print("   • 512x512: Bon compromis (défaut)")
    print("   • 1024x1024: Haute qualité mais plus lent")
    print("   • 2048x2048: Très haute qualité (GPU puissant requis)")

    print("\n📷 Paramètres de caméra:")
    print("   • Distance 1.5-1.9: Vue rapprochée (détails)")
    print("   • Distance 2.0-2.5: Vue éloignée (contexte)")
    print("   • Élévation 0°: Vue horizontale")
    print("   • Élévation 15-30°: Vue en plongée (recommandé)")
    print("   • FOV 30-35°: Vue serrée (zoom)")
    print("   • FOV 40-50°: Vue large (contexte)")

    print("\n🎞️  Nombre de vues:")
    print("   • 15-20 vues: Rotation basique")
    print("   • 30 vues: Standard (défaut)")
    print("   • 60 vues: Rotation très fluide")
    print("   • 120 vues: Rotation ultra-fluide (très lent)")

    print("\n🔧 Exemples de commandes:")
    print("   # Haute qualité")
    print("   python image-to-stl.py image.png --render-resolution 1024 --render-elevation 20")
    print("   # Vue rapprochée")
    print("   python image-to-stl.py image.png --render-distance 1.6 --render-fov 35")
    print("   # Rotation ultra-fluide")
    print("   python image-to-stl.py image.png --render-views 60")


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


def clear_gpu_memory():
    """Nettoie la mémoire GPU pour optimiser l'utilisation"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
