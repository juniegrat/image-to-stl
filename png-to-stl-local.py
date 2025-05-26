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


def convert_png_to_stl(input_image_path, output_dir="output", remove_bg=False, render_video=True):
    """
    Convertit une image PNG en modèle STL

    Args:
        input_image_path: Chemin vers l'image PNG d'entrée
        output_dir: Répertoire de sortie pour les fichiers générés
        remove_bg: Si True, supprime l'arrière-plan de l'image
        render_video: Si True, génère une vidéo du modèle 3D
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

    if device == "cuda":
        # Nettoyer la mémoire GPU avant de commencer
        torch.cuda.empty_cache()

    # Paramètres TripoSR
    pretrained_model_name_or_path = "stabilityai/TripoSR"
    chunk_size = 8192
    foreground_ratio = 0.85
    model_save_format = "obj"

    # Créer le répertoire de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger l'image
    print(f"\n📷 Chargement de l'image: {input_image_path}")
    try:
        original_image = Image.open(input_image_path)
        print(f"   Taille originale: {original_image.size}")
        print(f"   Mode: {original_image.mode}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'image: {e}")
        return None

    # Redimensionner l'image
    original_image_resized = original_image.resize((512, 512))

    # Créer le dossier examples
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    original_image_resized.save(examples_dir / "product.png")

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

    # Traiter l'image
    print(f"\n🎨 Traitement de l'image...")
    try:
        if remove_bg:
            print("   Suppression de l'arrière-plan...")
            rembg_session = rembg.new_session()
            image = remove_background(original_image_resized, rembg_session)
        else:
            image = original_image_resized

        # Redimensionner l'image
        image = resize_foreground(image, foreground_ratio)

        # Gérer les images RGBA
        if image.mode == "RGBA":
            print("   Conversion RGBA vers RGB...")
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + \
                (1 - image[:, :, 3:4]) * 0.5
            image = Image.fromarray((image * 255.0).astype(np.uint8))

        # Créer le répertoire de sortie pour cette image
        image_dir = output_dir / "0"
        image_dir.mkdir(exist_ok=True)

        # Sauvegarder l'image traitée
        image.save(image_dir / "input.png")
        print("✅ Image traitée et sauvegardée")

    except Exception as e:
        print(f"⚠️  Erreur lors du traitement de l'image: {e}")
        print("   Utilisation de l'image originale...")
        image = original_image_resized
        image_dir = output_dir / "0"
        image_dir.mkdir(exist_ok=True)
        image.save(image_dir / "input.png")

    # Générer le modèle 3D
    print(f"\n🏗️  Génération du modèle 3D...")
    start_time = time.time()

    try:
        print("   Génération des codes de scène...")
        with torch.no_grad():
            scene_codes = model([image], device=device)

        # Rendre les vues si demandé
        if render_video:
            print("   Rendu des vues multiples (30 vues)...")
            render_images = model.render(
                scene_codes, n_views=30, return_type="pil")
            for ri, render_image in enumerate(render_images[0]):
                render_image.save(image_dir / f"render_{ri:03d}.png")

            # Créer une vidéo
            print("   Création de la vidéo...")
            save_video(render_images[0], str(image_dir / "render.mp4"), fps=30)
            print(f"   ✅ Vidéo sauvegardée: {image_dir / 'render.mp4'}")

        # Extraire et sauvegarder le maillage
        print("   Extraction du maillage 3D...")
        meshes = model.extract_mesh(scene_codes, has_vertex_color=False)
        mesh_file = image_dir / f"mesh.{model_save_format}"
        meshes[0].export(str(mesh_file))
        print(f"   ✅ Modèle 3D sauvegardé: {mesh_file}")

        # Convertir en STL
        print("   Conversion en STL...")
        ms = pymesh.MeshSet()
        ms.load_new_mesh(str(mesh_file))

        stl_file = output_dir / f"{Path(input_image_path).stem}.stl"
        ms.save_current_mesh(str(stl_file))

        elapsed_time = time.time() - start_time
        print(f"✅ Fichier STL généré: {stl_file}")
        print(f"⏱️  Temps total: {elapsed_time:.1f} secondes")

        return str(stl_file)

    except Exception as e:
        print(f"❌ Erreur lors de la génération du modèle 3D: {e}")
        if "out of memory" in str(e).lower():
            print(
                "   💡 Conseil: Fermez les autres applications utilisant le GPU et réessayez.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convertir une image PNG en modèle STL")
    parser.add_argument("input", help="Chemin vers l'image PNG d'entrée")
    parser.add_argument("-o", "--output", default="output",
                        help="Répertoire de sortie (défaut: output)")
    parser.add_argument("--remove-bg", action="store_true",
                        help="Supprimer l'arrière-plan de l'image")
    parser.add_argument("--no-video", action="store_true",
                        help="Ne pas générer de vidéo du modèle")
    parser.add_argument("--setup", action="store_true",
                        help="Installer les dépendances et configurer l'environnement")

    args = parser.parse_args()

    print("🚀 Convertisseur PNG vers STL avec TripoSR")
    print("=" * 50)

    if args.setup:
        print("⚙️  Configuration de l'environnement...")
        check_and_install_dependencies()
        setup_triposr()
        print("\n✅ Configuration terminée! Vous pouvez maintenant utiliser le script.")
        print(f"💡 Exemple: python {sys.argv[0]} mon_image.png")
        return

    if not Path(args.input).exists():
        print(f"❌ Erreur: Le fichier '{args.input}' n'existe pas.")
        return

    # Vérifier CUDA
    check_cuda_compatibility()

    # Setup TripoSR si nécessaire
    print(f"\n📦 Configuration de TripoSR...")
    setup_triposr()

    # Convertir l'image
    stl_file = convert_png_to_stl(
        args.input,
        args.output,
        remove_bg=args.remove_bg,
        render_video=not args.no_video
    )

    print("\n" + "=" * 50)
    if stl_file:
        print(f"🎉 Conversion terminée avec succès!")
        print(f"📁 Fichier STL: {stl_file}")
        print(f"📁 Dossier de sortie: {args.output}")
        if not args.no_video:
            print(f"🎬 Vidéo disponible dans: {args.output}/0/render.mp4")
    else:
        print("❌ Échec de la conversion.")
        print("💡 Vérifiez les messages d'erreur ci-dessus pour plus d'informations.")


if __name__ == "__main__":
    main()
