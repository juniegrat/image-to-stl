#!/usr/bin/env python3
"""
Module de configuration et vérifications runtime pour le convertisseur STL
Séparé du install.py qui gère l'installation complète
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path
import torch

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
    # Calculer le chemin vers TripoSR depuis ce fichier
    current_dir = Path(__file__).parent.parent  # De tsr/lib/ vers tsr/
    triposr_path = current_dir.parent / "TripoSR"  # De tsr/ vers /TripoSR

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
