#!/usr/bin/env python3
"""
Script d'installation simplifié pour le convertisseur PNG vers STL
Évite les problèmes de compilation en installant des versions compatibles
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Exécute une commande et affiche le résultat"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Échec")
        print(f"Erreur: {e.stderr}")
        return False


def install_packages():
    """Installe les packages un par un pour éviter les conflits"""

    # Packages de base (ordre important)
    base_packages = [
        "pip --upgrade",
        "wheel",
        "setuptools",
        "numpy>=1.21,<2.1",
        "scipy>=1.4.1",
        "pillow==9.5.0",
    ]

    # Packages PyTorch avec CUDA
    torch_packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    ]

    # Packages d'image processing
    image_packages = [
        "opencv-python>=4.5.0",
        "imageio>=2.4.1",
        "tifffile>=2019.7.26",
        "PyWavelets>=1.1.1",
        "networkx>=2.2",
        "scikit-image>=0.20.0",  # Version avec wheels précompilés
    ]

    # Packages 3D
    packages_3d = [
        "PyMCubes>=0.1.6",
        "trimesh>=3.9.0",
        "plyfile>=0.7.0",
        "pymeshlab>=2022.2",
    ]

    # Packages ML/AI
    ml_packages = [
        "huggingface_hub>=0.16.0",
        "transformers>=4.30.0",
        "diffusers>=0.21.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
    ]

    # Background removal
    bg_packages = [
        "onnxruntime-gpu>=1.12.0",
        "rembg[gpu]>=2.0.0",
    ]

    # Utilities
    util_packages = [
        "omegaconf>=2.3.0",
        "tqdm>=4.65.0",
        "packaging>=20.0",
        "matplotlib>=3.5.0",
        "imageio-ffmpeg>=0.4.8",
    ]

    all_package_groups = [
        ("Packages de base", base_packages),
        ("PyTorch avec CUDA", torch_packages),
        ("Traitement d'images", image_packages),
        ("Traitement 3D", packages_3d),
        ("Machine Learning", ml_packages),
        ("Suppression d'arrière-plan", bg_packages),
        ("Utilitaires", util_packages),
    ]

    print("🚀 Installation des dépendances...")
    print("=" * 50)

    for group_name, packages in all_package_groups:
        print(f"\n📦 {group_name}")
        print("-" * 30)

        for package in packages:
            cmd = f"pip install {package}"
            if not run_command(cmd, f"Installation de {package.split()[0]}"):
                print(f"⚠️  Échec de l'installation de {package}")
                response = input("Continuer malgré l'erreur? (o/N): ")
                if response.lower() != 'o':
                    return False

    return True


def setup_triposr():
    """Clone TripoSR si nécessaire"""
    triposr_path = Path("TripoSR")

    if triposr_path.exists():
        print("✅ TripoSR déjà présent")
        return True

    print("\n📥 Clonage de TripoSR...")
    cmd = "git clone https://github.com/pyimagesearch/TripoSR.git"
    return run_command(cmd, "Clonage de TripoSR")


def verify_installation():
    """Vérifie que les packages principaux sont installés"""
    print("\n🔍 Vérification de l'installation...")

    critical_packages = [
        "torch",
        "torchvision",
        "PIL",
        "numpy",
        "opencv-python",
        "pymeshlab",
        "rembg",
        "transformers",
    ]

    failed_packages = []

    for package in critical_packages:
        try:
            if package == "PIL":
                import PIL
            elif package == "opencv-python":
                import cv2
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_packages.append(package)

    if failed_packages:
        print(f"\n⚠️  Packages manquants: {', '.join(failed_packages)}")
        return False

    # Vérifier CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA non disponible")
    except:
        print("❌ Impossible de vérifier CUDA")

    return True


def main():
    print("🎯 Installation du convertisseur PNG vers STL")
    print("=" * 50)

    # Vérifier Python
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requis")
        return

    print(f"✅ Python {sys.version}")

    # Installer les packages
    if not install_packages():
        print("\n❌ Échec de l'installation des packages")
        return

    # Setup TripoSR
    if not setup_triposr():
        print("\n❌ Échec du setup de TripoSR")
        return

    # Vérifier l'installation
    if not verify_installation():
        print("\n❌ Vérification échouée")
        return

    print("\n" + "=" * 50)
    print("🎉 Installation terminée avec succès!")
    print("\n💡 Vous pouvez maintenant utiliser:")
    print("   python png-to-stl-local.py votre_image.png")
    print("\n📖 Consultez le README.md pour plus d'informations")


if __name__ == "__main__":
    main()
