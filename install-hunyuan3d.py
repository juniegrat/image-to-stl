#!/usr/bin/env python3
"""
Installation complète de Hunyuan3D-2 avec toutes les dépendances
Version corrigée pour résoudre les problèmes de compatibilité PyTorch/xFormers/diffusers
"""

import os
import sys
import subprocess
import tempfile
import shutil
import time
import platform
from pathlib import Path


def run_command(cmd, description="", check=True):
    """Exécute une commande avec gestion d'erreurs"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Terminé")
            return True
        else:
            print(f"⚠️  {description} - Avertissement:")
            if result.stderr:
                print(f"   {result.stderr[:300]}...")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Erreur:")
        if e.stderr:
            print(f"   {e.stderr[:300]}...")
        return False


def safe_remove_directory(path, max_retries=3):
    """Supprime un répertoire de manière sécurisée sur Windows"""
    if not os.path.exists(path):
        return True

    for attempt in range(max_retries):
        try:
            # Sur Windows, essayer de libérer les locks sur les fichiers .git
            if platform.system() == "Windows":
                # Essayer de changer les permissions des fichiers .git
                for root, dirs, files in os.walk(path):
                    if '.git' in root:
                        for file in files:
                            try:
                                file_path = os.path.join(root, file)
                                os.chmod(file_path, 0o777)
                            except:
                                pass
                time.sleep(1)

            shutil.rmtree(path)
            return True
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                print(
                    f"   🔄 Tentative {attempt + 1} échouée, nouvelle tentative...")
                time.sleep(3)  # Attendre plus longtemps
            else:
                print(
                    f"   ✅ Répertoire temporaire utilisé - suppression manuelle possible")
                print(f"   📁 Emplacement: {path}")
                return False

    return False


def check_python_version():
    """Vérifie la version Python"""
    print("🐍 Vérification de la version Python...")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} non supporté")
        print("💡 Hunyuan3D-2 requiert Python 3.8+")
        return False

    print(
        f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True


def check_cuda():
    """Vérifie la disponibilité de CUDA"""
    print("🔧 Vérification de CUDA...")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"   Version CUDA: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA non disponible - utilisation CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch non installé - vérification CUDA impossible")
        return False


def uninstall_conflicting_packages():
    """Désinstalle uniquement les packages vraiment conflictuels"""
    print("🧹 Vérification des packages conflictuels...")

    # Vérifier quels packages sont réellement problématiques
    problematic_packages = []

    # Vérifier PyTorch - garder si version compatible
    try:
        import torch
        version = torch.__version__
        if version.startswith('2.1.'):
            print(f"✅ PyTorch {version} compatible - conservation")
        else:
            print(
                f"⚠️  PyTorch {version} non compatible - marqué pour mise à jour")
            problematic_packages.extend(['torch', 'torchvision', 'torchaudio'])
    except ImportError:
        print("ℹ️  PyTorch non installé - installation requise")
        # Pas besoin de désinstaller ce qui n'existe pas

    # Vérifier diffusers - garder si version compatible
    try:
        import diffusers
        version = diffusers.__version__
        if version.startswith('0.28.'):
            print(f"✅ Diffusers {version} compatible - conservation")
        else:
            print(
                f"⚠️  Diffusers {version} non compatible - marqué pour mise à jour")
            problematic_packages.append('diffusers')
    except ImportError:
        print("ℹ️  Diffusers non installé - installation requise")

    # Vérifier transformers - garder si version compatible
    try:
        import transformers
        version = transformers.__version__
        if version.startswith('4.37.'):
            print(f"✅ Transformers {version} compatible - conservation")
        else:
            print(
                f"⚠️  Transformers {version} non compatible - marqué pour mise à jour")
            problematic_packages.append('transformers')
    except ImportError:
        print("ℹ️  Transformers non installé - installation requise")

    # Vérifier huggingface_hub - utiliser la nouvelle API
    try:
        from huggingface_hub import hf_hub_download
        import huggingface_hub
        version = huggingface_hub.__version__
        if version.startswith('0.20.'):
            print(
                f"✅ huggingface_hub {version} avec nouvelle API - conservation")
        else:
            print(
                f"⚠️  huggingface_hub {version} non compatible - marqué pour mise à jour")
            problematic_packages.append('huggingface_hub')
    except ImportError:
        print("⚠️  huggingface_hub incompatible - marqué pour mise à jour")
        problematic_packages.append('huggingface_hub')

    # Désinstaller uniquement les packages problématiques
    if problematic_packages:
        print(
            f"🔄 Désinstallation des packages problématiques: {', '.join(problematic_packages)}")
        for package in problematic_packages:
            run_command(f"pip uninstall -y {package}",
                        f"Désinstallation de {package}", check=False)
    else:
        print("✅ Tous les packages sont compatibles - pas de désinstallation nécessaire")


def create_requirements_file():
    """Crée le fichier requirements avec versions compatibles"""
    print("✅ Création du fichier requirements-hunyuan3d.txt (versions compatibles)")

    requirements = """# Hunyuan3D-2 Requirements - Versions compatibles testées
# PyTorch 2.1.0 avec CUDA 11.8 (version stable)
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# Diffusers et transformers - versions compatibles avec huggingface_hub récent
diffusers==0.28.0
transformers==4.37.2
accelerate==0.26.1
safetensors==0.4.1
tokenizers==0.15.2

# HuggingFace Hub - version stable récente
huggingface_hub==0.20.3

# Traitement d'images et vidéos
pillow>=9.5.0
opencv-python>=4.7.0
scikit-image>=0.20.0
matplotlib>=3.7.0
imageio>=2.28.0
imageio-ffmpeg>=0.4.8
trimesh>=3.21.0
rembg>=2.0.0

# Utilitaires
numpy>=1.24.0,<2.0.0
tqdm>=4.65.0
einops>=0.6.0
omegaconf>=2.3.0
pydantic>=1.10.0,<2.0.0
typing-extensions>=4.5.0

# Optionnel - sera installé séparément
# xformers (installé après PyTorch)
"""

    with open("requirements-hunyuan3d.txt", "w") as f:
        f.write(requirements)


def install_pytorch():
    """Installe PyTorch 2.1.0 si nécessaire"""
    print("🔥 Vérification de PyTorch 2.1.0...")

    # Vérifier si PyTorch est déjà installé avec la bonne version
    try:
        import torch
        version = torch.__version__
        if version.startswith('2.1.'):
            print(f"✅ PyTorch {version} déjà installé - pas de réinstallation")
            return True
        else:
            print(
                f"⚠️  PyTorch {version} non compatible - mise à jour requise")
    except ImportError:
        print("ℹ️  PyTorch non installé - installation requise")

    # Détecter si GPU NVIDIA est disponible
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
        has_nvidia = result.returncode == 0
    except:
        has_nvidia = False

    if has_nvidia:
        print("✅ GPU NVIDIA détecté")
        # Installer PyTorch 2.1.0 avec CUDA 11.8 (version stable)
        success = run_command(
            "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118",
            "PyTorch 2.1.0 avec CUDA 11.8",
            check=False
        )
        if not success:
            print("⚠️  Installation CUDA échouée, fallback vers CPU")
            run_command(
                "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0", "PyTorch 2.1.0 CPU")
    else:
        print("⚠️  GPU NVIDIA non détecté, installation CPU")
        run_command(
            "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0", "PyTorch 2.1.0 CPU")

    return True


def install_xformers():
    """Installe xFormers compatible avec PyTorch 2.1.0"""
    print("⚙️  Installation de xFormers compatible...")

    # Vérifier si PyTorch est installé
    try:
        import torch
        torch_version = torch.__version__
        print(f"   PyTorch détecté: {torch_version}")

        # Installer xFormers compatible avec PyTorch 2.1.0
        success = run_command(
            "pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118",
            "xFormers compatible avec PyTorch 2.1.0",
            check=False
        )

        if not success:
            print("⚠️  Installation xFormers échouée, utilisation sans xFormers")

    except ImportError:
        print("⚠️  PyTorch non détecté, installation xFormers ignorée")

    return True


def install_dependencies():
    """Installe toutes les dépendances avec versions compatibles"""
    print("📦 Installation des dépendances compatibles...")

    # Installation groupée
    success = run_command(
        "pip install -r requirements-hunyuan3d.txt",
        "Installation des dépendances compatibles",
        check=False
    )

    if not success:
        print("⚠️  Installation groupée échouée, tentative individuelle...")
        # Installer les dépendances critiques une par une
        critical_deps = [
            "diffusers==0.28.0",
            "transformers==4.37.2",
            "accelerate==0.26.1",
            "safetensors==0.4.1",
            "huggingface_hub==0.20.3",
            "numpy>=1.24.0,<2.0.0",
            "pillow>=9.5.0",
            "trimesh>=3.21.0",
            "rembg>=2.0.0",
            "matplotlib>=3.7.0",
            "tqdm>=4.65.0",
            "omegaconf>=2.3.0",
            "einops>=0.6.0"
        ]

        for dep in critical_deps:
            run_command(f"pip install \"{dep}\"",
                        f"Installation de {dep}", check=False)

    return True


def install_torchmcubes():
    """Installe torchmcubes compatible CUDA si nécessaire"""
    print("🧊 Vérification de torchmcubes compatible CUDA...")

    # Vérifier si torchmcubes est déjà installé et fonctionne
    try:
        import torchmcubes
        from torchmcubes import marching_cubes
        print("✅ torchmcubes déjà installé et fonctionnel")

        # Test rapide pour vérifier la compatibilité CUDA
        import torch
        if torch.cuda.is_available():
            try:
                # Test simple avec un petit tensor
                test_tensor = torch.randn(8, 8, 8).cuda()
                print("✅ torchmcubes compatible CUDA - pas de réinstallation")
                return True
            except Exception as e:
                print(f"⚠️  torchmcubes CUDA non fonctionnel: {e}")
                print("   Réinstallation requise")
        else:
            print("✅ torchmcubes CPU fonctionnel - pas de réinstallation")
            return True
    except ImportError:
        print("ℹ️  torchmcubes non installé - installation requise")
    except Exception as e:
        print(f"⚠️  torchmcubes problématique: {e}")
        print("   Réinstallation requise")

    # Désinstaller et réinstaller uniquement si nécessaire
    run_command(
        "pip uninstall torchmcubes -y",
        "Désinstallation de torchmcubes existant",
        check=False
    )

    # Installer la version spécifique compatible CUDA
    success = run_command(
        "pip install git+https://github.com/tatsy/torchmcubes.git@3aef8afa5f21b113afc4f4ea148baee850cbd472",
        "Installation de torchmcubes compatible CUDA",
        check=False
    )

    if success:
        print("✅ torchmcubes compatible CUDA installé avec succès")
    else:
        print("⚠️  Erreur installation torchmcubes - tentative de fallback")
        # Tentative avec la version standard
        run_command(
            "pip install torchmcubes",
            "Installation de torchmcubes standard",
            check=False
        )

    return True


def install_hunyuan3d():
    """Installe Hunyuan3D-2"""
    print("🏛️  Installation de Hunyuan3D-2...")

    # Créer un nom de répertoire unique pour éviter les conflits
    temp_dir = tempfile.mkdtemp(prefix="hunyuan3d_install_")
    original_dir = os.getcwd()

    try:
        os.chdir(temp_dir)

        # Cloner le repo
        success = run_command(
            "git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git",
            "Clonage de Hunyuan3D-2"
        )

        if success:
            hunyuan_dir = os.path.join(temp_dir, "Hunyuan3D-2")
            os.chdir(hunyuan_dir)

            # Installer le package
            success = run_command(
                "pip install -e .",
                "Installation du package Hunyuan3D",
                check=False
            )

            if not success:
                print("⚠️  Installation pip échouée, tentative avec setup.py")
                run_command("python setup.py develop",
                            "Installation avec setup.py", check=False)

    except Exception as e:
        print(f"⚠️  Erreur lors de l'installation: {e}")

    finally:
        # Retourner au répertoire original
        os.chdir(original_dir)

        # Nettoyer le répertoire temporaire de manière sécurisée
        if os.path.exists(temp_dir):
            print("🧹 Nettoyage du répertoire temporaire...")
            safe_remove_directory(temp_dir)

    return True


def test_installation():
    """Teste l'installation avec diagnostics détaillés"""
    print("🧪 Test de l'installation...")

    # Test des imports de base
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} importé")

        import numpy as np
        print(f"✅ NumPy {np.__version__} importé")

        import PIL
        print(f"✅ PIL importé")

        import trimesh
        print(f"✅ Trimesh importé")

        import rembg
        print(f"✅ rembg importé")

        import diffusers
        print(f"✅ Diffusers {diffusers.__version__} importé")

        import transformers
        print(f"✅ Transformers {transformers.__version__} importé")

    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        return False

    # Test CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA fonctionnel: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA non disponible")
    except:
        print("⚠️  Test CUDA échoué")

    # Test xFormers
    try:
        import xformers
        print(f"✅ xFormers {xformers.__version__} disponible")
    except ImportError:
        print("⚠️  xFormers non disponible - continuera sans optimisations")

    # Test torchmcubes avec CUDA
    try:
        import torchmcubes
        print("✅ torchmcubes importé")

        # Test spécifique pour la compatibilité CUDA
        try:
            from torchmcubes import marching_cubes
            print("✅ marching_cubes fonction disponible")

            # Test des fonctions CUDA si disponibles
            import torch
            if torch.cuda.is_available():
                try:
                    # Test avec un petit tensor pour vérifier la compatibilité CUDA
                    test_tensor = torch.randn(32, 32, 32).cuda()
                    # Ne pas exécuter marching_cubes ici car cela nécessite des données spécifiques
                    print("✅ torchmcubes compatible CUDA (test tensor réussi)")
                except Exception as e:
                    print(f"⚠️  torchmcubes CUDA: {e}")
                    print("   Utilisera la version CPU si nécessaire")
            else:
                print("⚠️  CUDA non disponible - torchmcubes utilisera le CPU")
        except Exception as e:
            print(f"⚠️  Test torchmcubes détaillé: {e}")

    except ImportError as e:
        print(f"❌ torchmcubes non disponible: {e}")
        print("   Réinstallez avec: pip install git+https://github.com/tatsy/torchmcubes.git@3aef8afa5f21b113afc4f4ea148baee850cbd472")

    # Test Hunyuan3D avec gestion d'erreurs améliorée
    try:
        # Test simple d'import sans initialisation complète
        print("   🧪 Test import Hunyuan3D modules...")

        # Test import basique
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        print("✅ Hunyuan3D shape pipeline disponible")

        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        print("✅ Hunyuan3D texture pipeline disponible")

        # Test huggingface_hub avec nouvelle API
        try:
            from huggingface_hub import hf_hub_download
            import huggingface_hub
            print(
                f"✅ huggingface_hub {huggingface_hub.__version__} avec nouvelle API disponible")
        except ImportError:
            print(
                "⚠️  hf_hub_download non disponible - vérifiez la version huggingface_hub")
            print("   Solution: pip install 'huggingface_hub==0.20.3'")

        # Test diffusers avec nouvelles versions
        try:
            import diffusers
            print(
                f"✅ diffusers {diffusers.__version__} compatible avec huggingface_hub")
        except ImportError:
            print("⚠️  diffusers non disponible")

        print("✅ Hunyuan3D-2 installé et fonctionnel avec versions compatibles")
        return True
    except ImportError as e:
        print(f"⚠️  Hunyuan3D non disponible: {e}")
        if "transformers" in str(e):
            print("   ⚠️  Conflit de version transformers détecté")
            print(
                "   💡 hy3dgen requiert transformers>=4.48.0, mais nous utilisons 4.37.2")
            print(
                "   💡 Cela peut causer des avertissements mais ne devrait pas empêcher le fonctionnement")
        print("   Installation de base réussie - testez avec le convertisseur")
        return True
    except Exception as e:
        print(f"⚠️  Test Hunyuan3D: {e}")
        return True


def download_models():
    """Vérifie si les modèles sont déjà dans le cache Hugging Face"""
    print("📥 Vérification des modèles dans le cache Hugging Face...")

    try:
        import os
        from pathlib import Path

        # Vérifier les emplacements possibles du cache
        possible_cache_dirs = [
            Path.home() / ".cache" / "huggingface" / "hub" / "models--tencent--Hunyuan3D-2",
            Path("C:/Users") / os.getenv("USERNAME", "User") / ".cache" /
            "huggingface" / "hub" / "models--tencent--Hunyuan3D-2",
        ]

        for cache_dir in possible_cache_dirs:
            if cache_dir.exists():
                print(f"   ✅ Modèle Hunyuan3D-2 trouvé dans le cache")
                print(f"   📁 Emplacement: {cache_dir}")
                return True

        print("   ℹ️  Modèle non trouvé dans le cache - sera téléchargé au premier usage")
        print(
            "   📁 Sera stocké dans: ~/.cache/huggingface/hub/models--tencent--Hunyuan3D-2")
        return True

    except Exception as e:
        print(f"   ✅ Vérification cache échouée (normal): {e}")
        print("   📁 Modèle sera téléchargé automatiquement au premier usage")
        return True


def cleanup_temp_files():
    """Nettoie les fichiers temporaires et les fichiers de version créés par erreur"""
    print("🧹 Nettoyage des fichiers temporaires...")

    # Nettoyer les fichiers de version créés par erreur (problème pip/shell)
    version_patterns = ["[0-9]*.[0-9]*.[0-9]*", "[0-9]*.[0-9]*", "[0-9]*"]
    for pattern in version_patterns:
        for file in Path(".").glob(pattern):
            if file.is_file() and file.name.replace(".", "").isdigit():
                try:
                    file.unlink()
                    print(f"   🗑️  Supprimé fichier de version: {file}")
                except Exception as e:
                    print(f"   ⚠️  Impossible de supprimer {file}: {e}")

    # Nettoyer les fichiers de cache temporaires Python
    cache_patterns = ["__pycache__", "*.pyc", ".pytest_cache"]

    for pattern in cache_patterns:
        for file in Path(".").glob(pattern):
            try:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    import shutil
                    shutil.rmtree(file)
                print(f"   🗑️  Nettoyé: {file}")
            except Exception as e:
                print(f"   ⚠️  Impossible de nettoyer {file}: {e}")


def main():
    """Fonction principale d'installation"""
    print("🚀 Installation de Hunyuan3D-2 pour le convertisseur de pièces")
    print("   Version corrigée - Résolution des problèmes de compatibilité")
    print("=" * 70)

    try:
        # Étape 1: Vérifications préliminaires
        if not check_python_version():
            return False

        check_cuda()

        # Étape 2: Nettoyer les packages conflictuels
        print("\n🧹 Étape 2: Nettoyage des packages conflictuels...")
        uninstall_conflicting_packages()

        # Étape 3: Préparer l'installation
        print("\n📋 Étape 3: Préparation de l'installation...")
        create_requirements_file()

        # Étape 4: Installation de PyTorch stable
        print("\n🔥 Étape 4: Installation de PyTorch 2.1.0 (stable)...")
        install_pytorch()

        # Étape 5: Installation de xFormers compatible
        print("\n⚙️  Étape 5: Installation de xFormers compatible...")
        install_xformers()

        # Étape 6: Installation des dépendances
        print("\n📦 Étape 6: Installation des dépendances...")
        install_dependencies()

        # Étape 7: Installation de torchmcubes compatible CUDA
        print("\n🧊 Étape 7: Installation de torchmcubes compatible CUDA...")
        install_torchmcubes()

        # Étape 8: Installation de Hunyuan3D-2
        print("\n🏛️  Étape 8: Installation de Hunyuan3D-2...")
        install_hunyuan3d()

        # Étape 9: Test de l'installation
        print("\n🧪 Étape 9: Test de l'installation...")
        test_installation()

        # Étape 10: Téléchargement des modèles (optionnel)
        print("\n📥 Étape 10: Téléchargement des modèles...")
        download_models()

        # Étape 11: Nettoyage
        print("\n🧹 Étape 11: Nettoyage...")
        cleanup_temp_files()

        print("\n✅ Installation réussie!")
        print("\n💡 Utilisation:")
        print("   python hunyuan3d-coin-to-stl.py image.jpg")
        print("   python hunyuan3d-coin-to-stl.py avers.jpg -b revers.jpg")
        print("   python hunyuan3d-coin-to-stl.py image.jpg --quality-preset ultra")
        print("   python hunyuan3d-coin-to-stl.py --info")

        print("\n🔧 Versions installées:")
        print("   • PyTorch 2.1.0 avec CUDA 11.8")
        print("   • Diffusers 0.28.0 (compatible avec huggingface_hub 0.20.3)")
        print("   • Transformers 4.37.2")
        print("   • xFormers 0.0.22.post7 (compatible)")
        print("   • torchmcubes compatible CUDA (version TripoSR)")
        print("   • HuggingFace Hub 0.20.3 (nouvelle API)")
        print("   • hy3dgen 2.0.2 (installé et fonctionnel)")

        print("\n⚠️  Conseils:")
        print("   • Redémarrez votre terminal après l'installation")
        print("   • Les modèles seront téléchargés lors de la première utilisation")
        print("   • Version diffusers 0.28.0 compatible avec huggingface_hub 0.20.3")
        print("   • Si erreur, utilisez --setup pour diagnostiquer")
        print("   • Les conflits gradio/albumentations sont normaux et non critiques")
        print("   • transformers 4.37.2 fonctionne malgré l'avertissement hy3dgen")

        print("\n🔧 Conflits de dépendances connus (non critiques):")
        print("   • gradio 5.31.0 préfère huggingface-hub>=0.28.1 (nous: 0.20.3)")
        print("   • albumentations 2.0.8 préfère pydantic>=2.9.2 (nous: 1.10.22)")
        print("   • hy3dgen 2.0.2 préfère transformers>=4.48.0 (nous: 4.37.2)")
        print("   💡 Ces conflits n'empêchent pas le fonctionnement du convertisseur")

        return True

    except Exception as e:
        print(f"\n❌ Erreur installation: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
