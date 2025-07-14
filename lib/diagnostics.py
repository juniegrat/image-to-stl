#!/usr/bin/env python3
"""
Module de diagnostic pour le convertisseur STL
Gère les tests, analyses et informations système
"""

import os
import sys
import shutil
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import contextlib
import io


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


def test_triposr_installation():
    """
    Test l'installation de TripoSR et ses dépendances

    Returns:
        bool: True si tout fonctionne correctement
    """
    print("\n🧪 Test de l'installation TripoSR:")

    try:
        # Test d'importation des modules TripoSR
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground
        print("   ✅ Modules TripoSR importés avec succès")

        # Test de chargement du modèle (sans téléchargement)
        try:
            # Vérifier si le modèle est disponible localement
            import huggingface_hub
            try:
                model_info = huggingface_hub.model_info("stabilityai/TripoSR")
                print("   ✅ Modèle TripoSR accessible sur HuggingFace")
            except Exception as e:
                print(f"   ⚠️  Modèle non accessible: {e}")

        except ImportError:
            print("   ❌ huggingface_hub non installé")

        return True

    except ImportError as e:
        print(f"   ❌ Erreur d'importation TripoSR: {e}")
        return False


def analyze_system_performance():
    """
    Analyse les performances du système pour l'optimisation

    Returns:
        dict: Informations sur les performances
    """
    performance_info = {
        'cpu_count': os.cpu_count(),
        'available_memory': None,
        'gpu_available': torch.cuda.is_available(),
        'gpu_memory': None,
        'disk_space': None,
        'recommendations': []
    }

    # Mémoire système
    try:
        import psutil
        memory = psutil.virtual_memory()
        performance_info['available_memory'] = memory.available / \
            (1024**3)  # GB

        if memory.available < 8 * 1024**3:  # Moins de 8GB
            performance_info['recommendations'].append(
                "Mémoire système faible (< 8GB), considérez fermer d'autres applications")
    except ImportError:
        performance_info['recommendations'].append(
            "Installez psutil pour le monitoring mémoire")

    # Mémoire GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(
            0).total_memory / (1024**3)
        performance_info['gpu_memory'] = gpu_memory

        if gpu_memory < 6:
            performance_info['recommendations'].append(
                "Mémoire GPU faible (< 6GB), utilisez des résolutions plus faibles")

    # Espace disque
    try:
        total, used, free = shutil.disk_usage('.')
        performance_info['disk_space'] = free / (1024**3)  # GB

        if free < 10 * 1024**3:  # Moins de 10GB
            performance_info['recommendations'].append(
                "Espace disque faible (< 10GB), nettoyez le disque")
    except Exception:
        pass

    return performance_info


def validate_dependencies():
    """
    Valide que toutes les dépendances critiques sont installées

    Returns:
        dict: Statut des dépendances
    """
    dependencies = {
        'torch': False,
        'torchvision': False,
        'PIL': False,
        'numpy': False,
        'pymeshlab': False,
        'rembg': False,
        'transformers': False,
        'diffusers': False,
        'opencv-python': False,
        'triposr': False
    }

    # Test des importations
    modules_to_test = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('PIL', 'PIL'),
        ('numpy', 'numpy'),
        ('pymeshlab', 'pymeshlab'),
        ('rembg', 'rembg'),
        ('transformers', 'transformers'),
        ('diffusers', 'diffusers'),
        ('opencv-python', 'cv2'),
        ('triposr', 'tsr.system')
    ]

    for dep_name, module_name in modules_to_test:
        try:
            __import__(module_name)
            dependencies[dep_name] = True
        except ImportError:
            dependencies[dep_name] = False

    return dependencies


def print_system_report():
    """
    Affiche un rapport complet du système
    """
    print("📊 Rapport système complet")
    print("=" * 60)

    # Diagnostic de base
    diagnostic_info()

    # Test TripoSR
    test_triposr_installation()

    # Analyse des performances
    print("\n⚡ Analyse des performances:")
    perf_info = analyze_system_performance()

    print(f"   CPU: {perf_info['cpu_count']} cœurs")
    if perf_info['available_memory']:
        print(f"   Mémoire disponible: {perf_info['available_memory']:.1f} GB")
    if perf_info['gpu_memory']:
        print(f"   Mémoire GPU: {perf_info['gpu_memory']:.1f} GB")
    if perf_info['disk_space']:
        print(f"   Espace disque: {perf_info['disk_space']:.1f} GB")

    # Recommandations
    if perf_info['recommendations']:
        print("\n💡 Recommandations:")
        for rec in perf_info['recommendations']:
            print(f"   • {rec}")

    # Validation des dépendances
    print("\n📦 Statut des dépendances:")
    deps = validate_dependencies()

    for dep, status in deps.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {dep}")

    # Résumé
    total_deps = len(deps)
    working_deps = sum(deps.values())
    print(
        f"\n📈 Résumé: {working_deps}/{total_deps} dépendances fonctionnelles")

    if working_deps == total_deps:
        print("🎉 Système prêt pour la conversion STL!")
    else:
        print("⚠️  Certaines dépendances manquent, utilisez install.py pour les installer")


def benchmark_conversion_time(test_image_path=None, iterations=3):
    """
    Benchmark les temps de conversion pour optimiser les performances

    Args:
        test_image_path: Chemin vers une image de test (optionnel)
        iterations: Nombre d'itérations pour la moyenne

    Returns:
        dict: Résultats du benchmark
    """
    if not test_image_path:
        # Créer une image de test simple
        test_image_path = "test_benchmark.png"
        test_image = Image.new('RGB', (512, 512), color='red')
        test_image.save(test_image_path)
        cleanup_test_image = True
    else:
        cleanup_test_image = False

    print(f"\n⏱️  Benchmark de conversion ({iterations} itérations):")

    try:
        from .converter import convert_coin_to_stl_safe
        import time

        times = []

        for i in range(iterations):
            print(f"   Itération {i+1}/{iterations}...")
            start_time = time.time()

            # Test de conversion
            result = convert_coin_to_stl_safe(
                test_image_path,
                output_dir=f"benchmark_output_{i}",
                render_video=False  # Désactiver la vidéo pour le benchmark
            )

            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)

            # Nettoyer le dossier de sortie
            import shutil
            try:
                shutil.rmtree(f"benchmark_output_{i}")
            except:
                pass

        # Nettoyer l'image de test si créée
        if cleanup_test_image:
            Path(test_image_path).unlink()

        # Calculer les statistiques
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        benchmark_results = {
            'average_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'iterations': iterations,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        print(f"   Temps moyen: {avg_time:.1f}s")
        print(f"   Temps minimum: {min_time:.1f}s")
        print(f"   Temps maximum: {max_time:.1f}s")
        print(f"   Périphérique: {benchmark_results['device'].upper()}")

        return benchmark_results

    except Exception as e:
        print(f"   ❌ Erreur durante le benchmark: {e}")
        return {}
