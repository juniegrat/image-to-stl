#!/usr/bin/env python3
"""
Module utilitaire pour le convertisseur STL
Contient les fonctions utilitaires générales et helpers
"""

import torch
import time
from pathlib import Path


def clear_gpu_memory():
    """Nettoie la mémoire GPU pour optimiser l'utilisation"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_render_params(args):
    """
    Crée les paramètres de rendu basés sur les arguments de ligne de commande

    Args:
        args: Arguments de ligne de commande

    Returns:
        dict: Paramètres de rendu
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

    Args:
        args: Arguments de ligne de commande
    """
    print(f"\n🎬 Paramètres de rendu:")
    print(f"   Résolution: {args.render_resolution}x{args.render_resolution}")
    print(f"   Nombre de vues: {args.render_views}")
    print(f"   Élévation: {args.render_elevation}°")
    print(f"   Distance caméra: {args.render_distance}")
    print(f"   Champ de vision: {args.render_fov}°")


def format_time(seconds):
    """
    Formate le temps en secondes dans un format lisible

    Args:
        seconds: Temps en secondes

    Returns:
        str: Temps formaté
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def format_file_size(bytes_size):
    """
    Formate la taille de fichier dans une unité lisible

    Args:
        bytes_size: Taille en octets

    Returns:
        str: Taille formatée
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def validate_file_path(file_path, extensions=None):
    """
    Valide un chemin de fichier et vérifie son extension

    Args:
        file_path: Chemin vers le fichier
        extensions: Liste des extensions autorisées (optionnel)

    Returns:
        tuple: (bool, str) - (valide, message d'erreur)
    """
    path = Path(file_path)

    if not path.exists():
        return False, f"Le fichier '{file_path}' n'existe pas"

    if not path.is_file():
        return False, f"'{file_path}' n'est pas un fichier"

    if extensions:
        if not any(file_path.lower().endswith(ext.lower()) for ext in extensions):
            return False, f"Extension de fichier non supportée. Extensions autorisées: {', '.join(extensions)}"

    return True, "Fichier valide"


def ensure_output_directory(output_dir):
    """
    S'assure que le répertoire de sortie existe

    Args:
        output_dir: Chemin vers le répertoire de sortie

    Returns:
        Path: Chemin du répertoire créé
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_unique_filename(base_path, extension=""):
    """
    Génère un nom de fichier unique en ajoutant un suffixe numérique si nécessaire

    Args:
        base_path: Chemin de base
        extension: Extension du fichier (optionnel)

    Returns:
        Path: Chemin unique
    """
    base_path = Path(base_path)

    if extension and not extension.startswith('.'):
        extension = f'.{extension}'

    if extension:
        base_path = base_path.with_suffix(extension)

    if not base_path.exists():
        return base_path

    counter = 1
    while True:
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent

        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def progress_bar(current, total, width=50, prefix="Progress"):
    """
    Affiche une barre de progression simple

    Args:
        current: Valeur actuelle
        total: Valeur totale
        width: Largeur de la barre
        prefix: Préfixe à afficher
    """
    if total == 0:
        return

    percentage = current / total
    filled_width = int(width * percentage)

    bar = '█' * filled_width + '-' * (width - filled_width)
    print(f'\r{prefix}: |{bar}| {percentage:.1%} ({current}/{total})',
          end='', flush=True)

    if current == total:
        print()  # Nouvelle ligne à la fin


def timer_decorator(func):
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction

    Args:
        func: Fonction à décorer

    Returns:
        function: Fonction décorée
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        duration = end_time - start_time
        print(f"⏱️  {func.__name__} terminé en {format_time(duration)}")

        return result

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def memory_usage_monitor():
    """
    Monitore l'utilisation mémoire GPU et système

    Returns:
        dict: Informations sur l'utilisation mémoire
    """
    memory_info = {
        'gpu_available': torch.cuda.is_available(),
        'gpu_memory': None,
        'gpu_usage': None,
        'system_memory': None
    }

    # Mémoire GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_cached = torch.cuda.memory_reserved(0)

        memory_info['gpu_memory'] = {
            'total': gpu_memory / (1024**3),  # GB
            'allocated': gpu_allocated / (1024**3),  # GB
            'cached': gpu_cached / (1024**3),  # GB
            'free': (gpu_memory - gpu_allocated) / (1024**3)  # GB
        }

        memory_info['gpu_usage'] = gpu_allocated / gpu_memory

    # Mémoire système
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_info['system_memory'] = {
            'total': memory.total / (1024**3),  # GB
            'available': memory.available / (1024**3),  # GB
            'used': memory.used / (1024**3),  # GB
            'percentage': memory.percent
        }
    except ImportError:
        memory_info['system_memory'] = "psutil non installé"

    return memory_info


def print_memory_usage():
    """
    Affiche l'utilisation mémoire actuelle
    """
    memory_info = memory_usage_monitor()

    print(f"\n💾 Utilisation mémoire:")

    # Mémoire GPU
    if memory_info['gpu_available'] and memory_info['gpu_memory']:
        gpu_mem = memory_info['gpu_memory']
        print(f"   GPU: {gpu_mem['allocated']:.1f}GB / {gpu_mem['total']:.1f}GB "
              f"({memory_info['gpu_usage']:.1%})")
        print(f"   GPU libre: {gpu_mem['free']:.1f}GB")
    else:
        print("   GPU: Non disponible")

    # Mémoire système
    if isinstance(memory_info['system_memory'], dict):
        sys_mem = memory_info['system_memory']
        print(f"   Système: {sys_mem['used']:.1f}GB / {sys_mem['total']:.1f}GB "
              f"({sys_mem['percentage']:.1f}%)")
        print(f"   Système libre: {sys_mem['available']:.1f}GB")
    else:
        print(f"   Système: {memory_info['system_memory']}")


def cleanup_temp_files(temp_dir="temp", pattern="*"):
    """
    Nettoie les fichiers temporaires

    Args:
        temp_dir: Répertoire temporaire
        pattern: Motif des fichiers à supprimer
    """
    temp_path = Path(temp_dir)

    if not temp_path.exists():
        return

    try:
        import shutil
        removed_count = 0

        for file_path in temp_path.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                removed_count += 1
            elif file_path.is_dir():
                shutil.rmtree(file_path)
                removed_count += 1

        if removed_count > 0:
            print(f"🧹 {removed_count} fichiers temporaires supprimés")

    except Exception as e:
        print(f"⚠️  Erreur lors du nettoyage: {e}")


def create_backup(file_path, backup_dir="backups"):
    """
    Crée une sauvegarde d'un fichier

    Args:
        file_path: Chemin vers le fichier à sauvegarder
        backup_dir: Répertoire de sauvegarde

    Returns:
        Path: Chemin vers la sauvegarde créée
    """
    import shutil
    from datetime import datetime

    source_path = Path(file_path)
    if not source_path.exists():
        return None

    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)

    # Nom de sauvegarde avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{source_path.stem}_{timestamp}{source_path.suffix}"
    backup_file = backup_path / backup_name

    try:
        shutil.copy2(source_path, backup_file)
        print(f"💾 Sauvegarde créée: {backup_file}")
        return backup_file
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        return None


def log_operation(operation, details=None, log_file="operations.log"):
    """
    Enregistre une opération dans un fichier de log

    Args:
        operation: Description de l'opération
        details: Détails additionnels (optionnel)
        log_file: Nom du fichier de log
    """
    from datetime import datetime

    log_path = Path(log_file)

    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {operation}")
            if details:
                f.write(f" - {details}")
            f.write("\n")
    except Exception as e:
        print(f"⚠️  Erreur lors de l'écriture du log: {e}")


def get_system_info():
    """
    Récupère les informations système de base

    Returns:
        dict: Informations système
    """
    import platform
    import sys

    info = {
        'os': platform.system(),
        'os_version': platform.release(),
        'architecture': platform.architecture()[0],
        'python_version': sys.version,
        'python_executable': sys.executable,
        'torch_version': None,
        'cuda_available': False,
        'cuda_version': None
    }

    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
    except ImportError:
        pass

    return info


def print_system_info():
    """
    Affiche les informations système
    """
    info = get_system_info()

    print(f"\n🖥️  Informations système:")
    print(f"   OS: {info['os']} {info['os_version']}")
    print(f"   Architecture: {info['architecture']}")
    print(f"   Python: {info['python_version'].split()[0]}")

    if info['torch_version']:
        print(f"   PyTorch: {info['torch_version']}")
        if info['cuda_available']:
            print(f"   CUDA: {info['cuda_version']}")
        else:
            print(f"   CUDA: Non disponible")
    else:
        print(f"   PyTorch: Non installé")
