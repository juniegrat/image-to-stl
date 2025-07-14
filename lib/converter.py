#!/usr/bin/env python3
"""
Module de conversion STL pour le convertisseur d'images
Gère la conversion d'images en modèles STL 3D via TripoSR
"""

import os
import sys
import subprocess
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import pymeshlab as pymesh

from .setup import setup_triposr
from .image_processor import detect_and_convert_image_format, process_image_for_triposr, save_processed_images
from .utils import clear_gpu_memory


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
        processed_main = process_image_for_triposr(
            image, remove_bg=remove_bg, image_size=image_size)

        processed_images = [processed_main]

        if has_reverse_image:
            reverse_image, _ = detect_and_convert_image_format(
                reverse_image_path)
            processed_reverse = process_image_for_triposr(
                reverse_image, remove_bg=remove_bg, image_size=image_size)
            processed_images.append(processed_reverse)

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

    # Sauvegarder les images traitées
    output_dir = Path(output_dir)
    save_processed_images(processed_images, output_dir)

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

                image_dir = output_dir / "0"
                for ri, render_image in enumerate(render_images[0]):
                    render_image.save(image_dir / f"render_{ri:03d}.png")

                from tsr.utils import save_video
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
        image_dir = output_dir / "0"
        mesh_file = image_dir / "mesh.obj"
        meshes[0].export(str(mesh_file))
        print(f"   ✅ Maillage 3D sauvegardé: {mesh_file}")

        # Conversion en STL avec PyMeshLab
        stl_path = convert_mesh_to_stl(
            mesh_file, output_dir, Path(input_image_path).stem)

        elapsed_time = time.time() - start_time
        print(f"✅ Fichier STL généré: {stl_path}")
        print(f"⏱️  Temps total: {elapsed_time:.1f} secondes")

        if has_reverse_image:
            print(f"🎯 Modèle créé avec 2 vues (recto + verso)")

        return str(stl_path)

    except Exception as e:
        print(f"❌ Erreur génération 3D: {e}")
        return None


def convert_mesh_to_stl(mesh_file, output_dir, base_name):
    """
    Convertit un fichier de maillage OBJ en STL avec post-processing

    Args:
        mesh_file: Chemin vers le fichier OBJ
        output_dir: Répertoire de sortie
        base_name: Nom de base pour le fichier STL

    Returns:
        Path: Chemin vers le fichier STL généré
    """
    print("   📦 Conversion optimisée en STL...")
    try:
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
        stl_file = output_dir / f"{base_name}.stl"
        ms.save_current_mesh(str(stl_file))

        return stl_file

    except Exception as e:
        print(f"❌ Erreur post-processing: {e}")
        return None


def validate_stl_file(stl_path):
    """
    Valide un fichier STL et retourne des informations sur le maillage

    Args:
        stl_path: Chemin vers le fichier STL

    Returns:
        dict: Informations sur le maillage
    """
    try:
        ms = pymesh.MeshSet()
        ms.load_new_mesh(str(stl_path))
        mesh = ms.current_mesh()

        info = {
            'vertices': mesh.vertex_number(),
            'faces': mesh.face_number(),
            'edges': mesh.edge_number(),
            'is_manifold': mesh.is_manifold(),
            'is_watertight': mesh.is_watertight(),
            'file_size': Path(stl_path).stat().st_size / 1024 / 1024,  # MB
        }

        # Calculer les dimensions
        bbox = mesh.bounding_box()
        info['dimensions'] = {
            'min': bbox.min(),
            'max': bbox.max(),
            'size': bbox.diagonal()
        }

        return info

    except Exception as e:
        print(f"❌ Erreur validation STL: {e}")
        return {}


def optimize_stl_for_printing(stl_path, output_path=None, target_faces=None):
    """
    Optimise un fichier STL pour l'impression 3D

    Args:
        stl_path: Chemin vers le fichier STL d'entrée
        output_path: Chemin de sortie (optionnel)
        target_faces: Nombre cible de faces pour la simplification

    Returns:
        str: Chemin vers le fichier optimisé
    """
    if output_path is None:
        output_path = str(stl_path).replace('.stl', '_optimized.stl')

    try:
        ms = pymesh.MeshSet()
        ms.load_new_mesh(str(stl_path))

        print("   🔧 Optimisation pour impression 3D...")

        # Nettoyage avancé
        try:
            ms.apply_filter('meshing_remove_duplicate_vertices')
            ms.apply_filter('meshing_remove_unreferenced_vertices')
            ms.apply_filter('meshing_close_holes', maxholesize=30)
        except:
            print("   ⚠️  Certains filtres de nettoyage non disponibles")

        # Simplification si nécessaire
        if target_faces and ms.current_mesh().face_number() > target_faces:
            try:
                ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                                targetfacenum=target_faces)
                print(f"   📉 Maillage simplifié à {target_faces} faces")
            except:
                print("   ⚠️  Simplification non disponible")

        # Lissage final
        try:
            ms.apply_filter('apply_coord_taubin_smoothing', stepsmoothnum=1)
        except:
            pass

        ms.save_current_mesh(str(output_path))
        print(f"   ✅ STL optimisé: {output_path}")

        return str(output_path)

    except Exception as e:
        print(f"❌ Erreur optimisation: {e}")
        return str(stl_path)  # Retourner le fichier original en cas d'erreur
