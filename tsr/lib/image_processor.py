#!/usr/bin/env python3
"""
Module de traitement d'images pour le convertisseur STL
Gère la détection de formats, conversion et préparation des images
"""

from PIL import Image
import numpy as np
from pathlib import Path
import cv2


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


def process_image_for_triposr(image, remove_bg=True, foreground_ratio=0.85, image_size=512):
    """
    Traite une image pour l'utilisation avec TripoSR

    Args:
        image: PIL.Image à traiter
        remove_bg: Si True, supprime l'arrière-plan
        foreground_ratio: Ratio de l'objet dans l'image
        image_size: Taille de l'image de sortie

    Returns:
        PIL.Image: Image traitée
    """
    try:
        # Redimensionner l'image
        image_resized = image.resize((image_size, image_size))

        # Supprimer l'arrière-plan si demandé
        if remove_bg:
            try:
                import rembg
                from tsr.utils import remove_background, resize_foreground

                rembg_session = rembg.new_session()
                processed_image = remove_background(
                    image_resized, rembg_session)
                processed_image = resize_foreground(
                    processed_image, foreground_ratio)
            except ImportError:
                print("⚠️  rembg non disponible, utilisation de l'image originale")
                processed_image = image_resized
        else:
            processed_image = image_resized

        # Traiter les images RGBA
        if processed_image.mode == "RGBA":
            processed_image = np.array(
                processed_image).astype(np.float32) / 255.0
            processed_image = processed_image[:, :, :3] * processed_image[:, :, 3:4] + \
                (1 - processed_image[:, :, 3:4]) * 0.5
            processed_image = Image.fromarray(
                (processed_image * 255.0).astype(np.uint8))

        return processed_image

    except Exception as e:
        print(f"⚠️  Erreur traitement image: {e}, utilisation image originale")
        return image.resize((image_size, image_size))


def save_processed_images(processed_images, output_dir):
    """
    Sauvegarde les images traitées dans le répertoire de sortie

    Args:
        processed_images: Liste des images traitées
        output_dir: Répertoire de sortie
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "0"
    image_dir.mkdir(exist_ok=True)

    for i, proc_img in enumerate(processed_images):
        suffix = "" if i == 0 else f"_reverse"
        proc_img.save(image_dir / f"input{suffix}.png")
        print(f"   💾 Image sauvegardée: input{suffix}.png")


def analyze_image_quality(image_path):
    """
    Analyse la qualité d'une image et suggère des améliorations

    Args:
        image_path: Chemin vers l'image à analyser

    Returns:
        dict: Informations sur la qualité
    """
    try:
        image = Image.open(image_path)
        width, height = image.size
        mode = image.mode

        # Convertir en array numpy pour analyse
        img_array = np.array(image)

        analysis = {
            'width': width,
            'height': height,
            'mode': mode,
            'aspect_ratio': width / height,
            'file_size': Path(image_path).stat().st_size / 1024 / 1024,  # MB
        }

        # Analyser la luminosité
        if len(img_array.shape) == 3:
            # Image couleur
            brightness = np.mean(img_array)
            analysis['brightness'] = brightness

            # Détecter les zones très sombres ou très claires
            dark_pixels = np.sum(img_array < 50) / img_array.size
            bright_pixels = np.sum(img_array > 200) / img_array.size

            analysis['dark_ratio'] = dark_pixels
            analysis['bright_ratio'] = bright_pixels

        # Recommandations
        recommendations = []

        if width < 512 or height < 512:
            recommendations.append(
                "Image de faible résolution, considérez une image plus grande")

        if analysis['aspect_ratio'] < 0.8 or analysis['aspect_ratio'] > 1.2:
            recommendations.append(
                "Ratio d'aspect non carré, l'image sera déformée")

        if analysis.get('dark_ratio', 0) > 0.7:
            recommendations.append("Image très sombre, considérez l'éclaircir")

        if analysis.get('bright_ratio', 0) > 0.7:
            recommendations.append("Image très claire, considérez l'assombrir")

        analysis['recommendations'] = recommendations

        return analysis

    except Exception as e:
        print(f"❌ Erreur analyse image: {e}")
        return {}


def get_supported_formats():
    """
    Retourne la liste des formats d'images supportés

    Returns:
        list: Liste des extensions supportées
    """
    return ['.png', '.webp', '.jpeg', '.jpg', '.bmp', '.tiff', '.tif']
