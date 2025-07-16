#!/usr/bin/env python3
"""
Traitement d'images pour Hunyuan3D-2mv
Gère la suppression d'arrière-plan, redimensionnement et préparation des images
"""

import numpy as np
from PIL import Image
from typing import List, Optional, Union, Tuple
import rembg
from pathlib import Path


class ImageProcessor:
    """
    Processeur d'images spécialisé pour Hunyuan3D-2mv
    Optimisé pour les pièces numismatiques et objets circulaires
    """

    def __init__(self, rembg_model: str = 'u2net'):
        """
        Initialise le processeur d'images

        Args:
            rembg_model: Modèle rembg à utiliser ('u2net', 'silueta', etc.)
        """
        self.rembg_session = None
        self.rembg_model = rembg_model
        self._init_rembg()

    def _init_rembg(self):
        """Initialise la session rembg"""
        try:
            self.rembg_session = rembg.new_session(self.rembg_model)
            print(f"✅ Session rembg initialisée: {self.rembg_model}")
        except Exception as e:
            print(f"⚠️  Session rembg non disponible: {e}")
            self.rembg_session = None

    def remove_background(
        self,
        image: Image.Image,
        force: bool = False,
        **rembg_kwargs,
    ) -> Image.Image:
        """
        Remove background from image using rembg (basé sur TripoSR)

        Args:
            image: PIL Image
            force: Force background removal even if image has alpha
            **rembg_kwargs: Additional arguments for rembg.remove

        Returns:
            Image with background removed (RGBA format)
        """
        do_remove = True
        if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
            do_remove = False
        do_remove = do_remove or force

        if do_remove and self.rembg_session:
            # Utiliser la session configurée
            image = rembg.remove(image, session=self.rembg_session)
        elif do_remove:
            # Fallback sans session
            filtered_kwargs = {k: v for k,
                               v in rembg_kwargs.items() if k != 'session'}
            image = rembg.remove(image, **filtered_kwargs)

        return image

    def resize_foreground(
        self,
        image: Image.Image,
        ratio: float,
    ) -> Image.Image:
        """
        Resize foreground object to specific ratio within image (basé sur TripoSR)

        Args:
            image: PIL Image with alpha channel (RGBA)
            ratio: Desired ratio of foreground to image size

        Returns:
            Resized image with foreground at specified ratio
        """
        image = np.array(image)
        assert image.shape[-1] == 4
        alpha = np.where(image[..., 3] > 0)
        y1, y2, x1, x2 = (
            alpha[0].min(),
            alpha[0].max(),
            alpha[1].min(),
            alpha[1].max(),
        )
        # crop the foreground
        fg = image[y1:y2, x1:x2]
        # pad to square
        size = max(fg.shape[0], fg.shape[1])
        ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
        ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
        new_image = np.pad(
            fg,
            ((ph0, ph1), (pw0, pw1), (0, 0)),
            mode="constant",
            constant_values=((0, 0), (0, 0), (0, 0)),
        )

        # compute padding according to the ratio
        new_size = int(new_image.shape[0] / ratio)
        # pad to size, double side
        ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
        ph1, pw1 = new_size - size - ph0, new_size - size - pw0
        new_image = np.pad(
            new_image,
            ((ph0, ph1), (pw0, pw1), (0, 0)),
            mode="constant",
            constant_values=((0, 0), (0, 0), (0, 0)),
        )
        new_image = Image.fromarray(new_image)
        return new_image

    def prepare_single_image(
        self,
        image_path: Union[str, Path],
        target_size: int = 1024,
        remove_bg: bool = False,
        foreground_ratio: float = 0.9,
        enhance_contrast: bool = False
    ) -> Optional[Image.Image]:
        """
        Prépare une seule image pour le traitement

        Args:
            image_path: Chemin vers l'image
            target_size: Taille cible (carré)
            remove_bg: Supprimer l'arrière-plan
            foreground_ratio: Ratio du premier plan après suppression d'arrière-plan
            enhance_contrast: Améliorer le contraste

        Returns:
            Image préparée ou None en cas d'erreur
        """
        try:
            # Charger l'image
            image = Image.open(image_path).convert('RGB')
            print(f"   📄 Image chargée: {image.size}")

            # Supprimer l'arrière-plan si demandé
            if remove_bg:
                print("   🔄 Suppression arrière-plan...")
                image = self.remove_background(image)

                # Traitement de l'image avec canal alpha
                if image.mode == 'RGBA':
                    image = self.resize_foreground(image, foreground_ratio)
                    image_array = np.array(image).astype(np.float32) / 255.0
                    # Convertir en RGB avec fond gris
                    image_array = image_array[:, :, :3] * image_array[:,
                                                                      :, 3:4] + (1 - image_array[:, :, 3:4]) * 0.5
                    image = Image.fromarray(
                        (image_array * 255.0).astype(np.uint8))

            # Améliorer le contraste si demandé
            if enhance_contrast:
                image = self.enhance_contrast(image)

            # Redimensionner
            if image.size != (target_size, target_size):
                image = image.resize(
                    (target_size, target_size), Image.Resampling.LANCZOS)

            print(f"   ✅ Image préparée: {image.size}")
            return image

        except Exception as e:
            print(f"   ❌ Erreur préparation image: {e}")
            return None

    def prepare_multiple_images(
        self,
        image_paths: List[Union[str, Path]],
        target_size: int = 1024,
        remove_bg: bool = False,
        foreground_ratio: float = 0.9,
        enhance_contrast: bool = False
    ) -> List[Image.Image]:
        """
        Prépare plusieurs images pour le traitement

        Args:
            image_paths: Liste des chemins vers les images
            target_size: Taille cible (carré)
            remove_bg: Supprimer l'arrière-plan
            foreground_ratio: Ratio du premier plan
            enhance_contrast: Améliorer le contraste

        Returns:
            Liste des images préparées
        """
        print(f"🖼️  Préparation de {len(image_paths)} image(s)...")

        images = []
        for i, path in enumerate(image_paths):
            print(f"   📄 Image {i+1}/{len(image_paths)}: {path}")
            image = self.prepare_single_image(
                path, target_size, remove_bg, foreground_ratio, enhance_contrast
            )
            if image:
                images.append(image)

        print(f"✅ {len(images)}/{len(image_paths)} images préparées")
        return images

    def enhance_contrast(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """
        Améliore le contraste de l'image

        Args:
            image: Image PIL
            factor: Facteur d'amélioration du contraste (>1 = plus de contraste)

        Returns:
            Image avec contraste amélioré
        """
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def enhance_sharpness(self, image: Image.Image, factor: float = 1.1) -> Image.Image:
        """
        Améliore la netteté de l'image

        Args:
            image: Image PIL
            factor: Facteur d'amélioration de la netteté (>1 = plus net)

        Returns:
            Image avec netteté améliorée
        """
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    def adjust_brightness(self, image: Image.Image, factor: float = 1.0) -> Image.Image:
        """
        Ajuste la luminosité de l'image

        Args:
            image: Image PIL
            factor: Facteur de luminosité (1.0 = normal, >1 = plus lumineux)

        Returns:
            Image avec luminosité ajustée
        """
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def crop_square_center(self, image: Image.Image) -> Image.Image:
        """
        Recadre l'image en carré depuis le centre

        Args:
            image: Image PIL

        Returns:
            Image carrée centrée
        """
        width, height = image.size
        min_dim = min(width, height)

        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim

        return image.crop((left, top, right, bottom))

    def add_padding(
        self,
        image: Image.Image,
        padding_ratio: float = 0.1,
        background_color: tuple = (255, 255, 255)
    ) -> Image.Image:
        """
        Ajoute du padding autour de l'image

        Args:
            image: Image PIL
            padding_ratio: Ratio de padding par rapport à la taille de l'image
            background_color: Couleur de fond du padding

        Returns:
            Image avec padding
        """
        width, height = image.size
        padding_x = int(width * padding_ratio)
        padding_y = int(height * padding_ratio)

        new_width = width + 2 * padding_x
        new_height = height + 2 * padding_y

        # Créer une nouvelle image avec padding
        new_image = Image.new('RGB', (new_width, new_height), background_color)
        new_image.paste(image, (padding_x, padding_y))

        return new_image

    def create_collage(
        self,
        images: List[Image.Image],
        arrangement: str = 'horizontal'
    ) -> Image.Image:
        """
        Crée un collage à partir de plusieurs images

        Args:
            images: Liste d'images PIL
            arrangement: 'horizontal', 'vertical', ou 'grid'

        Returns:
            Image collage
        """
        if not images:
            raise ValueError("Aucune image fournie pour le collage")

        if len(images) == 1:
            return images[0]

        # S'assurer que toutes les images ont la même taille
        reference_size = images[0].size
        resized_images = [img.resize(reference_size, Image.Resampling.LANCZOS)
                          if img.size != reference_size else img for img in images]

        if arrangement == 'horizontal':
            total_width = sum(img.size[0] for img in resized_images)
            max_height = max(img.size[1] for img in resized_images)
            collage = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for img in resized_images:
                collage.paste(img, (x_offset, 0))
                x_offset += img.size[0]

        elif arrangement == 'vertical':
            max_width = max(img.size[0] for img in resized_images)
            total_height = sum(img.size[1] for img in resized_images)
            collage = Image.new('RGB', (max_width, total_height))

            y_offset = 0
            for img in resized_images:
                collage.paste(img, (0, y_offset))
                y_offset += img.size[1]

        elif arrangement == 'grid':
            n_images = len(resized_images)
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))

            img_width, img_height = reference_size
            total_width = cols * img_width
            total_height = rows * img_height

            collage = Image.new('RGB', (total_width, total_height))

            for i, img in enumerate(resized_images):
                row = i // cols
                col = i % cols
                x = col * img_width
                y = row * img_height
                collage.paste(img, (x, y))

        else:
            raise ValueError(
                f"Arrangement '{arrangement}' non supporté. Utilisez 'horizontal', 'vertical', ou 'grid'")

        return collage

    def save_processed_images(
        self,
        images: List[Image.Image],
        output_dir: Union[str, Path],
        prefix: str = 'processed_',
        format: str = 'PNG'
    ) -> List[str]:
        """
        Sauvegarde les images traitées

        Args:
            images: Liste d'images PIL
            output_dir: Répertoire de sortie
            prefix: Préfixe pour les noms de fichiers
            format: Format de fichier ('PNG', 'JPEG', etc.)

        Returns:
            Liste des chemins des fichiers sauvegardés
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, image in enumerate(images):
            filename = f"{prefix}{i:03d}.{format.lower()}"
            filepath = output_dir / filename
            image.save(filepath, format=format)
            saved_paths.append(str(filepath))

        print(f"✅ {len(images)} images sauvegardées dans {output_dir}")
        return saved_paths

    def analyze_image_stats(self, image: Image.Image) -> dict:
        """
        Analyse les statistiques d'une image

        Args:
            image: Image PIL

        Returns:
            Dictionnaire avec les statistiques
        """
        # Convertir en array numpy
        img_array = np.array(image)

        stats = {
            'size': image.size,
            'mode': image.mode,
            'format': image.format,
            'has_transparency': image.mode in ('RGBA', 'LA'),
        }

        if len(img_array.shape) >= 3:
            # Statistiques par canal
            stats['channels'] = img_array.shape[2] if len(
                img_array.shape) == 3 else 1
            stats['mean_rgb'] = np.mean(img_array, axis=(0, 1)).tolist()
            stats['std_rgb'] = np.std(img_array, axis=(0, 1)).tolist()
            stats['min_rgb'] = np.min(img_array, axis=(0, 1)).tolist()
            stats['max_rgb'] = np.max(img_array, axis=(0, 1)).tolist()
        else:
            # Image en niveaux de gris
            stats['channels'] = 1
            stats['mean'] = float(np.mean(img_array))
            stats['std'] = float(np.std(img_array))
            stats['min'] = float(np.min(img_array))
            stats['max'] = float(np.max(img_array))

        return stats

    def get_info(self) -> dict:
        """Retourne des informations sur le processeur d'images"""
        return {
            'name': 'Hunyuan3D Image Processor',
            'description': 'Traitement d\'images optimisé pour Hunyuan3D-2mv',
            'rembg_model': self.rembg_model,
            'rembg_available': self.rembg_session is not None,
            'features': [
                'Suppression d\'arrière-plan avec rembg',
                'Redimensionnement intelligent du premier plan',
                'Amélioration de contraste et netteté',
                'Recadrage et padding automatiques',
                'Création de collages',
                'Analyse des statistiques d\'images',
                'Support formats multiples (PNG, JPEG, etc.)',
                'Optimisé pour pièces numismatiques'
            ],
            'functions': [
                'remove_background: Suppression d\'arrière-plan',
                'resize_foreground: Redimensionnement du premier plan',
                'prepare_single_image: Préparation complète d\'une image',
                'prepare_multiple_images: Traitement par lot',
                'enhance_contrast: Amélioration du contraste',
                'enhance_sharpness: Amélioration de la netteté',
                'crop_square_center: Recadrage carré centré',
                'create_collage: Création de collages',
                'analyze_image_stats: Analyse statistique'
            ]
        }


# Instance globale du processeur
image_processor = ImageProcessor()


def get_image_processor() -> ImageProcessor:
    """Retourne l'instance globale du processeur d'images"""
    return image_processor


def prepare_images(
    image_paths: List[Union[str, Path]],
    target_size: int = 1024,
    remove_bg: bool = False,
    foreground_ratio: float = 0.9
) -> List[Image.Image]:
    """
    Fonction de convenance pour préparer des images

    Args:
        image_paths: Liste des chemins vers les images
        target_size: Taille cible
        remove_bg: Supprimer l'arrière-plan
        foreground_ratio: Ratio du premier plan

    Returns:
        Liste des images préparées
    """
    return image_processor.prepare_multiple_images(
        image_paths, target_size, remove_bg, foreground_ratio
    )


def remove_background_from_image(image: Image.Image) -> Image.Image:
    """
    Fonction de convenance pour supprimer l'arrière-plan

    Args:
        image: Image PIL

    Returns:
        Image sans arrière-plan
    """
    return image_processor.remove_background(image)


def get_image_processing_info() -> dict:
    """Fonction de convenance pour obtenir les informations du processeur"""
    return image_processor.get_info()
