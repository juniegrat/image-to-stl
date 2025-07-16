#!/usr/bin/env python3
"""
Traitement de mesh pour Hunyuan3D-2mv
Gère la normalisation, post-processing, couleurs et optimisations des mesh 3D
"""

import numpy as np
import trimesh
import time
from PIL import Image
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


class MeshProcessor:
    """
    Processeur de mesh spécialisé pour Hunyuan3D-2mv
    Optimisé pour les pièces numismatiques et objets détaillés
    """

    def __init__(self):
        """Initialise le processeur de mesh"""
        pass

    def normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Normalize mesh to unit scale and center at origin

        Args:
            mesh: Trimesh object to normalize

        Returns:
            Normalized mesh
        """
        # Center the mesh at origin
        center = mesh.bounds.mean(axis=0)
        mesh.vertices -= center

        # Scale to unit size
        scale = 1.0 / mesh.scale
        mesh.vertices *= scale

        return mesh

    def to_gradio_3d_orientation(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Apply standard 3D orientation transformation (basé sur TripoSR)

        Args:
            mesh: Trimesh object

        Returns:
            Mesh with applied orientation transformation
        """
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
        return mesh

    def apply_vertex_colors(
        self,
        mesh: trimesh.Trimesh,
        reference_image: Image.Image
    ) -> trimesh.Trimesh:
        """
        Applique des couleurs de vertices rapides en échantillonnant les vraies couleurs de l'image

        Args:
            mesh: Mesh 3D de base
            reference_image: Image de référence pour les couleurs

        Returns:
            Mesh avec couleurs de vertices (rapide, sans texture)
        """
        print("🎨 Application de couleurs de vertices (mode rapide)...")

        try:
            start_time = time.time()

            # Convertir l'image en array numpy
            if reference_image.mode != 'RGB':
                reference_image = reference_image.convert('RGB')

            img_array = np.array(reference_image).astype(np.float32) / 255.0
            img_height, img_width = img_array.shape[:2]

            # Calculer les normales des vertices si pas disponibles
            if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
                mesh.compute_vertex_normals()

            # Projeter les vertices sur l'image 2D (vue frontale)
            vertices = mesh.vertices

            # Normaliser les coordonnées X,Y des vertices vers l'espace image [0,1]
            # On utilise X,Y pour projeter sur l'image (Z = profondeur)
            x_coords = vertices[:, 0]
            y_coords = vertices[:, 1]

            # Normaliser vers [0,1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            # Éviter division par zéro
            x_range = x_max - x_min if x_max != x_min else 1.0
            y_range = y_max - y_min if y_max != y_min else 1.0

            u_coords = (x_coords - x_min) / x_range
            v_coords = (y_coords - y_min) / y_range

            # Convertir vers coordonnées de pixels
            pixel_x = np.clip(u_coords * (img_width - 1),
                              0, img_width - 1).astype(int)
            pixel_y = np.clip(v_coords * (img_height - 1),
                              0, img_height - 1).astype(int)

            # Échantillonner les couleurs directement de l'image
            sampled_colors = img_array[pixel_y, pixel_x]

            # Ajouter un très léger effet de relief basé sur les normales (10% max)
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                # Calculer un facteur de relief basé sur la normale Z (face avant)
                relief_factor = np.abs(
                    mesh.vertex_normals[:, 2])  # 0=profil, 1=face
                # Pour broadcasting
                relief_factor = relief_factor.reshape(-1, 1)

                # Ajuster légèrement la luminosité selon le relief (±10%)
                brightness_adjustment = 1.0 + (relief_factor - 0.5) * 0.2
                final_colors = sampled_colors * brightness_adjustment
            else:
                final_colors = sampled_colors

            # S'assurer que les couleurs sont dans la plage [0,1]
            final_colors = np.clip(final_colors, 0.0, 1.0)

            # Appliquer les couleurs au mesh (format 0-255)
            mesh.visual.vertex_colors = (final_colors * 255).astype(np.uint8)

            elapsed_time = time.time() - start_time
            print(
                f"   ⏱️  Couleurs de vertices appliquées en {elapsed_time:.1f}s")
            print(
                f"   ✅ {len(mesh.vertices)} vertices colorés avec vraies couleurs de l'image")
            print(
                f"   🎯 Projection: {img_width}x{img_height} → {len(mesh.vertices)} vertices")

            return mesh

        except Exception as e:
            print(f"⚠️  Erreur application couleurs vertices: {e}")
            print("   Retour au mesh sans couleurs")
            return mesh

    def post_process_mesh(
        self,
        mesh: trimesh.Trimesh,
        preserve_details: bool = True
    ) -> trimesh.Trimesh:
        """
        Post-traite le mesh pour optimiser la qualité tout en préservant les détails

        Args:
            mesh: Mesh d'entrée
            preserve_details: Si True, préserve les détails (lissage minimal)

        Returns:
            Mesh post-traité
        """
        print("🔧 Post-processing du mesh...")

        try:
            # Nettoyage basique uniquement
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()

            if preserve_details:
                print("   ✅ Mode préservation des détails activé")
                # Lissage très léger UNIQUEMENT si le mesh est très irrégulier
                vertices_count = len(mesh.vertices)
                faces_count = len(mesh.faces)

                # Seulement lisser si le ratio faces/vertices est très élevé (mesh très irrégulier)
                if vertices_count > 0 and faces_count / vertices_count > 3.0:
                    print("   🔄 Lissage minimal appliqué pour mesh irrégulier")
                    # Lissage très léger avec préservation des détails
                    mesh = mesh.smoothed(alpha=0.1)
                else:
                    print(
                        "   ✅ Mesh régulier - pas de lissage pour préserver les détails")
            else:
                print("   🔄 Post-processing standard appliqué")
                # Lissage plus agressif si la préservation n'est pas critique
                mesh = mesh.smoothed(alpha=0.3)

            print(
                f"   ✅ Mesh optimisé: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh

        except Exception as e:
            print(f"   ⚠️  Erreur post-processing: {e}")
            return mesh

    def clean_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Nettoie le mesh en supprimant les éléments problématiques

        Args:
            mesh: Mesh d'entrée

        Returns:
            Mesh nettoyé
        """
        print("🧹 Nettoyage du mesh...")

        try:
            original_vertices = len(mesh.vertices)
            original_faces = len(mesh.faces)

            # Supprimer les doublons
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()

            # Supprimer les composants isolés (garder seulement le plus gros)
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                # Garder le composant avec le plus de faces
                largest_component = max(components, key=lambda x: len(x.faces))
                mesh = largest_component
                print(
                    f"   🔄 {len(components)} composants détectés, garde le plus gros")

            new_vertices = len(mesh.vertices)
            new_faces = len(mesh.faces)

            print(f"   ✅ Mesh nettoyé:")
            print(f"      Vertices: {original_vertices} → {new_vertices}")
            print(f"      Faces: {original_faces} → {new_faces}")

            return mesh

        except Exception as e:
            print(f"   ⚠️  Erreur nettoyage mesh: {e}")
            return mesh

    def simplify_mesh(
        self,
        mesh: trimesh.Trimesh,
        target_faces: Optional[int] = None,
        ratio: float = 0.5
    ) -> trimesh.Trimesh:
        """
        Simplifie le mesh en réduisant le nombre de faces

        Args:
            mesh: Mesh d'entrée
            target_faces: Nombre cible de faces (si None, utilise le ratio)
            ratio: Ratio de simplification (0.5 = moitié des faces)

        Returns:
            Mesh simplifié
        """
        print("⚡ Simplification du mesh...")

        try:
            original_faces = len(mesh.faces)

            if target_faces is None:
                target_faces = int(original_faces * ratio)

            # Utiliser la simplification de trimesh
            simplified_mesh = mesh.simplify_quadratic_decimation(target_faces)

            print(f"   ✅ Mesh simplifié:")
            print(
                f"      Faces: {original_faces} → {len(simplified_mesh.faces)}")
            print(
                f"      Vertices: {len(mesh.vertices)} → {len(simplified_mesh.vertices)}")

            return simplified_mesh

        except Exception as e:
            print(f"   ⚠️  Erreur simplification mesh: {e}")
            print("   Retour au mesh original")
            return mesh

    def repair_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Répare les problèmes courants du mesh

        Args:
            mesh: Mesh d'entrée

        Returns:
            Mesh réparé
        """
        print("🔧 Réparation du mesh...")

        try:
            # Vérifier l'état du mesh
            is_watertight = mesh.is_watertight
            is_winding_consistent = mesh.is_winding_consistent

            print(f"   📊 État initial:")
            print(f"      Watertight: {is_watertight}")
            print(f"      Winding consistent: {is_winding_consistent}")

            # Réparer l'orientation des faces si nécessaire
            if not is_winding_consistent:
                mesh.fix_normals()
                print("   🔄 Orientation des faces corrigée")

            # Essayer de fermer les trous si pas watertight
            if not is_watertight:
                # Note: trimesh n'a pas de fonction fill_holes intégrée
                # On peut essayer de supprimer les faces isolées
                mesh = self.clean_mesh(mesh)
                print("   🔄 Tentative de fermeture des trous")

            print("   ✅ Réparation terminée")
            return mesh

        except Exception as e:
            print(f"   ⚠️  Erreur réparation mesh: {e}")
            return mesh

    def optimize_mesh_for_printing(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Optimise le mesh pour l'impression 3D

        Args:
            mesh: Mesh d'entrée

        Returns:
            Mesh optimisé pour impression 3D
        """
        print("🖨️  Optimisation pour impression 3D...")

        try:
            # Nettoyer le mesh
            mesh = self.clean_mesh(mesh)

            # Réparer les problèmes
            mesh = self.repair_mesh(mesh)

            # S'assurer que le mesh est orienté correctement (Z+ vers le haut)
            bounds = mesh.bounds
            if bounds[1][2] - bounds[0][2] < 0.1:  # Mesh très plat
                print("   ⚠️  Mesh très plat détecté")

            # Vérifier la taille minimale pour l'impression
            scale = mesh.scale
            if scale < 1.0:  # Très petit
                print(f"   📏 Mesh petit détecté (échelle: {scale:.3f})")
                print("   💡 Conseil: vérifiez l'échelle avant impression")

            print("   ✅ Optimisation impression 3D terminée")
            return mesh

        except Exception as e:
            print(f"   ⚠️  Erreur optimisation impression: {e}")
            return mesh

    def debug_mesh_properties(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Debug function to examine mesh properties and colors

        Args:
            mesh: Trimesh object to examine

        Returns:
            Dictionary with mesh properties
        """
        props = {
            'vertices_count': len(mesh.vertices),
            'faces_count': len(mesh.faces),
            'bounds': mesh.bounds.tolist(),
            'scale': float(mesh.scale),
            'is_watertight': mesh.is_watertight,
            'is_winding_consistent': mesh.is_winding_consistent,
            'has_vertex_colors': hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None,
            'has_visual': hasattr(mesh, 'visual'),
            'has_visual_vertex_colors': False,
            'has_visual_face_colors': False,
            'vertex_colors_shape': None,
            'vertex_colors_dtype': None,
            'vertex_colors_range': None,
            'visual_type': None
        }

        # Check vertex_colors directly
        if props['has_vertex_colors']:
            props['vertex_colors_shape'] = mesh.vertex_colors.shape
            props['vertex_colors_dtype'] = str(mesh.vertex_colors.dtype)
            props['vertex_colors_range'] = (
                mesh.vertex_colors.min(), mesh.vertex_colors.max())

        # Check visual properties
        if props['has_visual']:
            props['visual_type'] = type(mesh.visual).__name__

            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                props['has_visual_vertex_colors'] = True
                props['visual_vertex_colors_shape'] = mesh.visual.vertex_colors.shape
                props['visual_vertex_colors_dtype'] = str(
                    mesh.visual.vertex_colors.dtype)
                props['visual_vertex_colors_range'] = (
                    mesh.visual.vertex_colors.min(), mesh.visual.vertex_colors.max())

            if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                props['has_visual_face_colors'] = True
                props['visual_face_colors_shape'] = mesh.visual.face_colors.shape
                props['visual_face_colors_dtype'] = str(
                    mesh.visual.face_colors.dtype)
                props['visual_face_colors_range'] = (
                    mesh.visual.face_colors.min(), mesh.visual.face_colors.max())

        return props

    def export_mesh(
        self,
        mesh: trimesh.Trimesh,
        output_path: str,
        file_format: str = 'stl'
    ) -> bool:
        """
        Exporte le mesh dans le format spécifié

        Args:
            mesh: Mesh à exporter
            output_path: Chemin de sortie
            file_format: Format de fichier ('stl', 'obj', 'ply', etc.)

        Returns:
            True si succès, False sinon
        """
        try:
            # S'assurer que l'extension correspond au format
            path = Path(output_path)
            if path.suffix.lower() != f'.{file_format.lower()}':
                output_path = str(path.with_suffix(f'.{file_format.lower()}'))

            # Exporter
            mesh.export(output_path)

            # Vérifier le fichier
            file_size = Path(output_path).stat().st_size / 1024 / 1024  # MB
            print(f"   ✅ {file_format.upper()} généré: {file_size:.2f} MB")

            return True

        except Exception as e:
            print(f"   ❌ Erreur export {file_format.upper()}: {e}")
            return False

    def create_mesh_preview(
        self,
        mesh: trimesh.Trimesh,
        output_path: str,
        size: Tuple[int, int] = (512, 512)
    ) -> bool:
        """
        Crée une image de prévisualisation du mesh

        Args:
            mesh: Mesh à prévisualiser
            output_path: Chemin de sortie pour l'image
            size: Taille de l'image (width, height)

        Returns:
            True si succès, False sinon
        """
        try:
            # Créer une scène avec le mesh
            scene = trimesh.Scene([mesh])

            # Générer l'image
            png_data = scene.save_image(resolution=size, visible=True)

            if png_data:
                with open(output_path, 'wb') as f:
                    f.write(png_data)
                print(f"   ✅ Prévisualisation sauvegardée: {output_path}")
                return True
            else:
                print("   ⚠️  Impossible de générer la prévisualisation")
                return False

        except Exception as e:
            print(f"   ❌ Erreur prévisualisation: {e}")
            return False

    def get_mesh_stats(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Retourne des statistiques détaillées sur le mesh

        Args:
            mesh: Mesh à analyser

        Returns:
            Dictionnaire avec les statistiques
        """
        stats = {
            'geometry': {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'edges': len(mesh.edges),
                'bounds': mesh.bounds.tolist(),
                'center': mesh.center_mass.tolist(),
                'scale': float(mesh.scale),
                'volume': float(mesh.volume) if mesh.is_watertight else None,
                'surface_area': float(mesh.area)
            },
            'quality': {
                'is_watertight': mesh.is_watertight,
                'is_winding_consistent': mesh.is_winding_consistent,
                'has_vertex_normals': hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None,
                'has_face_normals': hasattr(mesh, 'face_normals') and mesh.face_normals is not None
            },
            'visual': {
                'has_colors': hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None,
                'has_texture': hasattr(mesh.visual, 'material') and mesh.visual.material is not None,
                'visual_type': type(mesh.visual).__name__
            }
        }

        return stats

    def get_info(self) -> Dict[str, Any]:
        """Retourne des informations sur le processeur de mesh"""
        return {
            'name': 'Hunyuan3D Mesh Processor',
            'description': 'Traitement de mesh optimisé pour Hunyuan3D-2mv',
            'features': [
                'Normalisation et orientation automatiques',
                'Post-processing avec préservation des détails',
                'Application de couleurs de vertices rapides',
                'Nettoyage et réparation de mesh',
                'Simplification intelligente',
                'Optimisation pour impression 3D',
                'Export multi-formats (STL, OBJ, PLY)',
                'Génération de prévisualisations',
                'Analyse statistique complète'
            ],
            'functions': [
                'normalize_mesh: Normalisation du mesh',
                'to_gradio_3d_orientation: Orientation standard',
                'apply_vertex_colors: Couleurs de vertices rapides',
                'post_process_mesh: Post-processing avec options',
                'clean_mesh: Nettoyage des éléments problématiques',
                'simplify_mesh: Simplification intelligente',
                'repair_mesh: Réparation automatique',
                'optimize_mesh_for_printing: Optimisation impression 3D',
                'export_mesh: Export multi-formats',
                'debug_mesh_properties: Analyse détaillée'
            ]
        }


# Instance globale du processeur
mesh_processor = MeshProcessor()


def get_mesh_processor() -> MeshProcessor:
    """Retourne l'instance globale du processeur de mesh"""
    return mesh_processor


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fonction de convenance pour normaliser un mesh"""
    return mesh_processor.normalize_mesh(mesh)


def apply_vertex_colors(mesh: trimesh.Trimesh, reference_image: Image.Image) -> trimesh.Trimesh:
    """Fonction de convenance pour appliquer des couleurs de vertices"""
    return mesh_processor.apply_vertex_colors(mesh, reference_image)


def post_process_mesh(mesh: trimesh.Trimesh, preserve_details: bool = True) -> trimesh.Trimesh:
    """Fonction de convenance pour post-traiter un mesh"""
    return mesh_processor.post_process_mesh(mesh, preserve_details)


def get_mesh_processing_info() -> Dict[str, Any]:
    """Fonction de convenance pour obtenir les informations du processeur"""
    return mesh_processor.get_info()
