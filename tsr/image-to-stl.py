#!/usr/bin/env python3
"""
Convertisseur PNG vers STL avec TripoSR - Version Locale
Adapté pour fonctionner sur Windows avec GPU NVIDIA
Version modulaire organisée
"""

import argparse
import sys
from pathlib import Path

# Imports des modules locaux
from lib.setup import setup_triposr, check_cuda_compatibility
from lib.converter import convert_png_to_stl, convert_coin_to_stl_safe
from lib.diagnostics import diagnostic_info, analyze_render_quality, print_system_report
from lib.utils import get_render_params, print_render_info, validate_file_path, ensure_output_directory
from lib.image_processor import get_supported_formats, analyze_image_quality
from lib.print_tips import print_render_tips, print_coin_tips


# Toutes les fonctions ont été déplacées vers les modules lib/


# Fonctions déplacées vers les modules lib/


def main():
    parser = argparse.ArgumentParser(
        description="Convertir une image (PNG, WebP, JPEG, etc.) en modèle STL")
    parser.add_argument(
        "input", help="Chemin vers l'image d'entrée (PNG, WebP, JPEG, BMP, TIFF)")
    parser.add_argument("-o", "--output", default="output",
                        help="Répertoire de sortie (défaut: output)")
    parser.add_argument("--remove-bg", action="store_true",
                        help="Supprimer l'arrière-plan de l'image (défaut: True)")
    parser.add_argument("--no-remove-bg", action="store_true",
                        help="NE PAS supprimer l'arrière-plan de l'image")
    parser.add_argument("--no-video", action="store_true",
                        help="Ne pas générer de vidéo du modèle")
    parser.add_argument("--reverse-image",
                        help="Chemin vers l'image de revers (verso) pour améliorer la reconstruction 3D")
    parser.add_argument("--mc-resolution", type=int, default=256,
                        help="Résolution du marching cubes (défaut: 256 - paramètre officiel TripoSR)")
    parser.add_argument("--mc-threshold", type=float, default=25.0,
                        help="Seuil du marching cubes (défaut: 25.0 - paramètre officiel TripoSR)")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Désactiver le lissage du maillage")
    parser.add_argument("--foreground-ratio", type=float, default=0.85,
                        help="Ratio de l'objet dans l'image (défaut: 0.85 - paramètre officiel TripoSR)")
    parser.add_argument("--debug", action="store_true",
                        help="Afficher les informations de diagnostic (filtres disponibles, etc.)")
    parser.add_argument("--setup", action="store_true",
                        help="Installer les dépendances et configurer l'environnement")
    parser.add_argument("--tips", action="store_true",
                        help="Afficher les conseils d'optimisation pour pièces numismatiques")
    parser.add_argument("--render-resolution", type=int, default=512,
                        help="Résolution des images de rendu (défaut: 512)")
    parser.add_argument("--render-elevation", type=float, default=0.0,
                        help="Angle d'élévation de la caméra en degrés (défaut: 0.0)")
    parser.add_argument("--render-distance", type=float, default=1.9,
                        help="Distance de la caméra (défaut: 1.9)")
    parser.add_argument("--render-fov", type=float, default=40.0,
                        help="Champ de vision de la caméra en degrés (défaut: 40.0)")
    parser.add_argument("--render-views", type=int, default=30,
                        help="Nombre de vues pour la vidéo de rotation (défaut: 30)")
    parser.add_argument("--analyze-render",
                        help="Analyser la qualité des rendus dans le dossier spécifié")
    parser.add_argument("--render-tips", action="store_true",
                        help="Afficher les conseils pour améliorer la qualité des rendus")

    args = parser.parse_args()

    print("🚀 Convertisseur d'Images vers STL avec TripoSR")
    print("   Formats supportés: PNG, WebP, JPEG, BMP, TIFF")
    print("=" * 50)

    if args.setup:
        print("⚙️  Configuration de l'environnement...")
        from lib.setup import check_and_install_dependencies
        check_and_install_dependencies()
        setup_triposr()
        print("\n✅ Configuration terminée! Vous pouvez maintenant utiliser le script.")
        print(f"💡 Exemple PNG: python {sys.argv[0]} mon_image.png")
        print(f"💡 Exemple WebP: python {sys.argv[0]} mon_image.webp")
        print(
            f"💡 Avec revers: python {sys.argv[0]} recto.png --reverse-image verso.webp")
        print(
            f"💡 Haute qualité: python {sys.argv[0]} image.png --mc-resolution 1024")
        print(f"💡 Diagnostic: python {sys.argv[0]} image.png --debug")
        print(f"💡 Conseils: python {sys.argv[0]} --tips")
        return

    if args.debug:
        diagnostic_info()
        return

    if args.tips:
        print_coin_tips()
        return

    if args.render_tips:
        print_render_tips()
        return

    if args.analyze_render:
        analyze_render_quality(args.analyze_render)
        return

    # Afficher les informations de rendu si mode debug ou si paramètres non-standard
    if args.debug or args.render_resolution != 512 or args.render_views != 30:
        print_render_info(args)

    # Vérifier l'image d'entrée
    supported_formats = get_supported_formats()
    is_valid, error_msg = validate_file_path(args.input, supported_formats)
    if not is_valid:
        print(f"❌ {error_msg}")
        return

    # Vérifier l'image de revers si spécifiée
    if args.reverse_image:
        is_valid, error_msg = validate_file_path(
            args.reverse_image, supported_formats)
        if not is_valid:
            print(f"❌ Image de revers: {error_msg}")
            return

    # Initialiser TripoSR (ajout du chemin) et vérifier CUDA
    setup_triposr()
    check_cuda_compatibility()

    # S'assurer que le répertoire de sortie existe
    ensure_output_directory(args.output)

    # Obtenir les paramètres de rendu
    render_params = get_render_params(args)

    # Gestion de la suppression d'arrière-plan (pour correspondre au comportement par défaut de run.py)
    if args.remove_bg and args.no_remove_bg:
        print("❌ Erreur: --remove-bg et --no-remove-bg sont incompatibles")
        return

    if args.no_remove_bg:
        remove_bg = False
        print("🖼️  Mode: Conservation de l'arrière-plan")
    elif args.remove_bg:
        remove_bg = True
        print("🖼️  Mode: Suppression de l'arrière-plan")
    else:
        # Par défaut, comme run.py, on supprime l'arrière-plan
        remove_bg = True
        print("🖼️  Mode: Suppression de l'arrière-plan (défaut comme run.py)")

    # Conversion unique via le script officiel TripoSR/run.py
    stl_file = convert_png_to_stl(
        args.input,
        args.output,
        remove_bg=remove_bg,
        render_video=not args.no_video,
        reverse_image_path=args.reverse_image,
        render_params=render_params,
    )

    print("\n" + "=" * 50)
    if stl_file:
        print(f"🎉 Conversion terminée avec succès!")
        print(f"📁 Fichier STL: {stl_file}")
        print(f"📁 Dossier de sortie: {args.output}")
        if not args.no_video:
            print(f"🎬 Vidéo disponible dans: {args.output}/0/render.mp4")
        if args.reverse_image:
            print(f"🔄 Images utilisées: {args.input} + {args.reverse_image}")
        print(
            f"⚙️  Paramètres utilisés: résolution={args.mc_resolution}, seuil={args.mc_threshold}")
    else:
        print("❌ Échec de la conversion.")
        print("💡 Vérifiez les messages d'erreur ci-dessus pour plus d'informations.")
        print("💡 Essayez d'ajuster --mc-resolution ou --mc-threshold")
        print("💡 Utilisez --debug pour plus d'informations de diagnostic")


# Toutes les fonctions ont été déplacées vers les modules lib/


if __name__ == "__main__":
    main()
