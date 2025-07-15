#!/usr/bin/env python3
"""
Script de conversion de pièces numismatiques en STL 3D avec Hunyuan3D-2
Optimisé spécifiquement pour les pièces avec paramètres anti-artefacts
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire lib au PYTHONPATH
lib_path = Path(__file__).parent / 'lib'
sys.path.insert(0, str(lib_path))


try:
    from lib.hunyuan3d_converter import Hunyuan3DConverter, get_hunyuan3d_info
except ImportError:
    print("❌ Erreur : Impossible d'importer les modules Hunyuan3D")
    print("💡 Vérifiez que le dossier 'lib' contient hunyuan3d_converter.py")
    print(f"💡 Chemin recherché : {lib_path}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="🪙 Convertisseur de Pièces vers STL avec Hunyuan3D-2 (Mode Optimisé)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "front_image", help="Chemin vers l'image de face (avers)")
    parser.add_argument("-b", "--back-image",
                        help="Chemin vers l'image de dos (revers)")
    parser.add_argument("-o", "--output", default="output_hunyuan3d",
                        help="Répertoire de sortie")
    parser.add_argument("--remove-background", action="store_true",
                        help="Supprimer l'arrière-plan des images")
    parser.add_argument("--no-video", action="store_true",
                        help="Pas de génération vidéo")
    parser.add_argument("--no-post-processing", action="store_true",
                        help="Désactiver le post-processing")
    parser.add_argument("--model-path", default="tencent/Hunyuan3D-2",
                        help="Chemin vers le modèle Hunyuan3D-2")
    parser.add_argument("--no-texture", action="store_true",
                        help="Générer uniquement la forme (pas de texture)")
    parser.add_argument("--vertex-colors", action="store_true",
                        help="🚀 Mode couleurs de vertices rapide (2-5s au lieu de 8+ min)")
    parser.add_argument("--info", action="store_true",
                        help="Afficher les informations sur Hunyuan3D-2")
    parser.add_argument("--quality",
                        choices=["debug", "low", "medium", "high", "ultra"],
                        default="high",
                        help="🎯 Niveau de qualité: debug=ultra-minimal (128px,5steps,flat), low=test rapide (256px,10steps), medium=équilibré (512px,50steps), high=optimal (1024px,100steps), ultra=maximum (1024px,150steps)")

    args = parser.parse_args()

    if args.info:
        info = get_hunyuan3d_info()
        print(f"🔧 {info['name']}")
        print(f"📋 {info['description']}")
        print(f"🔢 Version: {info['version']}")
        print("✨ Fonctionnalités:")
        for feature in info['features']:
            print(f"  • {feature}")
        print("📦 Utilitaires:")
        for util in info['utils']:
            print(f"  • {util}")
        return

    # Vérifier que les images existent
    if not Path(args.front_image).exists():
        print(f"❌ Image de face non trouvée: {args.front_image}")
        sys.exit(1)

    if args.back_image and not Path(args.back_image).exists():
        print(f"❌ Image de dos non trouvée: {args.back_image}")
        sys.exit(1)

    # Afficher l'en-tête
    print("🏛️  Convertisseur de Pièces vers STL avec Hunyuan3D-2")
    print("   Génération de modèles 3D haute fidélité")
    print("=" * 70)

    try:
        # Initialiser le convertisseur
        converter = Hunyuan3DConverter(
            model_path=args.model_path,
            texture_model_path=args.model_path,
            disable_texture=args.no_texture
        )

        # Appliquer le niveau de qualité demandé
        if args.quality == "debug":
            converter.enable_debug_mode()  # Ultra-minimal pour tests instantanés
        elif args.quality == "low":
            converter.enable_test_mode()  # Ultra-rapide pour tests
        elif args.quality == "medium":
            converter.enable_fast_mode()  # Équilibré
        elif args.quality == "high":
            # Mode haute qualité optimisé pour pièces (utilise les paramètres par défaut)
            pass
        elif args.quality == "ultra":
            converter.enable_ultra_mode()  # Qualité maximale

        # Afficher le mode de couleur choisi
        if args.vertex_colors:
            print("🚀 Mode VERTEX COLORS activé")
            print("   ⚡ Couleurs rapides basées sur l'image d'entrée (2-5s)")
            print("   💡 Alternative ultra-rapide à la texture complète")
        elif args.no_texture:
            print("🔘 Mode SANS COULEUR activé")
            print("   ⚡ Génération ultra-rapide sans texture ni couleur")
        else:
            print("🎨 Mode TEXTURE COMPLÈTE activé")
            print("   🕐 Génération de texture haute qualité (8+ minutes)")
            print("   💡 Utilisez --vertex-colors pour un mode plus rapide")

        # Vérifier l'environnement
        converter.check_environment()

        # Charger les modèles avec gestion d'erreur améliorée
        print("🤖 Chargement des modèles...")
        try:
            if not converter.load_models():
                print("❌ Erreur critique : Impossible de charger les modèles")
                print("💡 Vérifiez votre installation Hunyuan3D-2")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            print("💡 Continuons avec les modèles disponibles...")

        # Préparer les images
        images = [args.front_image]
        if args.back_image:
            images.append(args.back_image)

        # Convertir en STL
        stl_path = converter.convert_coin_to_stl(
            front_image=args.front_image,
            back_image=args.back_image,
            output_dir=args.output,
            remove_background=args.remove_background,
            render_video=not args.no_video,
            enable_post_processing=not args.no_post_processing,
            use_vertex_colors=args.vertex_colors
        )

        if stl_path:
            print(f"✅ Conversion terminée avec succès!")
            print(f"📁 Fichier STL: {stl_path}")
            print(f"📁 Dossier de sortie: {args.output}")

            # Afficher les fichiers générés
            output_dir = Path(args.output)
            if output_dir.exists():
                print("\n📋 Fichiers générés:")
                for file in output_dir.glob("*"):
                    print(f"  • {file.name}")
        else:
            print("❌ Erreur lors de la conversion")

    except KeyboardInterrupt:
        print("\n🛑 Conversion interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
