def print_render_tips():
    """
    Affiche des conseils pour améliorer la qualité des rendus
    """
    print("\n🎬 Conseils pour améliorer la qualité des rendus:")
    print("\n📐 Résolution:")
    print("   • 256x256: Rapide mais qualité basique")
    print("   • 512x512: Bon compromis (défaut)")
    print("   • 1024x1024: Haute qualité mais plus lent")
    print("   • 2048x2048: Très haute qualité (GPU puissant requis)")

    print("\n📷 Paramètres de caméra:")
    print("   • Distance 1.5-1.9: Vue rapprochée (détails)")
    print("   • Distance 2.0-2.5: Vue éloignée (contexte)")
    print("   • Élévation 0°: Vue horizontale")
    print("   • Élévation 15-30°: Vue en plongée (recommandé)")
    print("   • FOV 30-35°: Vue serrée (zoom)")
    print("   • FOV 40-50°: Vue large (contexte)")

    print("\n🎞️  Nombre de vues:")
    print("   • 15-20 vues: Rotation basique")
    print("   • 30 vues: Standard (défaut)")
    print("   • 60 vues: Rotation très fluide")
    print("   • 120 vues: Rotation ultra-fluide (très lent)")

    print("\n🔧 Exemples de commandes:")
    print("   # Haute qualité")
    print("   python tsr/image-to-stl.py image.png --render-resolution 1024 --render-elevation 20")
    print("   # Vue rapprochée")
    print("   python tsr/image-to-stl.py image.png --render-distance 1.6 --render-fov 35")
    print("   # Rotation ultra-fluide")
    print("   python tsr/image-to-stl.py image.png --render-views 60")


def print_coin_tips():
    """Affiche des conseils pour optimiser la conversion de pièces numismatiques"""
    print("\n💡 CONSEILS POUR PIÈCES NUMISMATIQUES:")
    print("=" * 50)
    print("🪙 QUALITÉ OPTIMALE:")
    print("   • Utilisez des images haute résolution (minimum 1000x1000 pixels)")
    print("   • Éclairage uniforme sans ombres portées")
    print("   • Fond contrasté (blanc ou noir uni)")
    print("   • Pièce bien centrée dans l'image")
    print("   • Ajoutez une image de revers avec --reverse-image pour de meilleurs résultats")
    print("")
    print("⚙️  PARAMÈTRES RECOMMANDÉS:")
    print("   • Résolution standard: --mc-resolution 640 (défaut optimisé)")
    print("   • Haute qualité: --mc-resolution 800 ou 1024")
    print("   • Si artefacts: ajuster --mc-threshold (0.1-0.2)")
    print("   • Images très détaillées: --foreground-ratio 0.7")
    print("")
    print("🚀 EXEMPLES DE COMMANDES:")
    print("   • Standard: python tsr/image-to-stl.py ma_piece.png")
    print("   • Avec revers: python tsr/image-to-stl.py recto.png --reverse-image verso.png")
    print("   • Très haute qualité: python tsr/image-to-stl.py piece.png --mc-resolution 1024")
    print("   • Supprimer fond: python tsr/image-to-stl.py piece.jpg --remove-bg")
    print("")
