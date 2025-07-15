# Image to STL Converter

Convertit des images en modèles STL 3D imprimables, avec support pour TripoSR et Hunyuan3D-2.

## 🚀 Fonctionnalités

- **Conversion d'images en STL** : PNG, WebP, JPEG, BMP, TIFF
- **Support multi-modèles** : TripoSR et Hunyuan3D-2mv
- **Génération vidéo 360°** : Rotation automatique du modèle
- **Post-processing avancé** : Optimisation pour impression 3D
- **Compatibilité TripoSR** : Utilise les mêmes utilitaires de rendu
- **Suppression d'arrière-plan** : Avec rembg intégré
- **Mode multi-view** : Support avers/revers pour pièces

## 📦 Installation

### 1. Cloner le projet

```bash
git clone https://github.com/votre-repo/image-to-stl.git
cd image-to-stl
```

### 2. Installer les dépendances

#### Pour TripoSR (recommandé)

```bash
pip install -r requirements.txt
python image-to-stl.py --setup
```

#### Pour Hunyuan3D-2mv (haute fidélité)

```bash
pip install -r requirements-hunyuan3d.txt
python hunyuan3d-coin-to-stl.py --setup
```

## 🔧 Utilisation

### TripoSR (Standard)

```bash
# Conversion basique
python image-to-stl.py image.png

# Avec paramètres avancés
python image-to-stl.py image.png --mc-resolution 512 --render-views 60

# Mode pièce avec avers/revers
python image-to-stl.py avers.png --reverse-image revers.png
```

### Hunyuan3D-2mv (Haute fidélité avec compatibilité TripoSR)

```bash
# Conversion basique
python hunyuan3d-coin-to-stl.py avers.png

# Avec paramètres TripoSR compatibles
python hunyuan3d-coin-to-stl.py avers.png --n-views 60 --height 1024 --width 1024

# Mode multi-view avec suppression d'arrière-plan TripoSR
python hunyuan3d-coin-to-stl.py avers.png -b revers.png --remove-bg

# Paramètres de caméra avancés (compatibles TripoSR)
python hunyuan3d-coin-to-stl.py piece.png --camera-distance 2.5 --elevation-deg 15 --fovy-deg 50
```

## 🎬 Nouveaux paramètres de rendu (compatibles TripoSR)

### Paramètres de caméra

- `--n-views` : Nombre de vues pour la vidéo (défaut: 30)
- `--height` / `--width` : Résolution de rendu (défaut: 512x512)
- `--elevation-deg` : Angle d'élévation en degrés (défaut: 0.0)
- `--camera-distance` : Distance de la caméra (défaut: 1.9)
- `--fovy-deg` : Champ de vision vertical (défaut: 40.0)

### Traitement d'image

- `--foreground-ratio` : Ratio de l'objet dans l'image (défaut: 0.85)
- `--remove-bg` : Suppression d'arrière-plan avec rembg (TripoSR)

### Exemples d'utilisation avancée

```bash
# Rendu haute qualité
python hunyuan3d-coin-to-stl.py piece.png --n-views 120 --height 1024 --width 1024 --fps 60

# Vue rapprochée avec angle
python hunyuan3d-coin-to-stl.py piece.png --camera-distance 1.5 --elevation-deg 20

# Champ de vision large
python hunyuan3d-coin-to-stl.py piece.png --fovy-deg 60 --remove-bg
```

## 🔄 Compatibilité TripoSR

Les scripts Hunyuan3D utilisent maintenant les mêmes utilitaires que TripoSR :

### Fonctionnalités partagées

- **Suppression d'arrière-plan** : `rembg` avec session réutilisable
- **Redimensionnement** : `resize_foreground` avec ratio configurable
- **Génération vidéo** : `save_video` avec imageio
- **Paramètres de caméra** : Mêmes conventions que TripoSR

### Avantages

- **Cohérence** : Même comportement entre les deux systèmes
- **Performance** : Réutilisation des optimisations TripoSR
- **Maintenance** : Code unifié pour le rendu

## 📊 Comparaison des modèles

| Modèle        | Qualité    | Vitesse    | Multi-view | Texture    | Compatibilité TripoSR |
| ------------- | ---------- | ---------- | ---------- | ---------- | --------------------- |
| TripoSR       | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ❌         | ⭐⭐⭐     | ✅ Natif              |
| Hunyuan3D-2mv | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | ✅         | ⭐⭐⭐⭐⭐ | ✅ Compatible         |

## 🛠️ Diagnostic et dépannage

### Vérifier l'environnement

```bash
# TripoSR
python image-to-stl.py --debug

# Hunyuan3D
python hunyuan3d-coin-to-stl.py --setup
```

### Problèmes courants

- **Mémoire GPU insuffisante** : Réduisez `--height` et `--width`
- **Rendu lent** : Diminuez `--n-views`
- **Qualité insuffisante** : Augmentez la résolution et utilisez `--remove-bg`

## 📁 Structure du projet

```
image-to-stl/
├── lib/
│   ├── converter.py              # Convertisseur TripoSR
│   ├── hunyuan3d_converter.py    # Convertisseur Hunyuan3D (compatible TripoSR)
│   ├── image_processor.py        # Traitement d'images
│   └── utils.py                  # Utilitaires partagés
├── TripoSR/                      # Sous-module TripoSR
├── image-to-stl.py              # Script principal TripoSR
├── hunyuan3d-coin-to-stl.py     # Script Hunyuan3D compatible TripoSR
└── requirements*.txt             # Dépendances
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez :

1. Maintenir la compatibilité TripoSR
2. Tester avec les deux modèles
3. Documenter les nouveaux paramètres
4. Respecter les conventions de nommage TripoSR

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
