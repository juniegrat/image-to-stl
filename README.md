# Convertisseur d'Images vers STL avec TripoSR - Version Locale

Ce script permet de convertir des images (PNG, WebP, JPEG, BMP, TIFF) en modèles 3D STL en utilisant TripoSR, optimisé pour fonctionner localement avec une carte graphique NVIDIA (testé avec RTX 2070 Super).

## 📸 Formats d'images supportés

- **PNG** - Avec transparence (format optimal)
- **WebP** - Format moderne et léger
- **JPEG/JPG** - Format standard
- **BMP** - Format bitmap
- **TIFF** - Format haute qualité

_Le script détecte automatiquement le format et effectue la conversion nécessaire._

## 🆕 Nouveautés

### Support des vues multiples (recto + verso)

- **Reconstruction améliorée** : Utilisez une image de recto ET une image de verso pour obtenir un modèle 3D plus précis et détaillé
- **Qualité supérieure** : Les deux vues permettent au modèle de mieux comprendre la géométrie complète de l'objet
- **Facilité d'usage** : Simple ajout du paramètre `--reverse-image`

## Prérequis

- Windows 10/11
- Python 3.8 ou supérieur
- NVIDIA GPU avec CUDA support (RTX 2070 Super ou équivalent)
- CUDA Toolkit 11.8 ou supérieur
- Git
- Au moins 8 GB de RAM
- ~10 GB d'espace disque libre

## Installation

### 1. Installer CUDA Toolkit

Téléchargez et installez CUDA Toolkit depuis le site NVIDIA :
https://developer.nvidia.com/cuda-11-8-0-download-archive

### 2. Cloner ce dépôt

```bash
git clone [votre-repo]
cd png-to-stl
```

### 3. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
# Sur Windows PowerShell
.\venv\Scripts\Activate.ps1
# Ou sur Windows CMD
.\venv\Scripts\activate.bat
```

### 4. Installer les dépendances et configurer l'environnement

**Option 1 - Installation automatique (recommandée) :**

```bash
python install.py
```

**Option 2 - Installation manuelle :**

```bash
pip install -r requirements.txt
python png-to-stl-local.py --setup
```

L'installation automatique va :

- Installer toutes les dépendances Python nécessaires dans le bon ordre
- Éviter les problèmes de compilation
- Cloner le dépôt TripoSR
- Vérifier que tout fonctionne correctement

## Utilisation

### Reconstruction avec une seule vue (classique)

```bash
# Image PNG classique
python png-to-stl-local.py mon_image.png

# Image WebP (format moderne)
python png-to-stl-local.py photo.webp --remove-bg

# Image JPEG
python png-to-stl-local.py produit.jpg --remove-bg -o modeles/
```

### 🎯 Reconstruction avec deux vues (NOUVEAU)

```bash
# Reconstruction avec recto + verso pour un modèle plus précis
python png-to-stl-local.py recto.png --reverse-image verso.png

# Formats mixtes (PNG + WebP)
python png-to-stl-local.py piece_face.png --reverse-image piece_pile.webp

# Images JPEG avec suppression d'arrière-plan
python png-to-stl-local.py avant.jpg --reverse-image arriere.jpg --remove-bg

# Exemple complet avec formats différents
python png-to-stl-local.py recto.webp --reverse-image verso.png --remove-bg -o modeles_2vues/
```

### Options disponibles

```bash
# Spécifier un dossier de sortie
python png-to-stl-local.py mon_image.png -o mon_dossier_sortie

# Supprimer l'arrière-plan de l'image
python png-to-stl-local.py mon_image.png --remove-bg

# Ne pas générer de vidéo de prévisualisation
python png-to-stl-local.py mon_image.png --no-video

# Combiner les options
python png-to-stl-local.py mon_image.png -o resultats --remove-bg --no-video
```

## Structure des fichiers de sortie

```
output/
├── 0/
│   ├── input.png           # Image principale traitée
│   ├── input_reverse.png   # Image de revers traitée (si fournie)
│   ├── mesh.obj           # Modèle 3D au format OBJ
│   ├── render.mp4         # Vidéo de rotation du modèle
│   └── render_XXX.png     # Images de rendu (30 vues)
└── votre_image.stl        # Fichier STL final
```

## Conseils d'utilisation

1. **Images recommandées** :

   - **PNG** : Idéal avec fond transparent
   - **WebP** : Excellent compromis qualité/taille
   - **JPEG** : Bon pour photos (utiliser --remove-bg)
   - Résolution entre 512x512 et 2048x2048
   - Objet bien centré et éclairé
   - Éviter les ombres portées

2. **Performance** :

   - La première exécution télécharge le modèle TripoSR (~2GB)
   - La génération prend généralement 1-3 minutes avec une RTX 2070 Super
   - La suppression d'arrière-plan ajoute ~30 secondes

3. **Mémoire GPU** :
   - Le processus utilise environ 4-6 GB de VRAM
   - Fermez les autres applications GPU-intensives

## Dépannage

### "CUDA non disponible"

- Vérifiez l'installation de CUDA Toolkit
- Réinstallez PyTorch avec support CUDA :
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

### Fix torchmcubes

- `pip install git+https://github.com/tatsy/torchmcubes.git@3aef8afa5f21b113afc4f4ea148baee850cbd472`

  ### "Out of memory"

- Fermez les autres applications utilisant le GPU
- Redémarrez votre ordinateur
- Utilisez une image plus petite

### "Module not found"

- Réexécutez `python install.py` ou `python png-to-stl-local.py --setup`
- Vérifiez que l'environnement virtuel est activé

### Erreurs de compilation (scikit-image, etc.)

- Utilisez le script d'installation automatique : `python install.py`
- Ou installez une version précompilée : `pip install scikit-image>=0.20.0`
- Assurez-vous d'avoir Visual Studio Build Tools installé

## Exemples de commandes

```bash
# Image PNG simple avec fond blanc
python png-to-stl-local.py logo.png

# Image WebP moderne
python png-to-stl-local.py image.webp --remove-bg

# Photo JPEG d'objet avec suppression du fond
python png-to-stl-local.py produit.jpg --remove-bg -o modeles_3d

# Traitement rapide sans vidéo
python png-to-stl-local.py sketch.png --no-video

# Pièce de monnaie avec deux faces en formats différents
python png-to-stl-local.py face.png --reverse-image pile.webp --remove-bg

# Batch processing (créer un script batch pour PNG)
for %f in (*.png) do python png-to-stl-local.py "%f" -o stl_files

# Batch processing pour WebP
for %f in (*.webp) do python png-to-stl-local.py "%f" --remove-bg -o stl_webp
```

## Limitations

- Fonctionne mieux avec des objets simples et bien définis
- Les détails très fins peuvent ne pas être bien capturés
- Les textures et couleurs ne sont pas préservées dans le STL
- La qualité dépend fortement de l'image d'entrée

## Crédits

Ce script est basé sur :

- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) par Stability AI
- Tutoriel de [PyImageSearch](https://pyimagesearch.com/)
- Adapté pour utilisation locale par [votre nom]

- [TripoSR issue 74](https://github.com/VAST-AI-Research/TripoSR/issues/74)
- [PyImageSearch](https://pyimagesearch.com/2024/12/11/png-image-to-stl-converter-in-python)
- [Colab](https://colab.research.google.com/drive/1S4-Xn3tW6_nLd0cWGHzbP1QBZuke6JRR)
