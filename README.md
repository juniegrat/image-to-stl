# Image to STL Converter

Convertisseur d'images en modèles STL 3D utilisant deux approches complémentaires :

- **TripoSR** : Conversion rapide et efficace
- **Hunyuan3D-2** : Qualité maximale avec architecture modulaire

## 🏗️ Architecture Modulaire Hunyuan3D-2

Le projet utilise désormais une **architecture modulaire** pour Hunyuan3D-2 avec des composants spécialisés :

### 📦 Modules Spécialisés (`lib/`)

- **`hunyuan3d_config.py`** - Configuration et modes de qualité (DEBUG, TEST, FAST, HIGH, ULTRA)
- **`hunyuan3d_models.py`** - Gestion des modèles et pipelines
- **`hunyuan3d_camera.py`** - Utilitaires de caméra et rayons
- **`hunyuan3d_rendering.py`** - Rendu 3D et génération vidéos
- **`hunyuan3d_mesh_processing.py`** - Traitement et optimisation mesh
- **`hunyuan3d_image_processing.py`** - Traitement d'images
- **`hunyuan3d_converter.py`** - Convertisseur principal modulaire
- **`hunyuan3d_utils.py`** - Compatibilité avec ancien code

### 🔧 Installation

```bash
# Installation avec versions validées
python install-hunyuan3d.py

# Ou directement depuis lib/
python lib/install-hunyuan3d.py
```

Le script d'installation utilise maintenant `requirements.txt` avec des **versions validées et testées** pour **Python 3.11.9** :

- PyTorch 2.7.1 avec CUDA 12.8
- Diffusers 0.34.0 (récente)
- Transformers 4.53.2 (récente)
- HuggingFace Hub 0.33.4 (récente)
- NumPy 2.3.1 (version 2.x)
- Et toutes les dépendances compatibles

## 🚀 Utilisation

### Conversion Simple

```bash
python hunyuan3d-coin-to-stl.py image.jpg
```

### Conversion Avers/Revers

```bash
python hunyuan3d-coin-to-stl.py avers.jpg -b revers.jpg
```

### Modes de Qualité

```bash
# Ultra qualité (recommandé pour pièces)
python hunyuan3d-coin-to-stl.py image.jpg --quality-preset ultra

# Mode rapide
python hunyuan3d-coin-to-stl.py image.jpg --quality-preset fast

# Mode test (développement)
python hunyuan3d-coin-to-stl.py image.jpg --quality-preset test
```

### Informations Système

```bash
python hunyuan3d-coin-to-stl.py --info
```

## 📁 Structure du Projet

```
image-to-stl/
├── lib/                          # Modules spécialisés
│   ├── hunyuan3d_*.py           # Architecture modulaire
│   └── install-hunyuan3d.py     # Installateur principal
├── install-hunyuan3d.py         # Wrapper d'installation
├── requirements.txt              # Versions validées (Python 3.11.9)
├── hunyuan3d-coin-to-stl.py     # Script principal
├── tsr/                          # TripoSR
├── Hunyuan3D-2/                  # Modèles Hunyuan3D
└── output_hunyuan3d/             # Résultats

```

## ✨ Avantages de l'Architecture Modulaire

- **Modularity** : Chaque module a une responsabilité unique
- **Maintainability** : Code organisé et plus facile à déboguer
- **Reusability** : Modules utilisables indépendamment
- **Extensibility** : Nouvelles fonctionnalités sans impact sur le reste
- **Backward Compatibility** : L'ancien code continue de fonctionner
- **Performance** : Optimisations ciblées par module

## 🔧 Versions Compatibles (Validées Python 3.11.9)

L'installateur utilise des **versions spécifiquement testées** avec **Python 3.11.9** :

| Package         | Version      | Statut                    |
| --------------- | ------------ | ------------------------- |
| PyTorch         | 2.7.1+cu128  | ✅ Récent avec CUDA 12.8  |
| Diffusers       | 0.34.0       | ✅ Version récente        |
| Transformers    | 4.53.2       | ✅ Version récente        |
| HuggingFace Hub | 0.33.4       | ✅ API récente            |
| xFormers        | 0.0.31.post1 | ✅ Compatible PyTorch 2.7 |
| NumPy           | 2.3.1        | ✅ Version 2.x récente    |
| Pillow          | 11.3.0       | ✅ Version récente        |
| Trimesh         | 4.0.5        | ✅ Version récente        |

## 🏃 Modes de Qualité

- **DEBUG** : Ultra-minimal, 256x256, 15 steps (test instantané)
- **TEST** : Ultra-rapide, 10 steps (développement)
- **FAST** : Compromis qualité/vitesse, 512x512, 50 steps
- **HIGH** : Optimisé pièces, 1024x1024, 100 steps
- **ULTRA** : Qualité maximale, 150 steps

## ⚠️ Configuration Recommandée

**Configuration validée et testée :**

- **Python 3.11.9** (version recommandée)
- **CUDA 12.8** (dernière version stable)
- **PyTorch 2.7.1+cu128** (compatible CUDA 12.8)
- **GPU NVIDIA** avec drivers récents

**Notes de compatibilité :**

- Redémarrer le terminal après installation
- Modèles téléchargés automatiquement au premier usage
- Architecture modulaire optimisée pour performances

## 📊 Performance

L'architecture modulaire avec les versions récentes permet :

- Conversion rapide en mode FAST (2-3 min)
- Qualité maximale en mode ULTRA (10-15 min)
- Debug instantané pour tests (<30 sec)
- Optimisations GPU/CPU automatiques avec PyTorch 2.7.1
- Support natif NumPy 2.x pour meilleures performances
