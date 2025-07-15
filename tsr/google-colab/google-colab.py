# Convertisseur PNG vers STL avec TripoSR
# Basé sur le tutoriel de PyImageSearch

# Si vous avez besoin de scikit-image, envisagez d'installer une version plus ancienne
!pip install scikit-image==0.19.3  # Cette version fonctionnait avec Pillow 9.5.0

# Étape 0: Réinstaller Pillow avec la version correcte
# !pip uninstall -y pillow
# !pip install --force-reinstall --no-deps pillow==9.5.0  # Version compatible avec TripoSR

# Installation des autres dépendances nécessaires
!pip install PyMCubes
!pip install -q torch torchvision
!pip install -q huggingface_hub
!pip install -q opencv-python
!pip install -q onnxruntime
!pip install -q rembg
!pip install -q pymeshlab
!pip install -q omegaconf
!pip install -q plyfile
!pip install -q tqdm

# Étape 1: Configurer l'environnement
# Cloner le dépôt TripoSR et se placer dans le bon répertoire
!git clone https://github.com/pyimagesearch/TripoSR.git
import sys
import os

# S'assurer que nous sommes dans le bon répertoire
!ls -la
if os.path.exists('TripoSR/TripoSR'):
    %cd TripoSR/TripoSR
else:
    %cd TripoSR

# Ajouter le chemin au PYTHONPATH
sys.path.append('.')
sys.path.append('./tsr')

# Installer les dépendances du projet
!pip install -r requirements.txt -q

# Étape 2: Importer les bibliothèques nécessaires
import torch
import time
from PIL import Image
import numpy as np
from IPython.display import Video, display, HTML
import rembg
from google.colab import files
import pymeshlab as pymesh

# Vérifier que les imports fonctionnent avant de continuer
print("Vérification des importations de base réussie")

# Étape 3: Importer les modules spécifiques à TripoSR
try:
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground, save_video
    print("Importations spécifiques à TripoSR réussies!")
except Exception as e:
    print(f"Erreur lors des importations spécifiques: {e}")
    print("Tentative de correction en installant les packages manquants...")
    !pip install -q mcubes trimesh
    !pip install -q diffusers transformers accelerate safetensors
    
    # Réessayer l'importation
    import importlib
    importlib.invalidate_caches()
    try:
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground, save_video
        print("Importations réussies après correction!")
    except Exception as e:
        print(f"Échec des importations après tentative de correction: {e}")
        print("Le script ne pourra pas continuer. Veuillez vérifier votre environnement.")
        raise e

# Étape 4: Configurer le périphérique (GPU ou CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du périphérique: {device}")

# Étape 5: Télécharger et préparer l'image
print("Veuillez télécharger votre image PNG:")
uploaded = files.upload()
original_image = Image.open(list(uploaded.keys())[0])

# Afficher l'image originale
display(original_image)

# Créer le dossier examples s'il n'existe pas
os.makedirs("examples", exist_ok=True)

# Redimensionner et sauvegarder l'image
original_image_resized = original_image.resize((512, 512))
original_image_resized.save("examples/product.png")
print("Image sauvegardée sous 'examples/product.png'")

# Étape 6: Configurer les paramètres de TripoSR
image_paths = "./examples/product.png"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pretrained_model_name_or_path = "stabilityai/TripoSR"
chunk_size = 8192
no_remove_bg = True  # Si True, ne supprime pas l'arrière-plan
foreground_ratio = 0.85
output_dir = "output/"
model_save_format = "obj"
render = True

# Créer le répertoire de sortie
output_dir = output_dir.strip()
os.makedirs(output_dir, exist_ok=True)

# Étape 7: Initialiser le modèle TripoSR
print("Chargement du modèle TripoSR...")
try:
    model = TSR.from_pretrained(
        pretrained_model_name_or_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(chunk_size)
    model.to(device)
    print("Modèle chargé avec succès!")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    print("Vérifiez votre connexion internet et réessayez.")
    raise e

# Étape 8: Traiter l'image pour la conversion
print("Traitement de l'image...")
images = []
try:
    rembg_session = rembg.new_session()

    # Supprimer l'arrière-plan si nécessaire
    if not no_remove_bg:
        image = remove_background(original_image_resized, rembg_session)
    else:
        image = original_image_resized

    # Redimensionner l'image
    image = resize_foreground(image, foreground_ratio)

    # Gérer les images RGBA
    if image.mode == "RGBA":
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))

    # Créer le répertoire de sortie pour cette image
    image_dir = os.path.join(output_dir, str(0))
    os.makedirs(image_dir, exist_ok=True)

    # Sauvegarder l'image traitée
    image.save(os.path.join(image_dir, "input.png"))
    images.append(image)

    # Afficher l'image traitée
    display(image)
    print("Image traitée et prête pour la génération 3D")
except Exception as e:
    print(f"Erreur lors du traitement de l'image: {e}")
    # En cas d'erreur, utiliser l'image originale redimensionnée
    image = original_image_resized
    image_dir = os.path.join(output_dir, str(0))
    os.makedirs(image_dir, exist_ok=True)
    image.save(os.path.join(image_dir, "input.png"))
    images.append(image)
    print("Utilisation de l'image originale sans traitement spécifique")

# Étape 9: Générer le modèle 3D
print("Génération du modèle 3D (cela peut prendre plusieurs minutes)...")
try:
    for i, image in enumerate(images):
        print(f"Traitement de l'image {i + 1}/{len(images)} ...")
        
        # Générer les codes de scène
        with torch.no_grad():
            scene_codes = model([image], device=device)
        
        # Rendre et sauvegarder les vues du modèle si demandé
        if render:
            print("Rendu des vues multiples...")
            render_images = model.render(scene_codes, n_views=30, return_type="pil")
            for ri, render_image in enumerate(render_images[0]):
                render_image.save(os.path.join(output_dir, str(i), f"render_{ri:03d}.png"))
            
            # Créer une vidéo du modèle
            save_video(
                render_images[0], os.path.join(output_dir, str(i), "render.mp4"), fps=30
            )
            print("Vidéo du modèle créée!")
        
        # Extraire et sauvegarder le maillage
        print("Extraction du maillage 3D...")
        meshes = model.extract_mesh(scene_codes, has_vertex_color=False)
        mesh_file = os.path.join(output_dir, str(i), f"mesh.{model_save_format}")
        meshes[0].export(mesh_file)
        print(f"Modèle 3D sauvegardé au format {model_save_format}!")

    # Étape 10: Convertir le modèle OBJ en STL
    print("Conversion du modèle OBJ en STL...")
    obj_file = os.path.join(output_dir, "0", f"mesh.{model_save_format}")
    ms = pymesh.MeshSet()
    ms.load_new_mesh(obj_file)

    # Sauvegarder en format STL
    stl_file = 'model.stl'
    ms.save_current_mesh(stl_file)
    print(f"Conversion terminée! Fichier STL sauvegardé sous '{stl_file}'")

    # Étape 11: Télécharger le fichier STL
    print("Téléchargement du fichier STL...")
    files.download(stl_file)

    # Étape 12: Afficher la vidéo du modèle 3D
    print("Affichage de la vidéo du modèle 3D:")
    #Video(os.path.join(output_dir, "0", "render.mp4"), embed=True)
    import base64
    video_path = os.path.join(output_dir, "0", "render.mp4")

    if os.path.exists(video_path):
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        # Encoder en base64
        b64_video = base64.b64encode(video_data).decode('utf-8')
        
        # Créer un HTML avec la vidéo encodée
        video_html = f"""
        <video width="640" height="480" controls autoplay>
            <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
        </video>
        """
        
        display(HTML(video_html))
    else:
        print(f"Erreur: Vidéo non trouvée à {video_path}")
    
except Exception as e:
    print(f"Erreur lors de la génération du modèle 3D: {e}")
    print("Conseils de débogage:")
    print("1. Vérifiez que toutes les dépendances sont correctement installées")
    print("2. Vérifiez que votre GPU a suffisamment de mémoire (si vous utilisez CUDA)")
    print("3. Essayez avec une image plus simple ou de plus petite taille")
    print("4. Vérifiez votre connexion internet pour le téléchargement du modèle TripoSR")