#!/usr/bin/env python3
"""
Gestion des modèles et pipelines pour Hunyuan3D-2mv
Gère le chargement des modèles, vérification d'environnement et pipelines
"""

import torch
import warnings
from typing import Optional, Dict, Any, List
from pathlib import Path


class ModelManager:
    """
    Gestionnaire de modèles pour Hunyuan3D-2mv
    Gère le chargement et la configuration des pipelines
    """

    def __init__(
        self,
        model_path: str = "tencent/Hunyuan3D-2",
        texture_model_path: str = "tencent/Hunyuan3D-2",
        device: Optional[str] = None,
        disable_texture: bool = False
    ):
        """
        Initialise le gestionnaire de modèles

        Args:
            model_path: Chemin vers le modèle de forme
            texture_model_path: Chemin vers le modèle de texture
            device: Device à utiliser (auto-détecté si None)
            disable_texture: Désactiver complètement le chargement du modèle de texture
        """
        self.model_path = model_path
        self.texture_model_path = texture_model_path
        self.device = device or self._detect_device()
        self.disable_texture = disable_texture

        # Pipelines (chargés à la demande)
        self.shape_pipeline = None
        self.texture_pipeline = None
        self._models_loaded = False

        # Supprimer les warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        print(f"🤖 Gestionnaire de modèles initialisé")
        print(f"   Device: {self.device}")
        print(f"   Modèle forme: {self.model_path}")
        print(f"   Modèle texture: {self.texture_model_path}")

    def _detect_device(self) -> str:
        """Détecte automatiquement le meilleur device disponible"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def check_environment(self) -> bool:
        """
        Vérifie l'environnement et les dépendances

        Returns:
            True si l'environnement est OK, False sinon
        """
        print("🔍 Vérification de l'environnement...")

        # Vérifier CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(
                0).total_memory / 1024**3
            print(f"✅ CUDA: {gpu_name}")
            print(f"   Mémoire GPU: {gpu_memory:.1f} GB")
        else:
            print("⚠️  CUDA non disponible, utilisation CPU")

        # Vérifier les dépendances critiques
        deps = [
            'diffusers', 'transformers', 'accelerate', 'torch',
            'torchvision', 'PIL', 'numpy', 'trimesh', 'imageio',
            'rembg', 'tqdm', 'matplotlib'
        ]
        missing = []

        for dep in deps:
            try:
                if dep == 'PIL':
                    import PIL
                elif dep == 'matplotlib':
                    import matplotlib.pyplot as plt
                else:
                    __import__(dep)
            except ImportError:
                missing.append(dep)

        if missing:
            print(f"❌ Dépendances manquantes: {', '.join(missing)}")
            return False

        print("✅ Toutes les dépendances sont disponibles")

        # Vérifier Hunyuan3D
        try:
            import hy3dgen
            print("✅ Module hy3dgen détecté")
            return True
        except ImportError:
            print("❌ Module hy3dgen non trouvé")
            print("💡 Exécutez: python install-hunyuan3d.py")
            return False

    def load_models(self) -> bool:
        """
        Charge les modèles Hunyuan3D-2

        Returns:
            True si succès, False sinon
        """
        if self._models_loaded:
            print("✅ Modèles déjà chargés")
            return True

        print("🤖 Chargement des modèles...")

        try:
            # Importer les classes nécessaires
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            # Charger le modèle de forme
            success = self._load_shape_model(Hunyuan3DDiTFlowMatchingPipeline)
            if not success:
                return False

            # Charger le modèle de texture (optionnel)
            if not self.disable_texture:
                self._load_texture_model(Hunyuan3DPaintPipeline)
            else:
                print("   🚫 Chargement du modèle de texture désactivé")
                self.texture_pipeline = None

            self._models_loaded = True
            print("✅ Modèles chargés avec succès!")
            return True

        except ImportError as e:
            print(f"❌ Erreur lors de l'importation des modules Hunyuan3D: {e}")
            print("💡 Vérifiez que Hunyuan3D-2 est correctement installé")
            print("💡 Exécutez: python install-hunyuan3d.py")
            return False
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles: {e}")
            return False

    def _load_shape_model(self, pipeline_class) -> bool:
        """Charge le modèle de forme"""
        print(f"   📐 Chargement du modèle de forme...")
        try:
            self.shape_pipeline = pipeline_class.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                cache_dir=None  # Utilise le cache HF standard
            )
            print(f"   ✅ Modèle de forme chargé avec succès!")
            return True
        except Exception as e:
            print(f"   ❌ Erreur chargement modèle de forme: {e}")
            return False

    def _load_texture_model(self, pipeline_class) -> bool:
        """Charge le modèle de texture"""
        print("   🎨 Chargement du modèle de texture...")
        try:
            # Essayer plusieurs chemins possibles pour le modèle de texture
            texture_paths = [
                self.texture_model_path,
                "tencent/Hunyuan3D-2",  # Utilise le cache HF standard
            ]

            texture_loaded = False
            for path in texture_paths:
                try:
                    print(f"      Tentative avec: {path}")
                    self.texture_pipeline = pipeline_class.from_pretrained(
                        path)

                    # Déplacer vers le device manuellement si nécessaire
                    if hasattr(self.texture_pipeline, 'to'):
                        self.texture_pipeline = self.texture_pipeline.to(
                            self.device)

                    print("   ✅ Modèle de texture chargé avec succès!")
                    texture_loaded = True
                    break
                except Exception as e:
                    print(f"      ⚠️  Échec avec {path}: {e}")
                    continue

            if not texture_loaded:
                print("   ⚠️  Impossible de charger le modèle de texture")
                print("   📋 Mode disponible: génération de forme uniquement")
                self.texture_pipeline = None
                return False

            return True

        except Exception as e:
            print(f"   ⚠️  Erreur générale texture: {e}")
            print("   Continuation sans texture (mesh uniquement)")
            self.texture_pipeline = None
            return False

    def generate_3d_mesh(
        self,
        images: List,
        config: Dict[str, Any],
        progress_callback=None
    ):
        """
        Génère un mesh 3D à partir des images

        Args:
            images: Liste des images préparées
            config: Configuration des paramètres
            progress_callback: Callback de progression (optionnel)

        Returns:
            Mesh 3D généré ou None en cas d'erreur
        """
        if not self.shape_pipeline:
            print("❌ Modèle de forme non chargé")
            return None

        try:
            # Préparer le générateur
            generator = torch.Generator(
                device=self.device).manual_seed(config['seed'])

            # Générer selon le nombre d'images
            if len(images) > 1:
                # Mode multi-view
                print(f"   🔄 Mode multi-view avec {len(images)} images")
                mesh = self.shape_pipeline(
                    image=images,
                    guidance_scale=config['guidance_scale'],
                    num_inference_steps=config['num_inference_steps'],
                    octree_resolution=config['octree_resolution'],
                    num_chunks=config['num_chunks'],
                    generator=generator,
                    callback=progress_callback,
                    callback_steps=1 if progress_callback else None,
                    output_type='trimesh'
                )[0]
            else:
                # Mode single view
                print("   🔄 Mode single view")
                mesh = self.shape_pipeline(
                    image=images[0],
                    guidance_scale=config['guidance_scale'],
                    num_inference_steps=config['num_inference_steps'],
                    octree_resolution=config['octree_resolution'],
                    num_chunks=config['num_chunks'],
                    generator=generator,
                    callback=progress_callback,
                    callback_steps=1 if progress_callback else None,
                    output_type='trimesh'
                )[0]

            # Statistiques du mesh
            print(
                f"   ✅ Mesh généré: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh

        except Exception as e:
            print(f"❌ Erreur génération mesh: {e}")
            return None

    def apply_texture(
        self,
        mesh,
        reference_image,
        config: Dict[str, Any]
    ):
        """
        Applique une texture au mesh

        Args:
            mesh: Mesh 3D de base
            reference_image: Image de référence pour la texture
            config: Configuration des paramètres

        Returns:
            Mesh texturé
        """
        if not self.texture_pipeline:
            print("⚠️  Modèle de texture non chargé, conservation du mesh sans texture")
            return mesh

        try:
            # Préparer les paramètres de texture
            texture_steps = config.get('texture_steps', 40)
            guidance_scale = config.get('texture_guidance_scale', 2.0)

            print(
                f"   🔄 Application de texture ({texture_steps} steps, guidance={guidance_scale})...")

            # Appeler le pipeline de texture
            textured_mesh = self.texture_pipeline(
                mesh,
                image=reference_image,
                guidance_scale=guidance_scale,
                num_inference_steps=texture_steps
            )

            print("   ✅ Texture appliquée avec succès")
            return textured_mesh

        except Exception as e:
            print(f"⚠️  Erreur application texture: {e}")
            print("   Retour au mesh sans texture")
            return mesh

    def unload_models(self):
        """Décharge les modèles pour libérer la mémoire"""
        print("🗑️  Déchargement des modèles...")

        if self.shape_pipeline:
            del self.shape_pipeline
            self.shape_pipeline = None

        if self.texture_pipeline:
            del self.texture_pipeline
            self.texture_pipeline = None

        # Nettoyer le cache GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._models_loaded = False
        print("✅ Modèles déchargés")

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne des informations sur les modèles chargés"""
        return {
            'model_path': self.model_path,
            'texture_model_path': self.texture_model_path,
            'device': self.device,
            'disable_texture': self.disable_texture,
            'models_loaded': self._models_loaded,
            'shape_pipeline_loaded': self.shape_pipeline is not None,
            'texture_pipeline_loaded': self.texture_pipeline is not None,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

    def get_memory_usage(self) -> Dict[str, Any]:
        """Retourne l'utilisation mémoire actuelle"""
        memory_info = {
            'device': self.device,
            'models_loaded': self._models_loaded
        }

        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,   # GB
                # GB
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3
            })

        return memory_info

    def optimize_memory(self):
        """Optimise l'utilisation mémoire"""
        print("🧹 Optimisation mémoire...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   ✅ Cache GPU nettoyé")

        # Garbage collection Python
        import gc
        gc.collect()
        print("   ✅ Garbage collection effectué")

    def validate_installation(self) -> Dict[str, bool]:
        """Valide l'installation complète de Hunyuan3D"""
        validation = {
            'environment_ok': False,
            'hy3dgen_available': False,
            'models_loadable': False,
            'cuda_available': torch.cuda.is_available()
        }

        # Vérifier l'environnement
        validation['environment_ok'] = self.check_environment()

        # Vérifier hy3dgen
        try:
            import hy3dgen
            validation['hy3dgen_available'] = True
        except ImportError:
            validation['hy3dgen_available'] = False

        # Tester le chargement des modèles (sans vraiment charger)
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            validation['models_loadable'] = True
        except ImportError:
            validation['models_loadable'] = False

        return validation

    def get_info(self) -> Dict[str, Any]:
        """Retourne des informations complètes sur le gestionnaire"""
        return {
            'name': 'Hunyuan3D Model Manager',
            'description': 'Gestionnaire de modèles et pipelines pour Hunyuan3D-2mv',
            'model_info': self.get_model_info(),
            'memory_info': self.get_memory_usage(),
            'validation': self.validate_installation(),
            'features': [
                'Chargement automatique des modèles',
                'Gestion mémoire optimisée',
                'Support multi-device (CUDA/MPS/CPU)',
                'Validation d\'installation',
                'Génération mesh et texture',
                'Déchargement sélectif des modèles',
                'Monitoring mémoire GPU',
                'Gestion d\'erreurs robuste'
            ],
            'functions': [
                'check_environment: Vérification environnement',
                'load_models: Chargement des modèles',
                'generate_3d_mesh: Génération de mesh 3D',
                'apply_texture: Application de texture',
                'unload_models: Déchargement des modèles',
                'optimize_memory: Optimisation mémoire',
                'validate_installation: Validation complète'
            ]
        }


# Instance globale du gestionnaire
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Retourne l'instance globale du gestionnaire de modèles"""
    return model_manager


def check_environment() -> bool:
    """Fonction de convenance pour vérifier l'environnement"""
    return model_manager.check_environment()


def load_models(
    model_path: str = "tencent/Hunyuan3D-2",
    texture_model_path: str = "tencent/Hunyuan3D-2",
    device: Optional[str] = None,
    disable_texture: bool = False
) -> bool:
    """
    Fonction de convenance pour charger les modèles

    Args:
        model_path: Chemin vers le modèle de forme
        texture_model_path: Chemin vers le modèle de texture
        device: Device à utiliser
        disable_texture: Désactiver le modèle de texture

    Returns:
        True si succès, False sinon
    """
    global model_manager
    model_manager = ModelManager(
        model_path, texture_model_path, device, disable_texture)
    return model_manager.load_models()


def get_model_info() -> Dict[str, Any]:
    """Fonction de convenance pour obtenir les informations des modèles"""
    return model_manager.get_info()
