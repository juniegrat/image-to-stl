#!/usr/bin/env python3
"""
Configuration et paramètres pour Hunyuan3D-2mv
Gère tous les modes de qualité et leurs paramètres
"""

from typing import Dict, Any


class QualityMode:
    """Énumération des modes de qualité disponibles"""
    DEBUG = "debug"
    TEST = "test"
    FAST = "fast"
    HIGH = "high"
    ULTRA = "ultra"


class Hunyuan3DConfig:
    """
    Configuration centralisée pour Hunyuan3D-2mv
    Gère tous les paramètres selon les modes de qualité
    """

    def __init__(self):
        """Initialise avec la configuration par défaut (HIGH)"""
        self._configs = self._create_all_configs()
        self._current_mode = QualityMode.HIGH
        self.config = self._configs[self._current_mode].copy()

    def _create_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Crée toutes les configurations de qualité"""
        return {
            QualityMode.DEBUG: self._create_debug_config(),
            QualityMode.TEST: self._create_test_config(),
            QualityMode.FAST: self._create_fast_config(),
            QualityMode.HIGH: self._create_high_config(),
            QualityMode.ULTRA: self._create_ultra_config()
        }

    def _create_debug_config(self) -> Dict[str, Any]:
        """Configuration DEBUG ultra-minimal pour tests instantanés"""
        return {
            # Paramètres de génération (rapides mais cohérents)
            'image_size': 256,
            'guidance_scale': 3.0,
            'num_inference_steps': 15,
            'octree_resolution': 96,
            'num_chunks': 1500,
            'texture_guidance_scale': 2.0,
            'texture_steps': 8,
            'seed': 42,

            # Paramètres de rendu (simplifiés mais corrects)
            'n_views': 8,
            'elevation_deg': 5.0,
            'camera_distance': 2.0,
            'fovy_deg': 40.0,
            'height': 256,
            'width': 256,
            'fps': 12,
            'foreground_ratio': 0.8,

            # Flags de mode
            'debug_mode': True,
            'quick_render': True,
            'skip_post_processing': True,
            'simple_mesh': True,
            'preserve_shape': True,
            'minimal_vertices': True,

            # Descriptions
            'mode_name': 'DEBUG',
            'description': 'Mode debug ultra-minimal pour tests instantanés'
        }

    def _create_test_config(self) -> Dict[str, Any]:
        """Configuration TEST ultra-rapide pour développement"""
        return {
            # Paramètres de génération (ultra-rapides)
            'image_size': 256,
            'guidance_scale': 2.0,
            'num_inference_steps': 10,
            'octree_resolution': 64,
            'num_chunks': 1000,
            'texture_guidance_scale': 1.5,
            'texture_steps': 8,
            'seed': 42,

            # Paramètres de rendu (simplifiés)
            'n_views': 8,
            'elevation_deg': 0.0,
            'camera_distance': 2.5,
            'fovy_deg': 45.0,
            'height': 256,
            'width': 256,
            'fps': 12,
            'foreground_ratio': 0.8,

            # Flags de mode
            'test_mode': True,
            'quick_render': True,
            'skip_post_processing': True,
            'low_precision': True,

            # Descriptions
            'mode_name': 'TEST',
            'description': 'Mode test ultra-rapide pour développement'
        }

    def _create_fast_config(self) -> Dict[str, Any]:
        """Configuration RAPIDE (compromis qualité/vitesse)"""
        return {
            # Paramètres de génération
            'image_size': 512,
            'guidance_scale': 7.0,
            'num_inference_steps': 50,
            'octree_resolution': 192,
            'num_chunks': 5000,
            'texture_guidance_scale': 3.0,
            'texture_steps': 25,
            'seed': 42,

            # Paramètres de rendu équilibrés
            'n_views': 24,
            'elevation_deg': 12.0,
            'camera_distance': 1.6,
            'fovy_deg': 35.0,
            'height': 512,
            'width': 512,
            'fps': 24,
            'foreground_ratio': 0.90,

            # Optimisations
            'fast_mode': True,
            'moderate_post_processing': True,

            # Descriptions
            'mode_name': 'RAPIDE',
            'description': 'Compromis qualité/vitesse'
        }

    def _create_high_config(self) -> Dict[str, Any]:
        """Configuration HIGH (optimisé pour pièces)"""
        return {
            # Paramètres de génération (niveau "high" - optimisé pour pièces)
            'image_size': 1024,
            'guidance_scale': 15.0,
            'num_inference_steps': 100,
            'octree_resolution': 380,
            'num_chunks': 20000,
            'texture_guidance_scale': 6.0,
            'texture_steps': 60,
            'seed': 12345,

            # Paramètres de rendu optimisés
            'n_views': 36,
            'elevation_deg': 15.0,
            'camera_distance': 1.5,
            'fovy_deg': 30.0,
            'height': 1024,
            'width': 1024,
            'fps': 30,
            'foreground_ratio': 0.95,

            # Descriptions
            'mode_name': 'HIGH',
            'description': 'Qualité élevée optimisée pour pièces numismatiques'
        }

    def _create_ultra_config(self) -> Dict[str, Any]:
        """Configuration ULTRA qualité maximale"""
        return {
            # Paramètres de génération (qualité maximale)
            'image_size': 1024,
            'guidance_scale': 20.0,
            'num_inference_steps': 150,
            'octree_resolution': 512,
            'num_chunks': 30000,
            'texture_guidance_scale': 8.0,
            'texture_steps': 80,
            'seed': 12345,

            # Paramètres de rendu premium
            'n_views': 48,
            'elevation_deg': 20.0,
            'camera_distance': 1.4,
            'fovy_deg': 25.0,
            'height': 1024,
            'width': 1024,
            'fps': 30,
            'foreground_ratio': 0.98,

            # Optimisations qualité
            'ultra_mode': True,
            'max_post_processing': True,
            'anti_aliasing': True,
            'detail_preservation': True,

            # Descriptions
            'mode_name': 'ULTRA',
            'description': 'Qualité maximale - temps de rendu élevé'
        }

    def set_mode(self, mode: str):
        """
        Change le mode de qualité actuel

        Args:
            mode: Mode de qualité (debug, test, fast, high, ultra)
        """
        if mode not in self._configs:
            raise ValueError(
                f"Mode '{mode}' non supporté. Modes disponibles: {list(self._configs.keys())}")

        self._current_mode = mode
        self.config = self._configs[mode].copy()

        print(f"🔧 Mode {self.config['mode_name']} activé")
        print(f"   📝 {self.config['description']}")
        self._print_key_parameters()

    def _print_key_parameters(self):
        """Affiche les paramètres clés du mode actuel"""
        print(
            f"   📊 Résolution: {self.config['image_size']}x{self.config['image_size']}")
        print(f"   📊 Guidance scale: {self.config['guidance_scale']}")
        print(f"   📊 Steps: {self.config['num_inference_steps']}")
        print(f"   📊 Octree resolution: {self.config['octree_resolution']}")
        print(f"   📊 Chunks: {self.config['num_chunks']}")
        print(f"   📊 Texture steps: {self.config['texture_steps']}")
        print(f"   📊 Rendus: {self.config['n_views']} vues")

    def get_current_mode(self) -> str:
        """Retourne le mode actuel"""
        return self._current_mode

    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle"""
        return self.config.copy()

    def update_config(self, updates: Dict[str, Any]):
        """
        Met à jour des paramètres spécifiques

        Args:
            updates: Dictionnaire des paramètres à mettre à jour
        """
        self.config.update(updates)
        print(f"🔧 Configuration mise à jour: {list(updates.keys())}")

    def get_available_modes(self) -> Dict[str, str]:
        """Retourne la liste des modes disponibles avec leurs descriptions"""
        return {
            mode: config['description']
            for mode, config in self._configs.items()
        }

    def print_all_modes(self):
        """Affiche tous les modes disponibles"""
        print("🎯 Modes de qualité disponibles:")
        for mode, description in self.get_available_modes().items():
            marker = "→" if mode == self._current_mode else " "
            print(f"  {marker} {mode.upper()}: {description}")

    def is_debug_mode(self) -> bool:
        """Vérifie si on est en mode debug"""
        return self.config.get('debug_mode', False)

    def is_test_mode(self) -> bool:
        """Vérifie si on est en mode test"""
        return self.config.get('test_mode', False)

    def is_fast_mode(self) -> bool:
        """Vérifie si on est en mode fast"""
        return self.config.get('fast_mode', False)

    def is_ultra_mode(self) -> bool:
        """Vérifie si on est en mode ultra"""
        return self.config.get('ultra_mode', False)

    def should_skip_post_processing(self) -> bool:
        """Vérifie si le post-processing doit être ignoré"""
        return self.config.get('skip_post_processing', False)

    def get_render_params(self) -> Dict[str, Any]:
        """Retourne uniquement les paramètres de rendu"""
        render_keys = [
            'n_views', 'elevation_deg', 'camera_distance', 'fovy_deg',
            'height', 'width', 'fps', 'foreground_ratio'
        ]
        return {key: self.config[key] for key in render_keys if key in self.config}

    def get_generation_params(self) -> Dict[str, Any]:
        """Retourne uniquement les paramètres de génération"""
        gen_keys = [
            'image_size', 'guidance_scale', 'num_inference_steps',
            'octree_resolution', 'num_chunks', 'texture_guidance_scale',
            'texture_steps', 'seed'
        ]
        return {key: self.config[key] for key in gen_keys if key in self.config}


# Instance globale de configuration
hunyuan3d_config = Hunyuan3DConfig()


def get_config() -> Hunyuan3DConfig:
    """Retourne l'instance globale de configuration"""
    return hunyuan3d_config


def set_quality_mode(mode: str):
    """
    Fonction de convenance pour changer le mode de qualité

    Args:
        mode: Mode de qualité (debug, test, fast, high, ultra)
    """
    hunyuan3d_config.set_mode(mode)


def get_current_config() -> Dict[str, Any]:
    """Fonction de convenance pour obtenir la configuration actuelle"""
    return hunyuan3d_config.get_config()


def print_available_modes():
    """Fonction de convenance pour afficher les modes disponibles"""
    hunyuan3d_config.print_all_modes()
