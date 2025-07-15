"""
Convertisseur d'Images vers STL avec TripoSR
Package modulaire pour une meilleure organisation du code
"""

__version__ = "1.0.0"
__author__ = "Image-to-STL"

# Imports principaux pour faciliter l'utilisation
from .converter import convert_png_to_stl, convert_coin_to_stl_safe
from .image_processor import detect_and_convert_image_format
from .setup import setup_triposr, check_and_install_dependencies
from .diagnostics import diagnostic_info, check_cuda_compatibility
from .utils import clear_gpu_memory, get_render_params

__all__ = [
    'convert_png_to_stl',
    'convert_coin_to_stl_safe',
    'detect_and_convert_image_format',
    'setup_triposr',
    'check_and_install_dependencies',
    'diagnostic_info',
    'check_cuda_compatibility',
    'clear_gpu_memory',
    'get_render_params'
]

# Module de compatibilité pour Hunyuan3D
try:
    from .hunyuan3d_converter import Hunyuan3DConverter, convert_coin_hunyuan3d
    __all__.extend(['Hunyuan3DConverter', 'convert_coin_hunyuan3d'])
except ImportError:
    pass  # Hunyuan3D optionnel
