#!/usr/bin/env python3
"""
Wrapper pour l'installation de Hunyuan3D-2
Appelle le véritable installateur dans lib/install-hunyuan3d.py
"""

import os
import sys
from pathlib import Path


def main():
    """Fonction principale qui délègue à l'installateur dans lib/"""
    print("🔄 Démarrage de l'installation Hunyuan3D-2...")
    print("   Délégation vers lib/install-hunyuan3d.py")
    print("=" * 50)

    # Chemin vers le véritable installateur
    lib_dir = Path(__file__).parent / "lib"
    installer_path = lib_dir / "install-hunyuan3d.py"

    if not installer_path.exists():
        print(f"❌ Installateur non trouvé: {installer_path}")
        print("💡 Assurez-vous que lib/install-hunyuan3d.py existe")
        return False

    # Exécuter l'installateur
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(installer_path)], cwd=str(Path.cwd()))
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
