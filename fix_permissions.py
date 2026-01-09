import os
import stat
import subprocess
import sys

def fix_permissions():
    """Corriger les permissions des fichiers"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Fichiers à vérifier
    model_file = os.path.join(parent_dir, "model.pkl")
    scripts_dir = os.path.join(parent_dir, "scripts")
    
    print("🔧 Vérification des permissions...")
    
    # Donner tous les droits au dossier scripts
    try:
        os.chmod(scripts_dir, stat.S_IRWXU)
        print(f"✅ Permissions corrigées pour: {scripts_dir}")
    except Exception as e:
        print(f"⚠️ Impossible de modifier {scripts_dir}: {e}")
    
    # Si model.pkl existe, essayer de le supprimer
    if os.path.exists(model_file):
        try:
            os.chmod(model_file, stat.S_IRWXU)
            os.remove(model_file)
            print("✅ Ancien modèle supprimé")
        except Exception as e:
            print(f"⚠️ Impossible de supprimer {model_file}: {e}")
    
    print("🎯 Maintenant exécutez train_model.py à nouveau")

if __name__ == "__main__":
    fix_permissions()