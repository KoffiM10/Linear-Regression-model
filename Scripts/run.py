import os
import subprocess
import sys

def main():
    """Exécuter l'entraînement puis le dashboard"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(current_dir, "scripts")
    
    # Chemin vers train_model.py
    train_script = os.path.join(scripts_dir, "train_model.py")
    
    print("🔧 Vérification et entraînement du modèle...")
    
    # Exécuter l'entraînement
    try:
        subprocess.run([sys.executable, train_script], check=True)
        print("✅ Entraînement terminé avec succès!")
    except subprocess.CalledProcessError:
        print("❌ Erreur lors de l'entraînement")
        return
    
    # Lancer le dashboard
    print("🚀 Lancement du dashboard Streamlit...")
    dashboard_script = os.path.join(scripts_dir, "dashboard.py")
    subprocess.run(["streamlit", "run", dashboard_script])

if __name__ == "__main__":
    main()