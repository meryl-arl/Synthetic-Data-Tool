"""
test_generation.py
Fichier de test pour générer un seul document
"""

import random
import os
from dotenv import load_dotenv
from deepinfra_client import make_deepinfra_client

# Import des fonctions du fichier principal
# Supposons que ton fichier s'appelle "generator.py"
from synthetic_generator import creer_pdf, creer_odt

# Configuration
load_dotenv()

TECHNOLOGIES_LIST = [
    "AI", "Tesla", "blockchain", "robotique", "impression 3D", 
    "cybersécurité", "espace", "énergie durable", "agriculture",
    "éducation", "communication", "recherche scientifique", "audio",
    "juridique", "marketing", "ressources humaines", "service client", 
    "bases de données"
]

# Créer les dossiers de sortie s'ils n'existent pas
os.makedirs("output_pdf", exist_ok=True)
os.makedirs("output_odt", exist_ok=True)


def generer_texte_ia(client, sujet: str) -> str:
    """
    Génère un texte via l'API DeepInfra sur un sujet donné
    """
    prompt = f"Crée un texte de plusieurs paragraphes sur le thème de la technologie et de l'innovation en 2024, parle de ce sujet spécifique : {sujet}"
    
    print(f" Génération du texte sur le sujet: {sujet}...")
    
    response = client.chat.completions.create(
        temperature=0.8,
        frequency_penalty=0.9,
        model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    
    return response.choices[0].message.content


if __name__ == "__main__":
    print("Démarrage du test de génération de document...\n")
    
    # Initialiser le client
    client = make_deepinfra_client()
    
    # Choisir un sujet aléatoire
    sujet_choisi = random.choice(TECHNOLOGIES_LIST)
    
    # Générer le texte
    texte_genere = generer_texte_ia(client, sujet_choisi)
    
    print(f"\nTexte généré ({len(texte_genere)} caractères)")
    print(f"Aperçu: {texte_genere[:200]}...\n")
    
    # Créer les documents
    print(" Création des documents...")
    creer_pdf(texte_genere, "output_pdf/article_test.pdf")
    creer_odt(texte_genere, "output_odt/article_test.odt")
    
    print(" Test terminé avec succès!")
    print(f" Fichiers créés dans:")
    print(f"   - output_pdf/article_test.pdf")
    print(f"   - output_odt/article_test.odt")