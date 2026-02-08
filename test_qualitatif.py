# ============================================================
# MAIN — Génération de documents qualitatifs via LLM + exports
# ============================================================

import os 
from concurrent.futures import ThreadPoolExecutor, as_completed  
from tqdm import tqdm  


from utils.deepinfra_client import make_deepinfra_client

from utils.utils_qualitatif import (
    creer_odt,        
    creer_pdf,        
    creer_csv,        
    write_files,      
    inference,        
    choisir_formats  
)

# ============================================================
# Collecte des informations de l'utilisateur
# ============================================================

sujet_utilisateur = input("\nEntrez le sujet des articles : ").strip()

NUM_DOCUMENTS = int(input("\nCombien de documents voulez-vous générer ? : "))

while NUM_DOCUMENTS <= 0:
    NUM_DOCUMENTS = int(input("Le nombre doit être positif! Combien de documents voulez-vous ? : "))

formats = choisir_formats()

if NUM_DOCUMENTS == 1:
    print("\nGénération du document...")
else:
    print(f"\nGénération des {NUM_DOCUMENTS} documents...")


# Création d'un client 
client = make_deepinfra_client()

# ============================================================
# Préparation des dossiers de sortie selon les formats choisis
# ============================================================

if formats['pdf']:
    os.makedirs("output_pdf", exist_ok=True)
if formats['odt']:
    os.makedirs("output_odt", exist_ok=True)
if formats['csv']:
    os.makedirs("output_csv", exist_ok=True)

# ============================================================
#  Parallélisation des inférences LLM (génération des textes)
# ============================================================

with ThreadPoolExecutor(max_workers=min(NUM_DOCUMENTS, 50)) as executor:

    futures = [
        executor.submit(inference, client, sujet_utilisateur)
        for _ in range(NUM_DOCUMENTS)
    ]

    results = []

    for f in tqdm(as_completed(futures), total=NUM_DOCUMENTS, desc="Inférences LLM"):
        txt = f.result()    
        results.append(txt) 

# ============================================================
# Écriture des fichiers 
# ============================================================

print("\n=== Création des fichiers ===")

for i in range(len(results)):
    write_files((results[i], i, formats))

print("\nGénération terminée !")
