"""
Générateur de pdf et odt avec l'ia
Il génère automatiquement des articles grâce aux llm
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from deepinfra_client import make_deepinfra_client

from def_quali import (
    creer_odt, 
    creer_pdf, 
    write_files, 
    inference )

sujet_utilisateur = input("\nEntrez le sujet des articles : ").strip()

NUM_DOCUMENTS = int(input("\nCombien de documents voulez-vous générer ? :"))
while NUM_DOCUMENTS<= 0:
    NUM_DOCUMENTS = int(input("le nombre doit etre positif! Combien de documents voulez vous : "))

if NUM_DOCUMENTS == 1:
    print("Generation du document..." )
else: 
    print("Generation des documents ...")


client = make_deepinfra_client()

os.makedirs("output_pdf", exist_ok=True)
os.makedirs("output_odt", exist_ok=True)


with ThreadPoolExecutor(max_workers=min(NUM_DOCUMENTS, 50)) as executor:
    futures = [executor.submit(inference, client, sujet_utilisateur) for _ in range(NUM_DOCUMENTS)]
    results = []

    for f in tqdm(as_completed(futures), total=NUM_DOCUMENTS, desc="Inférences"):
        txt = f.result()
        results.append(txt)


for i in range(len(results)):
    write_files((results[i], i))

   