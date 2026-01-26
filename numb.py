
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from tqdm import tqdm
from pydantic import BaseModel, ValidationError
from deepinfra_client import make_deepinfra_client


# ---------------------------
# COLLECTE DE BESOINS
# ---------------------------

# Extraire des mots + enlever les doublons en gardant l'ordre
def sans_doublons(texte: str) -> list[str]:
    mots = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?", texte.lower())
    seen = set()
    out = []
    for m in mots:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out

def check_non_vide(valeur: str, nom: str = "entrée") -> None:
    if not valeur or not valeur.strip():
        raise ValueError(f"{nom} vide.")


theme = input("Entrer le thème général du dataset : ")
check_non_vide(theme, "Thème")

lignes_p = input("Entrer les noms des lignes : ")
check_non_vide(lignes_p, "Noms des lignes")

colonne_p = input("Entrer les noms des colonnes : ")
check_non_vide(colonne_p, "Noms des colonnes")

lignes = sans_doublons(lignes_p)
colonnes = sans_doublons(colonne_p)


#----------------------------
#    CONVERSSION EN JSON
#----------------------------

spec  = {
    "theme": theme,
    "lignes": lignes,  
    "colonnes": colonnes,  
    "nb_lignes": len(lignes),
    "nb_colonnes": len(colonnes)}


prompt = f"""
Tu es un expert en modélisation statistique et génération de données synthétiques.

Objectif: proposer une spécification (schema) pour générer un dataset synthétique réaliste.

Contexte:
- Thème: "{spec['theme']}"
- Nombre de lignes à générer: {spec['nb_lignes']}
- Colonnes à inclure (ne pas en inventer d'autres): {spec['colonnes']}

Contraintes de sortie:
- Réponds UNIQUEMENT en JSON valide (sans markdown, sans texte supplémentaire).
- Pour CHAQUE colonne, indique au minimum: "type".
- Types autorisés: "string", "integer", "float", "boolean", "date", "category".
- N’indique une "distribution" que pour les colonnes numériques ("integer" ou "float").
- Pour "category": fournir "categories" + "probas" (somme = 1).
- Pour "string" libre ou texte: mettre "distribution": "none".
- Ne pas utiliser de données personnelles réelles.


Format EXACT attendu:
{{
  "columns": {{
    "nom_colonne": {{
      "type": "integer|float|boolean|date|string|category",
      "distribution": "normal|lognormal|uniform|poisson|none",
      "params": {{"...": "..."}},
      "categories": ["A", "B"],
      "probas": [0.6, 0.4]
    }}
  }},
  "row_rules": ["règle 1", "règle 2"]
}}
""".strip()

#-------------------------
# GENERER LA LOI DE PROBA
#-------------------------


model = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
client = make_deepinfra_client()

def call_llm(client, prompt: str,temperature: float = 0.2, frequency_penalty: float = 0.2) -> str:
    reponse =  client.chat.completions.create(
        model=model, 
        temperature = temperature, 
        frequency_penalty=frequency_penalty, 
        messages =[{"role": "user", "content": prompt}]
        )
    return reponse.choices[0].message.content

# fall back

def looser_ca_marche_pas(client, bad_text: str) -> str:
    repair_prompt = f"""
Le texte ci-dessous est censé être du JSON mais il est invalide.
Corrige-le pour produire UNIQUEMENT un JSON valide, sans markdown, sans texte en plus.

Texte:
{bad_text}
""".strip()

    return call_llm(client, repair_prompt, temperature=0.0, frequency_penalty=0.0)

    
raw = call_llm(client, prompt)

try:
    spec_llm = json.loads(raw)

except json.JSONDecodeError:
    repaired = looser_ca_marche_pas(client, raw)
    spec_llm = json.loads(repaired)

print(json.dumps(spec_llm, ensure_ascii=False, indent=2))


