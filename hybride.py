
import json
import re
import numpy as np
import pandas as pd

from fonctions.def_n import (
    saisir_colonnes,
    call_llm,
    parse_llm_spec,
    creer_pdf_table,
    creer_odt_table,
    creer_csv_table,
    creer_dossiers_sortie,
    choisir_formats,
)

from fonctions.def_h import (
    _safe_int, 
    _truncate_words,
    _llm_generate_text_for_row,
    _llm_generate_derived_value,
    generer_prompt_llm,
    generate_dataframe_hybride,
    parse_llm_spec_robuste
)


theme = input("Entrer le thème général du dataset (le titre) : ").strip()
nb_lignes = int(input("Entrer le nombre de lignes : ").strip())

colonnes = saisir_colonnes()
formats = choisir_formats()

print("\n=== Génération du schéma JSON via LLM ===")
prompt = generer_prompt_llm(theme, colonnes, nb_lignes)
raw = call_llm(prompt)

spec_llm = parse_llm_spec_robuste(raw)
if spec_llm is None:
    print("Erreur: le LLM n'a pas renvoyé un JSON exploitable (ou a renvoyé null).")
    print("Réponse brute:")
    print(raw)
    raise SystemExit(1)

print("Schéma reçu :")
print(json.dumps(spec_llm, ensure_ascii=False, indent=2))

   
df = generate_dataframe_hybride(spec_llm, nb_lignes, theme=theme, seed=42)

print("\n=== Aperçu des données ===")
print(df.head())
print(f"\nDimensions: {df.shape}")
print("\nTypes de colonnes:")
print(df.dtypes)
print("\nStatistiques:")
print(df.describe(include="all"))

pdf_dir, odt_dir, csv_dir = creer_dossiers_sortie("output_pdf", "output_odt", "output_csv")

print("\n=== Génération des fichiers ===")

if formats.get("pdf"):
    creer_pdf_table(df, f"{pdf_dir}/test.pdf", titre=theme)
else:
    print("PDF : ignoré")

if formats.get("odt"):
    creer_odt_table(df, f"{odt_dir}/test.odt", titre=theme, max_rows=200, zebra=True)
else:
    print("ODT : ignoré")

if formats.get("csv"):
    creer_csv_table(
        df,
        f"{csv_dir}/test.csv",
        titre=theme,
        max_rows=None,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
        float_format="%.1f",
    )
else:
    print("CSV : ignoré")

print("\nGénération terminée ")
