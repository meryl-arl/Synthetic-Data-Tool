import json
from fonctions.def_n import (
    saisir_colonnes,
    generer_prompt_llm,
    call_llm,
    parse_llm_spec,
    generate_dataframe,
    creer_pdf_table,
    creer_odt_table,
    creer_csv_table,
    creer_dossiers_sortie,
    choisir_formats,
)

# === Programme principal ===

theme = input("Entrer le thème général du dataset (le titre) : ").strip()
nb_lignes = int(input("Entrer le nombre de lignes : ").strip())

colonnes = saisir_colonnes()

# Choix des formats
formats = choisir_formats()

print("\n=== Génération du schéma JSON via LLM ===")

prompt = generer_prompt_llm(theme, colonnes, nb_lignes)
raw = call_llm(prompt)

spec_llm = parse_llm_spec(raw)
if spec_llm is None:
    print("Erreur: le LLM n'a pas renvoyé un JSON exploitable (ou a renvoyé null).")
    print("Réponse brute:")
    print(raw)
    raise SystemExit(1)

print("Schéma reçu :")
print(json.dumps(spec_llm, ensure_ascii=False, indent=2))

# Génération du DataFrame
df = generate_dataframe(spec_llm, nb_lignes, seed=42)

print("\n=== Aperçu des données ===")
print(df.head())
print(f"\nDimensions: {df.shape}")
print("\nTypes de colonnes:")
print(df.dtypes)
print("\nStatistiques:")
print(df.describe(include="all"))

# Création des dossiers de sortie
pdf_dir, odt_dir, csv_dir = creer_dossiers_sortie("output_pdf", "output_odt", "output_csv")

# Génération des fichiers selon les choix de l'utilisateur
print("\n=== Génération des fichiers ===")

if formats['pdf']:
    creer_pdf_table(df, f"{pdf_dir}/test.pdf", titre=theme)
else:
    print("PDF : ignoré")

if formats['odt']:
    creer_odt_table(df, f"{odt_dir}/test.odt", titre=theme, max_rows=200, zebra=True)
else:
    print("ODT : ignoré")

if formats['csv']:
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

print("\nGénération terminée !")
