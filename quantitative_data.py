import json
from def_quant import (
    saisir_colonnes,
    generer_prompt_llm,
    call_llm,
    parse_llm_spec,
    generate_dataframe,
    creer_pdf_table,
    creer_odt_table,
    creer_dossiers_sortie,
)


theme = input("Entrer le thème général du dataset (le titre) : ").strip()
nb_lignes = int(input("Entrer le nombre de lignes : ").strip())

colonnes = saisir_colonnes()
print("Génération du schéma JSON via LLM...")

prompt = generer_prompt_llm(theme, colonnes, nb_lignes)
raw = call_llm(prompt)

spec_llm = parse_llm_spec(raw)

if spec_llm is None:
    print("Erreur: le LLM n'a pas renvoyé un JSON exploitable (ou a renvoyé null).")
    print("Réponse brute:")
    print(raw)
    

print("Schéma reçu :")
print(json.dumps(spec_llm, ensure_ascii=False, indent=2))

df = generate_dataframe(spec_llm, nb_lignes, seed=42)

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe(include="all"))

pdf_dir, odt_dir = creer_dossiers_sortie("output_pdf", "output_odt")
creer_pdf_table(df, f"{pdf_dir}/test.pdf", titre=theme)
creer_odt_table(df, f"{odt_dir}/test.odt", titre=theme, max_rows=200, zebra=True)

