# app_menu.py
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from deepinfra_client import make_deepinfra_client

from def_quali import (
    creer_odt,
    creer_pdf,
    write_files,
    inference,
)

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


def run_quali():
    """Générateur de pdf et odt avec l'IA : génère automatiquement des articles grâce aux LLM."""
    sujet_utilisateur = input("\nEntrez le sujet des articles : ").strip()
    if not sujet_utilisateur:
        print("Sujet vide, arrêt.")
        return 1

    try:
        num_documents = int(input("\nCombien de documents voulez-vous générer ? : ").strip())
    except ValueError:
        print("Il faut entrer un nombre.")
        return 1

    while num_documents <= 0:
        try:
            num_documents = int(input("Le nombre doit être positif. Combien de documents ? : ").strip())
        except ValueError:
            print("Il faut entrer un nombre.")
            return 1

    print("Génération du document..." if num_documents == 1 else "Génération des documents...")

    # Prépare le client + dossiers
    client = make_deepinfra_client()
    os.makedirs("output_pdf", exist_ok=True)
    os.makedirs("output_odt", exist_ok=True)

    # Inférences en parallèle
    results = []
    with ThreadPoolExecutor(max_workers=min(num_documents, 50)) as executor:
        futures = [
            executor.submit(inference, client, sujet_utilisateur)
            for _ in range(num_documents)
        ]

        for f in tqdm(as_completed(futures), total=num_documents, desc="Inférences"):
            try:
                txt = f.result()
            except Exception as e:
                print(f"\nErreur pendant une inférence: {e}")
                continue
            results.append(txt)

    if not results:
        print("Aucune sortie générée.")
        return 1

    # Écriture fichiers
    for i, txt in enumerate(results):
        try:
            write_files((txt, i))
        except Exception as e:
            print(f"Erreur écriture fichier #{i}: {e}")

    print(f"\nTerminé: {len(results)} document(s) généré(s).")
    return 0


def run_quant():
    """Génération d'un dataset + export table en PDF/ODT."""
    theme = input("\nEntrer le thème général du dataset (le titre) : ").strip()
    if not theme:
        print(" Thème vide, arrêt.")
        return 1

    try:
        nb_lignes = int(input("Entrer le nombre de lignes : ").strip())
    except ValueError:
        print(" Il faut entrer un nombre.")
        return 1

    if nb_lignes <= 0:
        print("Le nombre de lignes doit être > 0.")
        return 1

    colonnes = saisir_colonnes()
    print("\nGénération du schéma JSON via LLM...")

    prompt = generer_prompt_llm(theme, colonnes, nb_lignes)
    raw = call_llm(prompt)
    spec_llm = parse_llm_spec(raw)

    if spec_llm is None:
        print("\n❌ Erreur: le LLM n'a pas renvoyé un JSON exploitable (ou a renvoyé null).")
        print("Réponse brute:")
        print(raw)
        return 1

    print("\nSchéma reçu :")
    print(json.dumps(spec_llm, ensure_ascii=False, indent=2))

    df = generate_dataframe(spec_llm, nb_lignes, seed=42)

    print("\nAperçu :")
    print(df.head())
    print("\nInfos :")
    print("shape:", df.shape)
    print(df.dtypes)

    try:
        print("\nStats :")
        print(df.describe(include="all"))
    except Exception:
        pass

    pdf_dir, odt_dir = creer_dossiers_sortie("output_pdf", "output_odt")

    pdf_path = f"{pdf_dir}/dataset.pdf"
    odt_path = f"{odt_dir}/dataset.odt"

    creer_pdf_table(df, pdf_path, titre=theme)
    creer_odt_table(df, odt_path, titre=theme, max_rows=200, zebra=True)

    print(f"\n✅ Exports OK:\n- {pdf_path}\n- {odt_path}")
    return 0


def main():
    while True:
    
        print("1) Générer des articles (PDF/ODT) via LLM ")
        print("2) Générer un dataset + table (PDF/ODT) ")
        print("0) Quitter")
        choix = input("\nTon choix : ").strip()

        if choix == "1":
            code = run_quali()
            print(f"\n[Exit code: {code}]")

        elif choix == "2":
            code = run_quant()
            print(f"\n[Exit code: {code}]")

        elif choix == "0":
            raise SystemExit(0)

        else:
            print(" Choix invalide. Tape 1, 2, ou 0.")


if __name__ == "__main__":
    main()
