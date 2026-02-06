# cli.py
# Interface CLI (menu) qui orchestre : textuel / quanti / hybride
# Dépendance : click
# Usage :
#   python cli.py
# ou (optionnel) :
#   python cli.py menu
#   python cli.py quali
#   python cli.py text
#   python cli.py hybride

import os
import json
import click


from fonctions.deepinfra_client import make_deepinfra_client
from fonctions.def_t import (
    creer_odt,
    creer_pdf,
    creer_csv,
    write_files,
    inference,
    choisir_formats as choisir_formats_t,
)


from fonctions.def_h import (
    generer_prompt_llm as generer_prompt_llm_h,
    generate_dataframe_hybride,
    parse_llm_spec_robuste,
)

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Quanti
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
    choisir_formats as choisir_formats_n, 
)


def _ask_positive_int(prompt: str) -> int:
    while True:
        s = click.prompt(prompt, type=str).strip()
        try:
            n = int(s)
        except ValueError:
            click.echo("Entrée invalide. Mets un entier (ex: 100).")
            continue
        if n <= 0:
            click.echo("Le nombre doit être > 0.")
            continue
        return n


def _mkdir_if_needed(formats: dict, pdf="output_pdf", odt="output_odt", csv="output_csv"):
    if formats.get("pdf"):
        os.makedirs(pdf, exist_ok=True)
    if formats.get("odt"):
        os.makedirs(odt, exist_ok=True)
    if formats.get("csv"):
        os.makedirs(csv, exist_ok=True)


def _print_df_summary(df):
    click.echo("\n     Aperçu des données   ")
    click.echo(str(df.head()))
    click.echo(f"\nDimensions: {df.shape}")
    click.echo("\nTypes de colonnes:")
    click.echo(str(df.dtypes))
    click.echo("\nStatistiques:")
    try:
        click.echo(str(df.describe(include="all")))
    except Exception as e:
        click.echo(f"(describe indisponible: {e})")



@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Synthetic Data Tool — CLI menu (textuel / quanti / hybride)."""
    if ctx.invoked_subcommand is None:
        # Par défaut : menu interactif
        ctx.invoke(menu)


@cli.command()
def menu():
    """Menu interactif."""
    click.echo("\n====MENU====")
    click.echo("1) Data textuelle ")
    click.echo("2) Data quantitative ")
    click.echo("3) Data mixte ")
    click.echo("0) Quitter\n")

    choice = click.prompt("Choix", type=click.IntRange(0, 3))
    if choice == 0:
        raise SystemExit(0)
    if choice == 1:
        run_text()
    elif choice == 2:
        run_quanti()
    elif choice == 3:
        run_hybride()


@cli.command("text")
def cmd_text():
    """Mode textuel (équivalent text_data.py)."""
    run_text()


@cli.command("quali")
def cmd_quali():
    """Alias de text ."""
    run_text()


@cli.command("quanti")
def cmd_quanti():
    """Mode quantitatif (équivalent numb_data.py)."""
    run_quanti()


@cli.command("hybride")
def cmd_hybride():
    """Mode hybride (équivalent hybrise.py)."""
    run_hybride()



def run_quanti():
    theme = click.prompt("Entrer le thème général du dataset (le titre)", type=str).strip()
    nb_lignes = _ask_positive_int("Entrer le nombre de lignes")

    colonnes = saisir_colonnes()


    formats = choisir_formats_n()

    click.echo("\n Génération du schéma JSON via LLM (30s a 1min))")
    prompt = generer_prompt_llm(theme, colonnes, nb_lignes)
    raw = call_llm(prompt)

    spec_llm = parse_llm_spec(raw)
    if spec_llm is None:
        click.echo(" \n ERREUR: le LLM n'a pas renvoyé un JSON exploitable (ou a renvoyé null).")
        click.echo("Réponse brute:")
        click.echo(str(raw))
        raise SystemExit(1)

    click.echo("Schéma reçu :")
    click.echo(json.dumps(spec_llm, ensure_ascii=False, indent=2))

    df = generate_dataframe(spec_llm, nb_lignes, seed=42)
    _print_df_summary(df)

    pdf_dir, odt_dir, csv_dir = creer_dossiers_sortie("output_pdf", "output_odt", "output_csv")

    click.echo("\n=== Génération des fichiers ===")
    if formats.get("pdf"):
        creer_pdf_table(df, f"{pdf_dir}/test.pdf", titre=theme)
    else:
        click.echo("PDF : ignoré")

    if formats.get("odt"):
        creer_odt_table(df, f"{odt_dir}/test.odt", titre=theme, max_rows=200, zebra=True)
    else:
        click.echo("ODT : ignoré")

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
        click.echo("CSV : ignoré")

    click.echo("\n Génération terminée !")


def run_hybride():
    theme = click.prompt("Entrer le thème général du dataset (le titre)", type=str).strip()
    nb_lignes = _ask_positive_int("Entrer le nombre de lignes")

    colonnes = saisir_colonnes()


    formats = choisir_formats_n()

    click.echo("\n=== Génération du schéma JSON via LLM ===")
    prompt = generer_prompt_llm_h(theme, colonnes, nb_lignes)
    raw = call_llm(prompt)

    spec_llm = parse_llm_spec_robuste(raw)
    if spec_llm is None:
        click.echo(" Erreur: le LLM n'a pas renvoyé un JSON exploitable (ou a renvoyé null).")
        click.echo("Réponse brute:")
        click.echo(str(raw))
        raise SystemExit(1)

    click.echo("Schéma reçu :")
    click.echo(json.dumps(spec_llm, ensure_ascii=False, indent=2))

    df = generate_dataframe_hybride(spec_llm, nb_lignes, theme=theme, seed=42)
    _print_df_summary(df)

    pdf_dir, odt_dir, csv_dir = creer_dossiers_sortie("output_pdf", "output_odt", "output_csv")

    click.echo("\n=== Génération des fichiers ===")
    if formats.get("pdf"):
        creer_pdf_table(df, f"{pdf_dir}/test.pdf", titre=theme)
    else:
        click.echo("PDF : ignoré")

    if formats.get("odt"):
        creer_odt_table(df, f"{odt_dir}/test.odt", titre=theme, max_rows=200, zebra=True)
    else:
        click.echo("ODT : ignoré")

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
        click.echo("CSV : ignoré")

    click.echo("\n Génération terminée !")


def run_text():
    sujet_utilisateur = click.prompt("Entrez le sujet des articles", type=str).strip()
    num_docs = _ask_positive_int("Combien de documents voulez-vous générer ?")

  
    formats = choisir_formats_t()

    if num_docs == 1:
        click.echo("\nGénération du document...")
    else:
        click.echo(f"\nGénération des {num_docs} documents...")

    client = make_deepinfra_client()

   
    _mkdir_if_needed(formats, "output_pdf", "output_odt", "output_csv")


    with ThreadPoolExecutor(max_workers=min(num_docs, 50)) as executor:
        futures = [executor.submit(inference, client, sujet_utilisateur) for _ in range(num_docs)]
        results = []
        for f in tqdm(as_completed(futures), total=num_docs, desc="Inférences LLM"):
            results.append(f.result())

    click.echo("\n=== Création des fichiers ===")
    for i, txt in enumerate(results):
        write_files((txt, i, formats))

    click.echo("\nGénération terminée !")


if __name__ == "__main__":
    cli()
