import click
import os
import sys
from pathlib import Path
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



# ============================================================================
# COMMANDES PRINCIPALES
# ============================================================================

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    🤖 Synthetic Data Tool - Générateur de documents et datasets avec l'IA
    
    Utilise l'IA pour générer automatiquement :
    - Des articles/documents (PDF/ODT)
    - Des datasets structurés (CSV/PDF/ODT)
    
    Exemples :
    
      synthetic-cli quali --sujet "Intelligence Artificielle" --nombre 3
      synthetic-cli quant --theme "Ventes 2024" --lignes 100
      synthetic-cli quali --interactive
    """
    pass


# ============================================================================
# COMMANDE QUALI : Génération d'articles
# ============================================================================

@cli.command()
@click.option('--sujet', '-s', 
              help='Le sujet des articles à générer')
@click.option('--nombre', '-n', type=int, 
              help='Nombre de documents à générer')
@click.option('--interactive', '-i', is_flag=True, 
              help='Mode interactif (demande les infos)')
@click.option('--output-pdf', default='output_pdf', 
              help='Dossier de sortie pour les PDF')
@click.option('--output-odt', default='output_odt', 
              help='Dossier de sortie pour les ODT')
def quali(sujet, nombre, interactive, output_pdf, output_odt):
    """
    📝 Génère des articles en PDF et ODT
    
    Exemples :
    
      # Mode non-interactif
      synthetic-cli quali -s "IA et Santé" -n 5
      
      # Mode interactif
      synthetic-cli quali --interactive
      
      # Avec dossiers personnalisés
      synthetic-cli quali -s "Finance" -n 3 --output-pdf mes_pdfs
    """
    
    click.echo("=" * 60)
    click.echo("📝 GÉNÉRATEUR D'ARTICLES (QUALI)")
    click.echo("=" * 60)
    
    # Mode interactif
    if interactive or not sujet or nombre is None:
        sujet = click.prompt('\n📌 Entrez le sujet des articles', type=str)
        nombre = click.prompt('📊 Combien de documents voulez-vous générer ?', type=int)
    
    # Validation
    if not sujet or not sujet.strip():
        click.echo("❌ Le sujet ne peut pas être vide", err=True)
        return 1
    
    if nombre <= 0:
        click.echo("❌ Le nombre doit être positif", err=True)
        return 1
    
    # Confirmation
    click.echo(f"\n✅ Configuration :")
    click.echo(f"   • Sujet : {sujet}")
    click.echo(f"   • Nombre : {nombre}")
    click.echo(f"   • Output PDF : {output_pdf}/")
    click.echo(f"   • Output ODT : {output_odt}/")
    
    if not click.confirm("\n🚀 Lancer la génération ?", default=True):
        click.echo("❌ Annulé")
        return 0
    
    # Exécution
    try:
        code = run_quali_generation(sujet, nombre, output_pdf, output_odt)
        
        if code == 0:
            click.echo(f"\n✅ {nombre} document(s) généré(s) avec succès ! 🎉")
            click.echo(f"📂 Fichiers disponibles dans {output_pdf}/ et {output_odt}/")
        else:
            click.echo(f"\n⚠️  Génération terminée avec des erreurs", err=True)
        
        return code
        
    except KeyboardInterrupt:
        click.echo("\n\n❌ Génération interrompue par l'utilisateur")
        return 130
    except Exception as e:
        click.echo(f"\n❌ Erreur inattendue : {e}", err=True)
        return 1


def run_quali_generation(sujet_utilisateur, num_documents, output_pdf, output_odt):
    """Fonction de génération quali (non-interactive)"""
    
    click.echo(f"\n🔧 Préparation...")
    
    # Prépare le client + dossiers
    try:
        client = make_deepinfra_client()
    except Exception as e:
        click.echo(f"❌ Erreur de connexion au client DeepInfra : {e}", err=True)
        return 1
    
    os.makedirs(output_pdf, exist_ok=True)
    os.makedirs(output_odt, exist_ok=True)
    
    click.echo(f"🤖 Génération de {num_documents} document(s) en parallèle...")

    # Inférences en parallèle
    results = []
    with ThreadPoolExecutor(max_workers=min(num_documents, 50)) as executor:
        futures = [
            executor.submit(inference, client, sujet_utilisateur)
            for _ in range(num_documents)
        ]

        for f in tqdm(as_completed(futures), total=num_documents, desc="⚡ Inférences"):
            try:
                txt = f.result()
                results.append(txt)
            except Exception as e:
                click.echo(f"\n⚠️  Erreur pendant une inférence: {e}", err=True)
                continue

    if not results:
        click.echo("❌ Aucune sortie générée.", err=True)
        return 1

    # Écriture fichiers
    click.echo(f"\n💾 Écriture de {len(results)} fichier(s)...")
    
    # Temporairement change les dossiers globaux
    import def_quali
    original_pdf = getattr(def_quali, 'OUTPUT_PDF_DIR', 'output_pdf')
    original_odt = getattr(def_quali, 'OUTPUT_ODT_DIR', 'output_odt')
    
    # Override si possible
    if hasattr(def_quali, 'OUTPUT_PDF_DIR'):
        def_quali.OUTPUT_PDF_DIR = output_pdf
    if hasattr(def_quali, 'OUTPUT_ODT_DIR'):
        def_quali.OUTPUT_ODT_DIR = output_odt
    
    for i, txt in enumerate(results):
        try:
            write_files((txt, i))
        except Exception as e:
            click.echo(f"⚠️  Erreur écriture fichier #{i}: {e}", err=True)
    
    # Restore
    if hasattr(def_quali, 'OUTPUT_PDF_DIR'):
        def_quali.OUTPUT_PDF_DIR = original_pdf
    if hasattr(def_quali, 'OUTPUT_ODT_DIR'):
        def_quali.OUTPUT_ODT_DIR = original_odt

    return 0


# ============================================================================
# COMMANDE QUANT : Génération de datasets
# ============================================================================

@cli.command()
@click.option('--theme', '-t', 
              help='Le thème général du dataset')
@click.option('--lignes', '-l', type=int, 
              help='Nombre de lignes à générer')
@click.option('--interactive', '-i', is_flag=True, 
              help='Mode interactif (demande les infos)')
@click.option('--output-pdf', default='output_pdf', 
              help='Dossier de sortie pour le PDF')
@click.option('--output-odt', default='output_odt', 
              help='Dossier de sortie pour le ODT')
@click.option('--seed', type=int, default=42, 
              help='Seed pour la génération aléatoire')
def quant(theme, lignes, interactive, output_pdf, output_odt, seed):
    """
     Génère un dataset structuré en PDF et ODT
    
    Exemples :
    
      # Mode non-interactif
      synthetic-cli quant -t "Ventes 2024" -l 100
      
      # Mode interactif
      synthetic-cli quant --interactive
      
      # Avec seed personnalisé
      synthetic-cli quant -t "Clients" -l 50 --seed 123
    """
    
    click.echo("=" * 60)
    click.echo(" GÉNÉRATEUR DE DATASET (QUANT)")
    click.echo("=" * 60)
    
    # Mode interactif
    if interactive or not theme or lignes is None:
        theme = click.prompt('\n Entrer le thème général du dataset', type=str)
        lignes = click.prompt(' Entrer le nombre de lignes', type=int)
    
    # Validation
    if not theme or not theme.strip():
        click.echo(" Le thème ne peut pas être vide", err=True)
        return 1
    
    if lignes <= 0:
        click.echo(" Le nombre de lignes doit être > 0", err=True)
        return 1
    
    # Confirmation
    click.echo(f"\n✅ Configuration :")
    click.echo(f"   • Thème : {theme}")
    click.echo(f"   • Lignes : {lignes}")
    click.echo(f"   • Seed : {seed}")
    click.echo(f"   • Output PDF : {output_pdf}/")
    click.echo(f"   • Output ODT : {output_odt}/")
    
    if not click.confirm("\n Lancer la génération ?", default=True):
        click.echo(" Annulé")
        return 0
    
    # Exécution
    try:
        code = run_quant_generation(theme, lignes, output_pdf, output_odt, seed)
        
        if code == 0:
            click.echo(f"\n Dataset généré avec succès ! ")
            click.echo(f" Fichiers disponibles dans {output_pdf}/ et {output_odt}/")
        else:
            click.echo(f"\n  Génération terminée avec des erreurs", err=True)
        
        return code
        
    except KeyboardInterrupt:
        click.echo("\n\n Génération interrompue par l'utilisateur")
        return 130
    except Exception as e:
        click.echo(f"\n Erreur inattendue : {e}", err=True)
        return 1


def run_quant_generation(theme, nb_lignes, output_pdf, output_odt, seed):
    """Fonction de génération quant (non-interactive)"""
    
    click.echo(f"\n Configuration des colonnes...")
    colonnes = saisir_colonnes()
    
    click.echo("\n Génération du schéma JSON via LLM...")
    prompt = generer_prompt_llm(theme, colonnes, nb_lignes)
    raw = call_llm(prompt)
    spec_llm = parse_llm_spec(raw)

    if spec_llm is None:
        click.echo("\n Erreur: le LLM n'a pas renvoyé un JSON exploitable.", err=True)
        click.echo(" Réponse brute:", err=True)
        click.echo(raw, err=True)
        return 1

    click.echo("\n Schéma JSON généré :")
    click.echo(json.dumps(spec_llm, ensure_ascii=False, indent=2))

    click.echo(f"\n Génération du dataframe ({nb_lignes} lignes)...")
    df = generate_dataframe(spec_llm, nb_lignes, seed=seed)

    click.echo("\n Aperçu des données :")
    click.echo(df.head())
    click.echo(f"\n Dimensions : {df.shape}")
    click.echo(f" Types : {df.dtypes.to_dict()}")

    try:
        click.echo("\n Statistiques :")
        click.echo(df.describe(include="all"))
    except Exception:
        pass

    pdf_dir, odt_dir = creer_dossiers_sortie(output_pdf, output_odt)

    pdf_path = f"{pdf_dir}/dataset.pdf"
    odt_path = f"{odt_dir}/dataset.odt"

    click.echo(f"\nCréation des fichiers...")
    creer_pdf_table(df, pdf_path, titre=theme)
    creer_odt_table(df, odt_path, titre=theme, max_rows=200, zebra=True)
    
    click.echo(f" PDF : {pdf_path}")
    click.echo(f" ODT : {odt_path}")

    return 0



if __name__ == '__main__':
    cli()