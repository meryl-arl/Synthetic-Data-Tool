
import random 
import re     
import os      
from concurrent.futures import ThreadPoolExecutor, as_completed  

import polars as pl 
from tqdm import tqdm  


from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY

from odf.opendocument import OpenDocumentText
from odf.text import P
from odf.style import Style, TextProperties, ParagraphProperties 

from .deepinfra_client import make_deepinfra_client  


def creer_pdf(texte: str, nom_fichier: str):
    """
    Crée un PDF à partir d'un texte brut.
    """

    doc = SimpleDocTemplate(
        nom_fichier,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )


    styles = getSampleStyleSheet()

    style_titre = ParagraphStyle("Titre", parent=styles["Heading1"], fontSize=18)
    style_h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14)
    style_h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=12)
    style_normal = ParagraphStyle(
        "Normal", parent=styles["Normal"], fontSize=11, alignment=TA_JUSTIFY
    )

    elements = []

    lignes = texte.strip().split("\n")

    for ligne in lignes:
        ligne = ligne.strip()

        if not ligne:
            elements.append(Spacer(1, 0.3 * cm))
            continue

        ligne_clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ligne)

        if ligne.startswith("### "):
            elements.append(Paragraph(ligne_clean.replace("### ", ""), style_h3))
        elif ligne.startswith("## "):
            elements.append(Paragraph(ligne_clean.replace("## ", ""), style_h2))
        elif ligne.startswith("# "):
            elements.append(Paragraph(ligne_clean.replace("# ", ""), style_titre))
        else:
            elements.append(Paragraph(ligne_clean, style_normal))

    doc.build(elements)
    print(f"PDF créé: {nom_fichier}")



def creer_odt(texte: str, nom_fichier: str):
    """
    Crée un fichier ODT à partir d'un texte brut.
    """

    doc = OpenDocumentText()

    style_h1 = Style(name="Titre1", family="paragraph")
    style_h1.addElement(TextProperties(fontsize="18pt", fontweight="bold"))
    doc.styles.addElement(style_h1)

    style_normal = Style(name="Normal", family="paragraph")
    style_normal.addElement(TextProperties(fontsize="11pt"))
    doc.styles.addElement(style_normal)

    lignes = texte.strip().split("\n")

    for ligne in lignes:
        ligne = ligne.strip()

        if not ligne:
            doc.text.addElement(P(text=""))
            continue

        ligne_clean = re.sub(r"\*\*(.+?)\*\*", r"\1", ligne)

        if ligne.startswith("# "):
            doc.text.addElement(P(stylename=style_h1, text=ligne_clean.replace("# ", "")))
        else:
            doc.text.addElement(P(stylename=style_normal, text=ligne_clean))

    doc.save(nom_fichier)
    print(f"ODT créé: {nom_fichier}")


def creer_csv(texte: str, nom_fichier: str):
    """
    Crée un fichier CSV contenant le texte.
    """

    dossier = os.path.dirname(nom_fichier)
    if dossier:
        os.makedirs(dossier, exist_ok=True)

    texte_clean = texte.replace('"', '""').replace('\n', ' ')

    with open(nom_fichier, 'w', encoding='utf-8-sig') as f:
        f.write("contenu\n")
        f.write(f'"{texte_clean}"\n')

    print(f"CSV créé: {nom_fichier}")


def write_files(args):
    """
    Version qui accepte les formats choisis.
    """
    texte_llm, index, formats = args

    if formats['pdf']:
        nom_pdf = f"output_pdf/article_{index+1}.pdf"
        creer_pdf(texte_llm, nom_pdf)

    if formats['odt']:
        nom_odt = f"output_odt/article_{index+1}.odt"
        creer_odt(texte_llm, nom_odt)

    if formats['csv']:
        nom_csv = f"output_csv/article_{index+1}.csv"
        creer_csv(texte_llm, nom_csv)

    return index


# ============================================================
# Inference LLM 
# ============================================================

def inference(client, sujet):
    """
    Appelle le modèle LLM pour générer un texte sur un sujet donné.
    """

    prompt = (
        "Crée un texte de plusieurs paragraphes sur le thème de la technologie "
        "et de l'innovation en 2024, parle de ce sujet spécifique : "
        f"{sujet}"
    )

    response = client.chat.completions.create(
        temperature=0.8,
        frequency_penalty=0.9,
        model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        messages=[{"role": "user", "content": prompt}],
    )

   
    return response.choices[0].message.content


# ============================================================
# Choix formats 
# ============================================================

def choisir_formats() -> dict[str, bool]:
    """
    Demande à l'utilisateur quels formats il souhaite générer.
    """

    print("\n=== Choix des formats de sortie ===")
    print("Quels formats voulez-vous générer ?")
    print("(Répondez par 'o' pour oui, 'n' pour non, ou appuyez sur Entrée pour tout générer)")

    formats = {}

    reponse_pdf = input("  PDF ? [o/N] : ").strip().lower()
    formats['pdf'] = reponse_pdf == 'o'

    reponse_odt = input("  ODT ? [o/N] : ").strip().lower()
    formats['odt'] = reponse_odt == 'o'

    reponse_csv = input("  CSV ? [o/N] : ").strip().lower()
    formats['csv'] = reponse_csv == 'o'

    if not any(formats.values()):
        print("\nAucun format sélectionné. Génération de tous les formats par défaut.")
        formats = {'pdf': True, 'odt': True, 'csv': True}

    return formats
