"""
Générateur de PDF et ODT avec l'IA : il génère automatiquement des articles grâce aux LLM.
"""

import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from tqdm import tqdm

from deepinfra_client import make_deepinfra_client

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from odf.opendocument import OpenDocumentText
from odf.style import ParagraphProperties, Style, TextProperties
from odf.text import P


# --------------------
# Configuration
# --------------------
NUM_DOCUMENTS = 1  # debug: commence petit (1), puis augmente
TECHNOLOGIES = [
    "AI", "Tesla", "blockchain", "robotique", "impression 3D", "cybersécurité",
    "espace", "énergie durable", "agriculture", "éducation", "communication",
    "recherche scientifique", "audio", "juridique", "marketing",
    "ressources humaines", "service client", "bases de données"
]


def build_prompt(tech: str) -> str:
    return (
        "Crée un texte de plusieurs paragraphes sur le thème de la technologie et de l'innovation en 2024, "
        f"parle de ce sujet spécifique : {tech}\n"
        "Ajoute un titre (# ...) et deux sous-titres (## ...)."
    ) #demander si ca cest bien ou l'encient n'etait pas mieu (ici prompt different, avant prompt identique a chaque fois)


# --------------------
# Création de documents
# --------------------
def creer_pdf(texte: str, nom_fichier: str):
    """
    Crée un fichier PDF à partir d'un texte en Markdown.
    Gère les titres (###, ##, #) et les paragraphes.
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

    style_titre = ParagraphStyle(
        "Titre",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=10,
    )

    style_h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=15,
    )

    style_h3 = ParagraphStyle(
        "H3",
        parent=styles["Heading3"],
        fontSize=12,
        spaceAfter=10,
        spaceBefore=12,
    )

    style_normal = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
    )

    elements = []

    lignes = texte.strip().split("\n")

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            elements.append(Spacer(1, 0.3 * cm))
            continue

        # Markdown bold **texte** -> <b>texte</b>
        ligne_clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ligne)

        if ligne.startswith("### "):
            titre = ligne_clean.replace("### ", "")
            elements.append(Paragraph(titre, style_h3))
        elif ligne.startswith("## "):
            titre = ligne_clean.replace("## ", "")
            elements.append(Paragraph(titre, style_h2))
        elif ligne.startswith("# "):
            titre = ligne_clean.replace("# ", "")
            elements.append(Paragraph(titre, style_titre))
        else:
            elements.append(Paragraph(ligne_clean, style_normal))

    doc.build(elements)
    print(f"PDF créé: {nom_fichier}")


def creer_odt(texte: str, nom_fichier: str):
    """
    Crée un fichier ODT à partir d'une string.
    Gère les titres (###, ##, #) et les paragraphes.
    """
    doc = OpenDocumentText()

    style_h1 = Style(name="Titre1", family="paragraph")
    style_h1.addElement(TextProperties(fontsize="18pt", fontweight="bold"))
    style_h1.addElement(ParagraphProperties(marginbottom="0.5cm", margintop="0.3cm"))
    doc.styles.addElement(style_h1)

    style_h2 = Style(name="Titre2", family="paragraph")
    style_h2.addElement(TextProperties(fontsize="14pt", fontweight="bold"))
    style_h2.addElement(ParagraphProperties(marginbottom="0.4cm", margintop="0.5cm"))
    doc.styles.addElement(style_h2)

    style_h3 = Style(name="Titre3", family="paragraph")
    style_h3.addElement(TextProperties(fontsize="12pt", fontweight="bold"))
    style_h3.addElement(ParagraphProperties(marginbottom="0.3cm", margintop="0.4cm"))
    doc.styles.addElement(style_h3)

    style_normal = Style(name="Normal", family="paragraph")
    style_normal.addElement(TextProperties(fontsize="11pt"))
    style_normal.addElement(ParagraphProperties(marginbottom="0.3cm", textalign="justify"))
    doc.styles.addElement(style_normal)

    lignes = texte.strip().split("\n")

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            doc.text.addElement(P(text=""))
            continue

        # on retire juste le markdown bold **...**
        ligne_clean = re.sub(r"\*\*(.+?)\*\*", r"\1", ligne)

        if ligne.startswith("### "):
            titre = ligne_clean.replace("### ", "")
            doc.text.addElement(P(stylename=style_h3, text=titre))
        elif ligne.startswith("## "):
            titre = ligne_clean.replace("## ", "")
            doc.text.addElement(P(stylename=style_h2, text=titre))
        elif ligne.startswith("# "):
            titre = ligne_clean.replace("# ", "")
            doc.text.addElement(P(stylename=style_h1, text=titre))
        else:
            doc.text.addElement(P(stylename=style_normal, text=ligne_clean))

    doc.save(nom_fichier)
    print(f"ODT créé: {nom_fichier}")


# --------------------
# LLM + pipeline
# --------------------
client = make_deepinfra_client()
data = pl.DataFrame(schema={"content": pl.Utf8})


def write_files(texte_llm: str, index: int):
    creer_pdf(texte_llm, f"output_pdf/article_{index}.pdf")
    creer_odt(texte_llm, f"output_odt/article_{index}.odt")
    return index


def inference(client, technologies) -> str:
    tech = random.choice(technologies)
    prompt = build_prompt(tech)

    response = client.chat.completions.create(
        temperature=0.8,
        frequency_penalty=0.9,
        model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    # Dossiers de sortie (sinon crash si inexistants)
    os.makedirs("output_pdf", exist_ok=True)
    os.makedirs("output_odt", exist_ok=True)

    with ThreadPoolExecutor(max_workers=min(50, NUM_DOCUMENTS)) as executor:
        futures = [executor.submit(inference, client, TECHNOLOGIES) for _ in range(NUM_DOCUMENTS)]
        results = []

        for f in tqdm(as_completed(futures), total=NUM_DOCUMENTS, desc="Inférences"):
            results.append(f.result())

    for i, txt in enumerate(results):
        write_files(txt, i)


if __name__ == "__main__":
    main()
