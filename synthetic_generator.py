"""
Générateur de pdf et odt avec l'ia
Il génère automatiquement des articles grâce aux llm
"""

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

from deepinfra_client import make_deepinfra_client


# --------------------
# CONFIGURATION 
# --------------------
NUM_DOCUMENTS = 1

random_techrandom_tech = [
    "AI", "Tesla", "blockchain", "robotique", "impression 3D",
    "cybersécurité", "espace", "énergie durable", "agriculture",
    "éducation", "communication", "recherche scientifique",
    "audio", "juridique", "marketing", "ressources humaines",
    "service client", "bases de données"
]

# TODO: change prompt
PROMPT = (
    "Crée un texte de plusieurs paragraphes sur le thème de la technologie "
    "et de l'innovation en 2024, parle de ce sujet spécifique : "
    f"{random.choice(random_techrandom_tech)}"  #GROSSE KARBA CHANGE MOI CA TOUT DE SUITE ET PLSU VITE QUE CA 
    #OUBLIE PAS MV STP REINE 
)


# --------------------
# FONCTIONS DE CRÉATION
# --------------------
def creer_pdf(texte: str, nom_fichier: str):
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


# --------------------
# LLM 
# --------------------
client = make_deepinfra_client()
data = pl.DataFrame(schema={"content": pl.Utf8})


def write_files(args):
    texte_llm, index = args
    creer_pdf(texte_llm, f"output_pdf/article_{index}.pdf")
    creer_odt(texte_llm, f"output_odt/article_{index}.odt")
    return index


def inference(client, new_technologies):
   
    tech = random.choice(new_technologies)
    prompt = (
        "Crée un texte de plusieurs paragraphes sur le thème de la technologie "
        "et de l'innovation en 2024, parle de ce sujet spécifique : "
        f"{tech}"
    )

    response = client.chat.completions.create(
        temperature=0.8,
        frequency_penalty=0.9,
        model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


# --------------------
# EXÉCUTION
# --------------------
os.makedirs("output_pdf", exist_ok=True)
os.makedirs("output_odt", exist_ok=True)


with ThreadPoolExecutor(max_workers=min(NUM_DOCUMENTS, 50)) as executor:
    futures = [executor.submit(inference, client, random_techrandom_tech) for _ in range(NUM_DOCUMENTS)]
    results = []

    for f in tqdm(as_completed(futures), total=NUM_DOCUMENTS, desc="Inférences"):
        txt = f.result()
        results.append(txt)


for i in range(len(results)):
    write_files((results[i], i))
