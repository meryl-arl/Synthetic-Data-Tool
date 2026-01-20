NUM_DOCUMENTS = 1000
random_techrandom_tech = ["AI","Telsa","blockchain","robotique","impression 3D","cybersÃ©curitÃ©","espace","Ã©nergie durable","agriculture","Ã©ducation","communication","recherche scientifique","audio","juridique","marketing","ressources humaines","service client","bases de donnÃ©es"] 
PROMPT = f"CrÃ©e un texte de plusieurs paragraphe sur le thÃ¨me de la technologie et de l'innovation en 2024, parle de ce sujet spÃ©cifique : {random.choice(random_techrandom_tech)}

import random
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from odf.opendocument import OpenDocumentText
from odf.text import P
from openai import OpenAI

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY

from odf.opendocument import OpenDocumentText
from odf.text import P, H
from odf.style import Style, TextProperties, ParagraphProperties
import polars as pl
import re


def creer_pdf(texte: str, nom_fichier: str):
    """
    CrÃ©e un fichier PDF Ã  partir d'une string.
    GÃ¨re les titres (###, ##, #) et les paragraphes.
    """
    doc = SimpleDocTemplate(
        nom_fichier,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    style_titre = ParagraphStyle(
        'Titre',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=10
    )
    
    style_h2 = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=15
    )
    
    style_h3 = ParagraphStyle(
        'H3',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=10,
        spaceBefore=12
    )
    
    style_normal = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    )
    
    elements = []
    
    # Nettoyer et parser le texte
    lignes = texte.strip().split('\n')
    
    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            elements.append(Spacer(1, 0.3*cm))
            continue
        
        # Nettoyer le markdown bold *texte*
        ligne_clean = re.sub(r'\\(.+?)\\', r'<b>\1</b>', ligne)
        
        # DÃ©tecter les niveaux de titre
        if ligne.startswith('### '):
            titre = ligne_clean.replace('### ', '')
            elements.append(Paragraph(titre, style_h3))
        elif ligne.startswith('## '):
            titre = ligne_clean.replace('## ', '')
            elements.append(Paragraph(titre, style_h2))
        elif ligne.startswith('# '):
            titre = ligne_clean.replace('# ', '')
            elements.append(Paragraph(titre, style_titre))
        else:
            # Paragraphe normal
            elements.append(Paragraph(ligne_clean, style_normal))
    
    doc.build(elements)
    print(f"PDF crÃ©Ã©: {nom_fichier}")


def creer_odt(texte: str, nom_fichier: str):
    """
    CrÃ©e un fichier ODT Ã  partir d'une string.
    GÃ¨re les titres (###, ##, #) et les paragraphes.
    """
    doc = OpenDocumentText()
    
    # CrÃ©er les styles
    # Style titre principal
    style_h1 = Style(name="Titre1", family="paragraph")
    style_h1.addElement(TextProperties(fontsize="18pt", fontweight="bold"))
    style_h1.addElement(ParagraphProperties(marginbottom="0.5cm", margintop="0.3cm"))
    doc.styles.addElement(style_h1)
    
    # Style H2
    style_h2 = Style(name="Titre2", family="paragraph")
    style_h2.addElement(TextProperties(fontsize="14pt", fontweight="bold"))
    style_h2.addElement(ParagraphProperties(marginbottom="0.4cm", margintop="0.5cm"))
    doc.styles.addElement(style_h2)
    
    # Style H3
    style_h3 = Style(name="Titre3", family="paragraph")
    style_h3.addElement(TextProperties(fontsize="12pt", fontweight="bold"))
    style_h3.addElement(ParagraphProperties(marginbottom="0.3cm", margintop="0.4cm"))
    doc.styles.addElement(style_h3)
    
    # Style normal
    style_normal = Style(name="Normal", family="paragraph")
    style_normal.addElement(TextProperties(fontsize="11pt"))
    style_normal.addElement(ParagraphProperties(marginbottom="0.3cm", textalign="justify"))
    doc.styles.addElement(style_normal)
    
    # Parser le texte
    lignes = texte.strip().split('\n')
    
    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            doc.text.addElement(P(text=""))
            continue
        
        # Nettoyer le markdown bold
        ligne_clean = re.sub(r'\\(.+?)\\', r'\1', ligne)
        
        # DÃ©tecter les niveaux de titre
        if ligne.startswith('### '):
            titre = ligne_clean.replace('### ', '')
            p = P(stylename=style_h3, text=titre)
            doc.text.addElement(p)
        elif ligne.startswith('## '):
            titre = ligne_clean.replace('## ', '')
            p = P(stylename=style_h2, text=titre)
            doc.text.addElement(p)
        elif ligne.startswith('# '):
            titre = ligne_clean.replace('# ', '')
            p = P(stylename=style_h1, text=titre)
            doc.text.addElement(p)
        else:
            p = P(stylename=style_normal, text=ligne_clean)
            doc.text.addElement(p)
    
    doc.save(nom_fichier)
    print(f"ODT crÃ©Ã©: {nom_fichier}")


client = OpenAI(api_key="7jIPsm1yv398SZpzLaE0qw2DIs2Y5CZG",base_url="https://api.deepinfra.com/v1/openai")

data = pl.DataFrame(schema={"content": pl.Utf8})

def write_files(args):
    texte_llm, index = args
    creer_pdf(texte_llm, f"output_pdf/article_{index}.pdf")
    creer_odt(texte_llm, f"output_odt/article_{index}.odt")
    return index

def inference(client, new_technologies):
    random_techrandom_tech = random.choice(new_technologies)
    prompt=PROMPT
    response = client.chat.completions.create(
        temperature=0.8,
        frequency_penalty=0.9,
        model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from tqdm import tqdm
from concurrent.futures import as_completed


with ThreadPoolExecutor(max_workers=min(50, NUM_DOCUMENTS)) as executor:
    futures = [executor.submit(inference, client, TECHNOLOGIES_LIST) for _ in range(NUM_DOCUMENTS)]
    results = []
    for f in tqdm(as_completed(futures), total=NUM_DOCUMENTS, desc="InfÃ©rences"):
        results.append(f.result())

for i in range(len(results)):
    write_files((results[i], i))