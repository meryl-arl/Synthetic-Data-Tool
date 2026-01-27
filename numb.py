import json
import re
from deepinfra_client import make_deepinfra_client
import numpy as np
import pandas as pd

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors

from odf.opendocument import OpenDocumentText
from odf.text import P
from odf.style import Style, TextProperties
from odf.table import Table as ODTTable, TableRow, TableCell

"""
TODO : ameliorer a generation de pdf et odt: 
- enlever les lignes max et min,
- pour le titre des collones retirer les espacas, 
- pour la formation du titre mettre une majuscule 
- rajouter des chiffres signoficatif 
"""

# ---------------------------
# COLLECTE DE BESOINS
# ---------------------------
def sans_doublons(texte: str) -> list[str]:
    mots = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?", texte.lower())
    seen = set()
    out = []
    for m in mots:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def check_non_vide(valeur: str, nom: str = "entrée") -> None:
    if valeur is None or not valeur.strip():
        raise ValueError(f"{nom} vide.")


theme = input("Entrer le thème général du dataset (le titre)) : ")


nb_lignes = int(input("Entrer le nombre de lignes : "))


def to_safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def saisir_colonnes() -> list[str]:
    colonnes = []
    seen = set()

    print("Entre les colonnes une par une (entrée vide = terminé).")
    
    while True:
        col_raw = input(">  ").strip()
        if col_raw == "":
            break

        col = to_safe_name(col_raw)
        if not col:
            print("  (nom invalide, réessaiez)")
            continue

        if col in seen:
            print(f"  (déjà ajouté: {col})")
            continue

        colonnes.append(col)
        seen.add(col)
        

    return colonnes


colonnes = saisir_colonnes()
print("generalisation du json en cours (peut prendre 30s à 1 min)")



# ----------------------------
# PROMPT
# ----------------------------

spec = {
    "theme": theme,
    "colonnes": colonnes,
    "nb_lignes": nb_lignes
}

prompt = f"""
Tu es un expert en modélisation statistique.

Thème: {spec['theme']}
Colonnes: {spec['colonnes']}
Nombre de lignes : {spec['nb_lignes']}

Réponds UNIQUEMENT en JSON valide (sans texte, sans markdown).

Format:
{{
  "columns": {{
    "colonne": {{
      "type": "integer|float|boolean|string|category",
      "distribution": "normal|lognormal|uniform|poisson|none",
      "params": {{}},
      "categories": [],
      "probas": []
    }}
  }},
  "row_rules": []
}}
""".strip()


# -------------------------
# nettoyer le json
# -------------------------

def extract_json(text: str) -> str:
    if text is None:
        return ""

    t = text.strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    # cherche premier bloc JSON
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    return m.group(0) if m else t


# -------------------------
# LLM
# -------------------------

model = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
client = make_deepinfra_client()


def call_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        frequency_penalty=0.2,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content



raw = call_llm(prompt)
clean_json = extract_json(raw)
spec_llm = json.loads(clean_json)

print(json.dumps(spec_llm, ensure_ascii=False, indent=2))


#----------------------------
#CONVERSION DES LOI DE PROBA
#----------------------------

data = {}

np.random.seed(42)

columns = spec_llm["columns"]
for col_name, col_spec in columns.items():
    dist = col_spec.get("distribution", "none")
    params = col_spec.get("params", {}) or {}
    col_type = col_spec.get("type")

    match dist:
        case "normal":
            mean = params.get("mean", 0)
            std = params.get("std", params.get("stddev", 1))
            arr = np.random.normal(loc=mean, scale=std, size=nb_lignes)

        case "lognormal":
            mu = params.get("mu", 0)
            sigma = params.get("sigma", 1)
            arr = np.random.lognormal(mean=mu, sigma=sigma, size=nb_lignes)

        case "uniform":
            low = params.get("low", params.get("min", 0))
            high = params.get("high", params.get("max", 1))
            arr = np.random.uniform(low=low, high=high, size=nb_lignes)

        case "poisson":
            lam = params.get("lambda", 1)
            arr = np.random.poisson(lam=lam, size=nb_lignes)

        case _:
            if col_type == "category":
                cats = col_spec["categories"]
                p = col_spec["probas"]
                arr = np.random.choice(cats, size=nb_lignes, p=p)

            elif col_type == "boolean" and col_spec.get("probas"):
                arr = np.random.choice([True, False], size=nb_lignes, p=col_spec["probas"])

            elif col_type == "boolean":
                arr = np.random.choice([True, False], size=nb_lignes)

            else:
                arr = np.array([""] * nb_lignes, dtype=object)

    data[col_name] = arr

df = pd.DataFrame(data)
print(df.head())


#test

print(df.shape)
print(df.dtypes)
print(df.describe(include="all"))


#--------------------------------
#METTRE AU FORMAT PDF ET ODT
#--------------------------------


def creer_pdf_table(
    df: pd.DataFrame,
    nom_fichier: str,
    titre: str | None = None,
    max_rows: int = 50,
):
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
    style_normal = ParagraphStyle("Normal", parent=styles["Normal"], fontSize=11, alignment=TA_JUSTIFY)

    elements = []

    # Titre + métadonnées
    if titre:
        elements.append(Paragraph(titre, style_titre))
        elements.append(Spacer(1, 0.4 * cm))

    elements.append(Paragraph(f"Taille: {df.shape[0]} lignes × {df.shape[1]} colonnes", style_normal))
    elements.append(Spacer(1, 0.4 * cm))

    # Limiter lignes
    df_view = df.head(max_rows).copy()

    # Convertir en matrice pour ReportLab
    header = list(df_view.columns)
    rows = df_view.astype(str).values.tolist()
    data = [header] + rows

    # Largeurs de colonnes: réparties sur la page (simple)
    page_width = A4[0] - doc.leftMargin - doc.rightMargin
    ncols = max(len(header), 1)
    col_width = page_width / ncols
    col_widths = [col_width] * ncols

    table = Table(data, colWidths=col_widths, repeatRows=1)

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    elements.append(table)

    if df.shape[0] > max_rows:
        elements.append(Spacer(1, 0.4 * cm))
        elements.append(Paragraph(f"(Aperçu limité aux {max_rows} premières lignes.)", style_normal))

    doc.build(elements)
    print(f"PDF créé: {nom_fichier}")

import pandas as pd
from odf.opendocument import OpenDocumentText
from odf.text import P
from odf.style import Style, TextProperties, ParagraphProperties, TableCellProperties, TableRowProperties
from odf.table import Table as ODTTable, TableRow, TableCell


def creer_odt_table(
    df: pd.DataFrame,
    nom_fichier: str,
    titre: str | None = None,
    max_rows: int = 200,
    zebra: bool = True,
):
    doc = OpenDocumentText()

    # -----------------------
    # Styles (texte)
    # -----------------------
    style_h1 = Style(name="Titre1", family="paragraph")
    style_h1.addElement(TextProperties(fontsize="18pt", fontweight="bold"))
    style_h1.addElement(ParagraphProperties(marginbottom="0.3cm"))
    doc.styles.addElement(style_h1)

    style_meta = Style(name="Meta", family="paragraph")
    style_meta.addElement(TextProperties(fontsize="11pt", color="#444444"))
    style_meta.addElement(ParagraphProperties(marginbottom="0.4cm"))
    doc.styles.addElement(style_meta)

    # -----------------------
    # Styles (table)
    # -----------------------
    # Bordures + padding (lisibilité ++)
    base_cell = Style(name="CellBase", family="table-cell")
    base_cell.addElement(
        TableCellProperties(
            border="0.03cm solid #999999",
            padding="0.12cm",
        )
    )
    doc.styles.addElement(base_cell)

    header_cell = Style(name="CellHeader", family="table-cell", parentstylename=base_cell)
    header_cell.addElement(
        TableCellProperties(
            backgroundcolor="#E6E6E6",
            border="0.03cm solid #666666",
            padding="0.14cm",
        )
    )
    doc.styles.addElement(header_cell)

    zebra_cell = Style(name="CellZebra", family="table-cell", parentstylename=base_cell)
    zebra_cell.addElement(
        TableCellProperties(
            backgroundcolor="#F5F5F5",
            border="0.03cm solid #999999",
            padding="0.12cm",
        )
    )
    doc.styles.addElement(zebra_cell)

    # Style paragraphe dans les cellules
    p_cell = Style(name="PCell", family="paragraph")
    p_cell.addElement(TextProperties(fontsize="10.5pt"))
    doc.styles.addElement(p_cell)

    p_header = Style(name="PHeader", family="paragraph")
    p_header.addElement(TextProperties(fontsize="10.5pt", fontweight="bold"))
    doc.styles.addElement(p_header)

    # Hauteur de ligne (évite la compression)
    row_style = Style(name="RowComfort", family="table-row")
    row_style.addElement(TableRowProperties(minrowheight="0.55cm"))
    doc.styles.addElement(row_style)

    # -----------------------
    # Contenu
    # -----------------------
    if titre:
        doc.text.addElement(P(stylename=style_h1, text=titre))

    doc.text.addElement(
        P(stylename=style_meta, text=f"Taille: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    )

    df_view = df.head(max_rows).copy()

    table = ODTTable(name="Tableau")

    # Header
    tr_h = TableRow(stylename=row_style)
    for col in df_view.columns:
        tc = TableCell(stylename=header_cell)
        tc.addElement(P(stylename=p_header, text=str(col)))
        tr_h.addElement(tc)
    table.addElement(tr_h)

    # Rows
    for i, (_, row) in enumerate(df_view.iterrows()):
        tr = TableRow(stylename=row_style)

        use_zebra = zebra and (i % 2 == 1)
        cell_style = zebra_cell if use_zebra else base_cell

        for val in row.tolist():
            tc = TableCell(stylename=cell_style)
            tc.addElement(P(stylename=p_cell, text=str(val)))
            tr.addElement(tc)

        table.addElement(tr)

    doc.text.addElement(table)

    if df.shape[0] > max_rows:
        doc.text.addElement(
            P(stylename=style_meta, text=f"(Aperçu limité aux {max_rows} premières lignes.)")
        )

    doc.save(nom_fichier)
    print(f"ODT créé : {nom_fichier}")



def write_files_df(args):
    df, index = args
    creer_pdf_table(df, f"output_pdf/table_{index}.pdf", titre=f"Table {index}", max_rows=50)
    creer_odt_table(df, "output_odt/test.odt", titre=theme, max_rows=200, zebra=True)
    return index


os.makedirs("output_pdf", exist_ok=True)
os.makedirs("output_odt", exist_ok=True)

creer_pdf_table(df, "output_pdf/test.pdf", titre=theme)
creer_odt_table(df, "output_odt/test.odt", titre=theme, max_rows=200, zebra=True)


"""
inscription eleves
age nationalite classe moyenne absence
"""

