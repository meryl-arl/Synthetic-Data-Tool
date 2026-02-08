"""
Module de génération de données numériques synthétiques.

Ce module fournit les fonctions pour :
- Saisir les colonnes d'un dataset
- Générer un schéma JSON via LLM
- Créer un DataFrame avec distributions statistiques
- Exporter les données en PDF, ODT et CSV
"""

from __future__ import annotations

import json
import re
import os
import numpy as np
import pandas as pd

from .deepinfra_client import make_deepinfra_client

# Bibliothèques pour générer des documents ODT 
from odf.opendocument import OpenDocumentText
from odf.text import P
from odf.style import Style, TextProperties, ParagraphProperties, TableCellProperties, TableRowProperties
from odf.table import Table as ODTTable, TableRow, TableCell

# Bibliothèques pour générer des PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors


# ============================================================================
# UTILITAIRES DE FORMATAGE
# ============================================================================

def to_safe_name(s: str) -> str:
    """
    Convertit un nom de colonne en identifiant Python valide.
    """
    s = s.strip().lower()
    # Remplace tout ce qui n'est pas alphanumérique par underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)
    # Supprime les underscores multiples et ceux en début/fin
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def format_column_title(s: str) -> str:
    """
    Formate un nom de colonne pour l'affichage (titre lisible).
    """
    s = s.strip()
    s = s.replace("_", " ")
    return s.capitalize()


# ============================================================================
# SAISIE INTERACTIVE DES COLONNES
# ============================================================================

def saisir_colonnes() -> list[str]:
    """
    Demande à l'utilisateur de saisir les noms des colonnes une par une.
    """
    colonnes = []
    seen = set()  # Pour détecter les doublons

    print("Entre les colonnes une par une (entrée vide = terminé).")

    while True:
        col_raw = input(">  ").strip()
        
        # Ligne vide = fin de saisie
        if col_raw == "":
            break

        # Normalisation du nom
        col = to_safe_name(col_raw)
        
        # Vérification de validité
        if not col:
            print("  (nom invalide, réessaiez)")
            continue

        # Détection des doublons
        if col in seen:
            print(f"  (déjà ajouté: {col})")
            continue

        colonnes.append(col)
        seen.add(col)

    return colonnes


# ============================================================================
# EXTRACTION ET APPEL LLM
# ============================================================================

def extract_json(text: str) -> str:
    """
    Extrait le JSON d'une réponse potentiellement bruitée du LLM.
    """
    if text is None:
        return ""

    t = text.strip()
    
    # Supprime les blocs markdown ```json et ```
    t = re.sub(r"^\s*```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    # Cherche le premier objet JSON {…}
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    return m.group(0) if m else t


def call_llm(prompt: str) -> str:
    """
    Appelle le LLM avec un prompt et retourne la réponse textuelle.
    """
    client = make_deepinfra_client()
    resp = client.chat.completions.create(
        model="anthropic/claude-3-7-sonnet-latest",
        # model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",  # Alternative
        temperature=0.2,
        frequency_penalty=0.8,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ============================================================================
# GÉNÉRATION DU SCHÉMA JSON VIA LLM
# ============================================================================

def generer_prompt_llm(theme: str, colonnes: list[str], nb_lignes: int) -> str:
    """
    Construit le prompt pour demander au LLM de générer un schéma statistique.
    """
    spec = {"theme": theme, "colonnes": colonnes, "nb_lignes": nb_lignes}

    return f"""
Tu es un expert senior en modélisation statistique et génération de données synthétiques.

Objectif :
À partir des spécifications suivantes, propose un schéma statistique réaliste pour générer un dataset synthétique.

Spécifications :
- Thème : {spec['theme']}
- Colonnes : {spec['colonnes']}
- Nombre de lignes : {spec['nb_lignes']}

Contraintes de sortie :
1. Réponds UNIQUEMENT en JSON valide.
2. AUCUN texte explicatif, AUCUN markdown, AUCUN commentaire.
3. Le JSON doit être directement parsable par Python (json.loads).
4. Utilise uniquement les types et distributions autorisés.
5. Si tu ne peux pas produire une réponse conforme ou si l'information est insuffisante, renvoie strictement: null

Schéma attendu :
{{
  "columns": {{
    "<nom_colonne>": {{
      "type": "integer|float|boolean|string|category",
      "distribution": "normal|lognormal|uniform|poisson|none",
      "params": {{
        "mean": float,
        "std": float,
        "min": float,
        "max": float,
        "lambda": float
      }},
      "categories": ["A", "B", "C"],
      "probas": [0.5, 0.3, 0.2]
    }}
  }},
  "row_rules": [
    {{
      "if": "<condition logique en pseudo-code>",
      "then": "<contrainte sur les colonnes>"
    }}
  ]
}}

Règles :
- Toutes les colonnes listées doivent apparaître dans "columns".
- Si type != "category", laisse "categories" et "probas" vides.
- Si distribution = "none", laisse "params" vide.
- Les probas doivent sommer à 1.
- Les règles doivent être réalistes et cohérentes avec le thème.
""".strip()


# ============================================================================
# PARSING DU SCHÉMA JSON
# ============================================================================

def parse_llm_spec(raw: str) -> dict | None:
    """
    Parse la réponse JSON du LLM en dictionnaire Python.
    """
    cleaned = extract_json(raw).strip()
    
    if cleaned in ("null", ""):
        return None

    try:
        # Tentative de parsing standard
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Tentative de correction
        cleaned2 = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned2)
        except json.JSONDecodeError:
            return None


# ============================================================================
# GÉNÉRATION DU DATAFRAME
# ============================================================================

def generate_dataframe(spec_llm: dict, nb_lignes: int, seed: int = 42) -> pd.DataFrame:
    """
    Génère un DataFrame pandas selon le schéma statistique fourni.
    """
    data = {}
    np.random.seed(seed) 

    columns = spec_llm.get("columns", {})
    
    for col_name, col_spec in columns.items():
        dist = col_spec.get("distribution", "none")
        params = col_spec.get("params", {}) or {}
        col_type = col_spec.get("type")


        match dist:
            case "normal":
                # Loi normale 
                mean = params.get("mean", 0)
                std = params.get("std", params.get("stddev", 1))
                arr = np.random.normal(loc=mean, scale=std, size=nb_lignes)

            case "lognormal":
                # Loi log-normale 
                mu = params.get("mu", 0)
                sigma = params.get("sigma", 1)
                arr = np.random.lognormal(mean=mu, sigma=sigma, size=nb_lignes)

            case "uniform":
                # Loi uniforme 
                low = params.get("low", params.get("min", 0))
                high = params.get("high", params.get("max", 1))
                arr = np.random.uniform(low=low, high=high, size=nb_lignes)

            case "poisson":
                # Loi de Poisson 
                lam = params.get("lambda", 1)
                arr = np.random.poisson(lam=lam, size=nb_lignes)

            case _:
                # Pas de distribution spécifiée
                if col_type == "category":
                    cats = col_spec.get("categories", [])
                    p = col_spec.get("probas", None)
                    if cats:
                        arr = np.random.choice(cats, size=nb_lignes, p=p)
                    else:
                        arr = np.array([""] * nb_lignes, dtype=object)

                elif col_type == "boolean" and col_spec.get("probas"):
                    # Boolean avec probabilités personnalisées
                    arr = np.random.choice([True, False], size=nb_lignes, p=col_spec["probas"])

                elif col_type == "boolean":
                    # Boolean équiprobable 
                    arr = np.random.choice([True, False], size=nb_lignes)

                elif col_type in ("integer", "float"):
                    # Fallback pour numériques sans distribution
                    arr = np.random.uniform(0, 1, size=nb_lignes)

                else:
                    # Fallback pour strings et autres
                    arr = np.array([""] * nb_lignes, dtype=object)

        # Arrondi des float à 2 décimales pour lisibilité
        if arr.dtype in [np.float64, np.float32]:
            arr = np.round(arr, 2)

        # Ajout de la colonne au dictionnaire
        data[col_name] = arr

    return pd.DataFrame(data)


# ============================================================================
# CRÉATION DES DOSSIERS DE SORTIE
# ============================================================================

def creer_dossiers_sortie(
    pdf_dir: str = "output_pdf", 
    odt_dir: str = "output_odt", 
    csv_dir: str = "output_csv"
) -> tuple[str, str, str]:
    """
    Crée les dossiers de sortie s'ils n'existent pas déjà.
    """
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(odt_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    return pdf_dir, odt_dir, csv_dir


# ============================================================================
# EXPORT PDF
# ============================================================================

def creer_pdf_table(
    df: pd.DataFrame, 
    nom_fichier: str, 
    titre: str | None = None, 
    max_rows: int = 50
):
    """
    Exporte un DataFrame en PDF avec mise en forme professionnelle
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
    style_normal = ParagraphStyle(
        "Normal", 
        parent=styles["Normal"], 
        fontSize=11, 
        alignment=TA_JUSTIFY
    )

    elements = []  

    if titre:
        elements.append(Paragraph(format_column_title(titre), style_titre))
        elements.append(Spacer(1, 0.4 * cm))  # Espace vertical

    elements.append(
        Paragraph(
            f"Taille: {df.shape[0]} lignes × {df.shape[1]} colonnes", 
            style_normal
        )
    )
    elements.append(Spacer(1, 0.4 * cm))

    df_view = df.head(max_rows).copy()
    
    header = [format_column_title(col) for col in df_view.columns]

    rows = []
    for _, row in df_view.iterrows():
        formatted_row = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                formatted_row.append(f"{val:.1f}")  
            else:
                formatted_row.append(str(val))
        rows.append(formatted_row)

    data_table = [header] + rows

    page_width = A4[0] - doc.leftMargin - doc.rightMargin
    ncols = max(len(header), 1)
    col_width = page_width / ncols
    col_widths = [col_width] * ncols

    table = Table(data_table, colWidths=col_widths, repeatRows=1)


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
        elements.append(
            Paragraph(
                f"(Aperçu limité aux {max_rows} premières lignes.)", 
                style_normal
            )
        )

    doc.build(elements)
    print(f"PDF créé: {nom_fichier}")


# ============================================================================
# EXPORT ODT 
# ============================================================================

def creer_odt_table(
    df: pd.DataFrame,
    nom_fichier: str,
    titre: str | None = None,
    max_rows: int = 200,
    zebra: bool = True,
):
    """
    Exporte un DataFrame en document ODT 
    """

    doc = OpenDocumentText()


    style_h1 = Style(name="Titre1", family="paragraph")
    style_h1.addElement(TextProperties(fontsize="18pt", fontweight="bold"))
    style_h1.addElement(ParagraphProperties(marginbottom="0.3cm"))
    doc.styles.addElement(style_h1)

    style_meta = Style(name="Meta", family="paragraph")
    style_meta.addElement(TextProperties(fontsize="11pt", color="#444444"))
    style_meta.addElement(ParagraphProperties(marginbottom="0.4cm"))
    doc.styles.addElement(style_meta)

    base_cell = Style(name="CellBase", family="table-cell")
    base_cell.addElement(
        TableCellProperties(border="0.03cm solid #999999", padding="0.12cm")
    )
    doc.styles.addElement(base_cell)

    header_cell = Style(name="CellHeader", family="table-cell", parentstylename=base_cell)
    header_cell.addElement(
        TableCellProperties(
            backgroundcolor="#E6E6E6", 
            border="0.03cm solid #666666", 
            padding="0.14cm"
        )
    )
    doc.styles.addElement(header_cell)

    zebra_cell = Style(name="CellZebra", family="table-cell", parentstylename=base_cell)
    zebra_cell.addElement(
        TableCellProperties(
            backgroundcolor="#F5F5F5", 
            border="0.03cm solid #999999", 
            padding="0.12cm"
        )
    )
    doc.styles.addElement(zebra_cell)

    p_cell = Style(name="PCell", family="paragraph")
    p_cell.addElement(TextProperties(fontsize="10.5pt"))
    doc.styles.addElement(p_cell)

    p_header = Style(name="PHeader", family="paragraph")
    p_header.addElement(TextProperties(fontsize="10.5pt", fontweight="bold"))
    doc.styles.addElement(p_header)

    row_style = Style(name="RowComfort", family="table-row")
    row_style.addElement(TableRowProperties(minrowheight="0.55cm"))
    doc.styles.addElement(row_style)

    if titre:
        doc.text.addElement(P(stylename=style_h1, text=format_column_title(titre)))


    doc.text.addElement(
        P(stylename=style_meta, text=f"Taille: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    )

    df_view = df.head(max_rows).copy()
    
    table = ODTTable(name="Tableau")

    tr_h = TableRow(stylename=row_style)
    for col in df_view.columns:
        tc = TableCell(stylename=header_cell)
        tc.addElement(P(stylename=p_header, text=format_column_title(col)))
        tr_h.addElement(tc)
    table.addElement(tr_h)

 
    for i, (_, row) in enumerate(df_view.iterrows()):
        tr = TableRow(stylename=row_style)
        
        use_zebra = zebra and (i % 2 == 1)
        cell_style = zebra_cell if use_zebra else base_cell

        for val in row.tolist():
            tc = TableCell(stylename=cell_style)

            val_str = f"{val:.1f}" if isinstance(val, (float, np.floating)) else str(val)
            tc.addElement(P(stylename=p_cell, text=val_str))
            tr.addElement(tc)

        table.addElement(tr)


    doc.text.addElement(table)

 
    if df.shape[0] > max_rows:
        doc.text.addElement(
            P(stylename=style_meta, text=f"(Aperçu limité aux {max_rows} premières lignes.)")
        )

   
    doc.save(nom_fichier)
    print(f"ODT créé : {nom_fichier}")


# ============================================================================
# EXPORT CSV
# ============================================================================

def creer_csv_table(
    df: pd.DataFrame,
    nom_fichier: str,
    titre: str | None = None,
    max_rows: int | None = None,
    sep: str = ";",
    decimal: str = ",",
    encoding: str = "utf-8-sig",  
    index: bool = False,
    float_format: str = "%.1f",
) -> None:
    """
    Exporte un DataFrame en fichier CSV 
    """
   
    dossier = os.path.dirname(nom_fichier)
    if dossier:
        os.makedirs(dossier, exist_ok=True)

    df_view = df.head(max_rows).copy() if (max_rows is not None) else df.copy()

    with open(nom_fichier, "w", encoding=encoding, newline="") as f:
     
        if titre:
            f.write(f"# {titre}\n")
            f.write(f"# Taille: {df.shape[0]} lignes × {df.shape[1]} colonnes\n")
            if max_rows is not None and df.shape[0] > max_rows:
                f.write(f"# (Aperçu limité aux {max_rows} premières lignes.)\n")
        
    
        df_view.to_csv(
            f,
            header=True,              
            index=index,              
            sep=sep,                 
            decimal=decimal,         
            lineterminator="\n",    
            float_format=float_format, 
        )

    print(f"CSV créé : {nom_fichier}")


# ============================================================================
# EXPORT MULTI-FORMATS
# ============================================================================

def write_files_df(
    df: pd.DataFrame, 
    index: int, 
    theme: str, 
    pdf_dir="output_pdf", 
    odt_dir="output_odt"
):
    """
    Exporte un DataFrame en PDF et ODT simultanément.
    """
   
    pdf_path = os.path.join(pdf_dir, f"table_{index}.pdf")
    odt_path = os.path.join(odt_dir, f"table_{index}.odt")


    creer_pdf_table(df, pdf_path, titre=f"{theme} — Table {index}", max_rows=50)
    creer_odt_table(df, odt_path, titre=f"{theme} — Table {index}", max_rows=200, zebra=True)
    
    return index


# ============================================================================
# SÉLECTION INTERACTIVE DES FORMATS
# ============================================================================

def choisir_formats() -> dict[str, bool]:
    """
    Demande à l'utilisateur quels formats de sortie il souhaite générer.
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