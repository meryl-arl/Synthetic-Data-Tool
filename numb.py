import json
import re
from deepinfra_client import make_deepinfra_client
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY

from odf.opendocument import OpenDocumentText
from odf.text import P
from odf.style import Style, TextProperties, ParagraphProperties


# ---------------------------
# COLLECTE DE BESOINS
# ---------------------------
#TODO MIEUX BLINDER PCQ CEST QUOI CA , et faire en sorte que on puisse faire des mots avec des espaces 
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
check_non_vide(theme, "Thème")

nb_lignes = 200

#int(input("Entrer les nombre des lignes : "))
#TODO OUBLIE PAS DE RAJOUTER CA HEIN QUAND TAS FINI DE TOUT TESTER 


colonnes = input("Entrer les noms des colonnes : ")
check_non_vide(colonnes, "Noms des colonnes")


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

