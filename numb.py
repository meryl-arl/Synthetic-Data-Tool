import json
import re
from deepinfra_client import make_deepinfra_client


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


theme = input("Entrer le thème général du dataset : ")
check_non_vide(theme, "Thème")

lignes_p = input("Entrer les noms des lignes : ")
check_non_vide(lignes_p, "Noms des lignes")

colonne_p = input("Entrer les noms des colonnes : ")
check_non_vide(colonne_p, "Noms des colonnes")

lignes = sans_doublons(lignes_p)
colonnes = sans_doublons(colonne_p)


# ----------------------------
# PROMPT
# ----------------------------

spec = {
    "theme": theme,
    "colonnes": colonnes,
    "nb_lignes": len(lignes)
}

prompt = f"""
Tu es un expert en modélisation statistique.

Thème: {spec['theme']}
Colonnes: {spec['colonnes']}

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
# JSON CLEANER
# -------------------------

def extract_json(text: str) -> str:
    """Récupère le premier JSON valide trouvé"""
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
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


# -------------------------
# RUN
# -------------------------

raw = call_llm(prompt)
clean_json = extract_json(raw)
spec_llm = json.loads(clean_json)

print(json.dumps(spec_llm, ensure_ascii=False, indent=2))
