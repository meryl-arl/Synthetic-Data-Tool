import json
import re
import numpy as np
import pandas as pd



from fonctions.def_n import (
    saisir_colonnes,
    call_llm,
    creer_pdf_table,
    creer_odt_table,
    creer_csv_table,
    creer_dossiers_sortie,
    choisir_formats,
)



def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _truncate_words(s: str, max_words: int | None):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    if not max_words or max_words <= 0:
        return s
    return " ".join(s.split(" ")[:max_words]).strip()

def _llm_generate_text_for_row(theme: str, row_context: dict, col_name: str, instruction: str, max_words: int) -> str:

    prompt = f"""
Tu génères une donnée synthétique.

Thème: {theme}

Contexte ligne (JSON):
{json.dumps(row_context, ensure_ascii=False)}

Tâche:
Génère uniquement la valeur pour la colonne "{col_name}".

Instruction colonne:
{instruction}

Contraintes:
- Réponds uniquement par le texte final (pas de JSON, pas de guillemets, pas de markdown).
- Court, naturel, réaliste.
""".strip()

    raw = call_llm(prompt)
    if raw is None:
        return ""
    return _truncate_words(str(raw).strip().strip('"'), max_words)

def _llm_generate_derived_value(theme: str, row_context: dict, col_name: str, col_type: str):
    """
    Génère UNE valeur dérivée (ex: likes dépend du tweet).
    """
    prompt = f"""
Tu génères une donnée synthétique DERIVÉE.

Thème: {theme}

Contexte ligne (JSON):
{json.dumps(row_context, ensure_ascii=False)}

Tâche:
Donne une valeur réaliste pour la colonne "{col_name}" (type attendu: {col_type}).

Contraintes:
- Réponds uniquement par la valeur finale.
- Pas de JSON. Pas d'explication. Pas d'unités inutiles.
""".strip()

    raw = call_llm(prompt)
    if raw is None:
        return None

    val = str(raw).strip().strip('"')

    if col_type == "integer":
        m = re.findall(r"-?\d+", val)
        return int(m[0]) if m else 0

    if col_type == "float":
        m = re.findall(r"-?\d+(?:[.,]\d+)?", val)
        return float(m[0].replace(",", ".")) if m else 0.0

    if col_type == "boolean":
        return val.lower() in ("true", "1", "yes", "vrai")

    # string / text / category => on garde brut
    return val



def generer_prompt_llm(theme: str, colonnes: list[str], nb_lignes: int) -> str:
    spec = {"theme": theme, "colonnes": colonnes, "nb_lignes": nb_lignes}

    return f"""
Tu es un expert senior en modélisation statistique et génération de données synthétiques.

Objectif :
À partir des spécifications suivantes, propose un schéma réaliste pour générer un dataset synthétique HYBRIDE
(texte + nombres + catégories) pour le thème donné.

Spécifications :
- Thème : {spec['theme']}
- Colonnes : {spec['colonnes']}
- Nombre de lignes : {spec['nb_lignes']}

Contraintes de sortie :
1. Réponds UNIQUEMENT en JSON valide.
2. AUCUN texte explicatif, AUCUN markdown, AUCUN commentaire.
3. Le JSON doit être directement parsable par Python (json.loads).
4. Utilise uniquement les types et distributions autorisés.
5. Si tu ne peux pas produire une réponse conforme ou si l’information est insuffisante, renvoie strictement: null

Schéma attendu :
{{
  "columns": {{
    "<nom_colonne>": {{
      "type": "integer|float|boolean|string|text|category",
      "distribution": "normal|lognormal|uniform|poisson|none",
      "params": {{}},
      "categories": [],
      "probas": [],

      "text_generator": {{
        "mode": "llm|template|none",
        "instruction": "Instruction courte pour générer ce texte",
        "max_words": 20
      }},

      "derived_from": ["nom_autre_colonne_1", "nom_autre_colonne_2"]
    }}
  }},
  "row_rules": [
    {{
      "if": "<condition pseudo-code>",
      "then": "<contrainte sur les colonnes>"
    }}
  ],
  "row_text_prompt": "Optionnel: instruction globale pour cohérence sur une ligne"
}}

Règles :
- Toutes les colonnes listées doivent apparaître dans "columns".
- type = "text" ou "string" => tu peux utiliser text_generator.mode="llm" si besoin.
- Si type != "category", laisse "categories" et "probas" vides.
- Si distribution = "none", laisse "params" vide.
- Les probas doivent sommer à 1.
- "derived_from" indique les colonnes dont cette colonne dépend (ex: likes dépend du tweet).
- Les règles doivent être réalistes et cohérentes avec le thème.
"""

#json . repair


def generate_dataframe_hybride(spec_llm: dict, nb_lignes: int, theme: str, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    data = {}

    columns = spec_llm.get("columns", {}) or {}

    
    text_cols = []    
    derived_cols = []  

    for col_name, col_spec in columns.items():
        col_type = (col_spec.get("type") or "string").lower()
        dist = (col_spec.get("distribution") or "none").lower()
        params = col_spec.get("params", {}) or {}
        derived_from = col_spec.get("derived_from") or []

        tg = col_spec.get("text_generator", {}) or {}
        tg_mode = (tg.get("mode") or "none").lower()

        
        if col_type in ("text", "string") and tg_mode == "llm":
            text_cols.append(col_name)
            continue

        if derived_from:
            derived_cols.append(col_name)
            continue

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
                    cats = col_spec.get("categories", []) or []
                    p = col_spec.get("probas", None)
                    if cats:
                        arr = np.random.choice(cats, size=nb_lignes, p=p)
                    else:
                        arr = np.array([""] * nb_lignes, dtype=object)

                elif col_type == "boolean" and col_spec.get("probas"):
                    arr = np.random.choice([True, False], size=nb_lignes, p=col_spec["probas"])

                elif col_type == "boolean":
                    arr = np.random.choice([True, False], size=nb_lignes)

                elif col_type == "integer":
                    arr = np.random.randint(0, 100, size=nb_lignes)

                elif col_type == "float":
                    arr = np.random.uniform(0, 100, size=nb_lignes)

                else:
                    arr = np.array([""] * nb_lignes, dtype=object)

        if isinstance(arr, np.ndarray) and arr.dtype in [np.float64, np.float32]:
            arr = np.round(arr, 2)

        data[col_name] = arr

    df = pd.DataFrame(data)

    row_text_prompt = (spec_llm.get("row_text_prompt") or "").strip()

    for i in range(nb_lignes):
        row_ctx = df.iloc[i].to_dict()
        if row_text_prompt:
            row_ctx["_row_instruction"] = row_text_prompt

        for col_name in text_cols:
            col_spec = columns[col_name]
            tg = col_spec.get("text_generator", {}) or {}
            instruction = tg.get("instruction") or f"Génère une valeur réaliste pour {col_name}."
            max_words = _safe_int(tg.get("max_words", 20), 20)

            text_value = _llm_generate_text_for_row(
                theme=theme,
                row_context=row_ctx,
                col_name=col_name,
                instruction=instruction,
                max_words=max_words,
            )
            df.at[i, col_name] = text_value
            row_ctx[col_name] = text_value 

    for i in range(nb_lignes):
        row_ctx = df.iloc[i].to_dict()
        for col_name in derived_cols:
            col_spec = columns[col_name]
            col_type = (col_spec.get("type") or "integer").lower()
            derived_val = _llm_generate_derived_value(theme, row_ctx, col_name, col_type)
            if derived_val is not None:
                df.at[i, col_name] = derived_val

    return df

def parse_llm_spec_robuste(raw: str) -> dict | None:
    
    if raw is None:
        return None

    s = str(raw).strip()

    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    s = m.group(0)

    try:
        obj = json.loads(s)
    except Exception:
        return None

    if isinstance(obj, dict):

        keys = list(obj.get("columns", {}).keys()) if isinstance(obj.get("columns"), dict) else []
        if "columns" in obj and isinstance(obj["columns"], dict):
            fixed_cols = {}
            for k, v in obj["columns"].items():
                nk = k.strip()
                if nk.startswith("@") and nk.endswith("@"):
                    nk = nk[1:-1].strip()
                fixed_cols[nk] = v
            obj["columns"] = fixed_cols

        if "@row_rules@" in obj and "row_rules" not in obj:
            obj["row_rules"] = obj["@row_rules@"]
        if "row_rules" not in obj:
            obj["row_rules"] = []

    if not isinstance(obj, dict) or "columns" not in obj or not isinstance(obj["columns"], dict):
        return None

    for col_name, col_spec in obj["columns"].items():
        if isinstance(col_spec, dict):
            tg = col_spec.get("text_generator")
            if isinstance(tg, dict):
                if tg.get("mode") is None:
                    tg["mode"] = "none"

    return obj


