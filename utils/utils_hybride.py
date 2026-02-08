
import json              
import re                
import numpy as np       
import pandas as pd    

from utils.ulits_quantitatif import (
    saisir_colonnes,       
    call_llm,               
    creer_pdf_table,        
    creer_odt_table,      
    creer_csv_table,        
    creer_dossiers_sortie, 
    choisir_formats,       
)


def _safe_int(x, default=0):
    """
    Convertit en int 
    """
    try:
        return int(x)
    except Exception:
        return default

def _truncate_words(s: str, max_words: int | None):
    """
    Coupe un texte à N mots.
    """
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip()  
    if not max_words or max_words <= 0:
        return s
    return " ".join(s.split(" ")[:max_words]).strip()

# -------------------------------------------------------------------
# Génération LLM pour une cellule text (par ligne + contexte)
# -------------------------------------------------------------------

def _llm_generate_text_for_row(theme: str, row_context: dict, col_name: str, instruction: str, max_words: int) -> str:
    """
    Appelle le LLM pour générer une seule valeur textuelle d'une colonne,
    en tenant compte du contexte de la ligne (autres colonnes déjà remplies).
    """
   
    contexte_global = row_context.get("_contexte_global", "")
    row_instruction = row_context.get("_row_instruction", "")

    contexte_section = f"\nContexte global: {contexte_global}" if contexte_global else ""
    instruction_row_section = f"\nInstruction de cohérence: {row_instruction}" if row_instruction else ""

    contexte_formate = "\n".join([
        f"  • {k}: {v}"
        for k, v in row_context.items()
        if not k.startswith("_")
    ])

    
    prompt = f"""Tu génères UNE SEULE valeur textuelle pour compléter un tableau de données synthétiques.

Thème général: {theme}{contexte_section}

Données déjà présentes sur cette ligne:
{contexte_formate}{instruction_row_section}

TÂCHE:
Génère le contenu pour la colonne "{col_name}".

Instruction spécifique:
{instruction}

RÈGLES STRICTES:
1. Réponds UNIQUEMENT par le texte demandé, rien d'autre
2. Maximum {max_words} mots (respecte cette limite IMPÉRATIVEMENT)
3. Pas de préfixe type "Voici:", "Le texte est:", "Réponse:"
4. Pas de guillemets superflus, pas de markdown, pas de JSON
5. Texte naturel, réaliste et cohérent avec les autres colonnes de la ligne
6. Si c'est un avis/commentaire, adapte le ton à la note/sentiment des autres colonnes
7. Si c'est une description, utilise les informations des autres colonnes pour enrichir

Exemple de BONNE réponse (direct, sans préfixe):
Super produit, je recommande vivement !

Exemple de MAUVAISE réponse (à éviter):
Voici l'avis: "Super produit, je recommande vivement !"
""".strip()

   
    raw = call_llm(prompt)
    if raw is None:
        return ""

    # Nettoyage 
    val = str(raw).strip().strip('"').strip("'").strip('`')

    prefixes_a_supprimer = [
        "Voici", "Le texte est", "Réponse:", "Valeur:", "Result:",
        "Texte:", "Avis:", "Commentaire:", "Description:",
        "Voici le texte:", "Voici l'avis:", "Voici le commentaire:",
        "La réponse est:", "Le contenu est:"
    ]
    for prefix in prefixes_a_supprimer:
        if val.lower().startswith(prefix.lower()):
            val = val[len(prefix):].strip().strip(":").strip('"').strip("'").strip()
            break

    if val.startswith('"') and val.endswith('"'):
        val = val[1:-1]
    if val.startswith("'") and val.endswith("'"):
        val = val[1:-1]

    return _truncate_words(val, max_words)

# -------------------------------------------------------------------
# Génération LLM pour une valeur qui dépend d'autres colonnes
# -------------------------------------------------------------------

def _llm_generate_derived_value(theme: str, row_context: dict, col_name: str, col_type: str):
    """
    Génère une valeur dérivée cohérente avec la ligne 
    """
    type_instructions = {
        "date": "Format: YYYY-MM-DD (exemple: 2024-03-15)",
        "datetime": "Format: YYYY-MM-DD HH:MM:SS (exemple: 2024-03-15 14:30:00)",
        "integer": "Nombre entier uniquement (exemple: 42)",
        "float": "Nombre décimal (exemple: 3.14)",
        "boolean": "true ou false uniquement",
        "string": "Texte court et pertinent",
        "text": "Texte descriptif cohérent avec le contexte",
        "category": "Une catégorie parmi les valeurs logiques pour ce champ"
    }
    instruction = type_instructions.get(col_type, "Valeur appropriée au type")

    prompt = f"""Tu génères UNE SEULE valeur synthétique pour compléter un tableau de données.

Thème: {theme}

Contexte de la ligne actuelle:
{json.dumps(row_context, ensure_ascii=False, indent=2)}

TÂCHE:
Génère une valeur réaliste pour la colonne "{col_name}".
Type attendu: {col_type}
{instruction}

RÈGLES STRICTES:
- Réponds UNIQUEMENT par la valeur, rien d'autre
- Pas de phrase, pas d'explication, pas de "Voici", pas de guillemets superflus
- La valeur DOIT être cohérente avec le contexte fourni
- {f"Format obligatoire: {instruction}" if col_type in ["date", "datetime"] else ""}

Exemple de réponse valide: {"2024-01-15" if col_type == "date" else ("42" if col_type == "integer" else "exemple de texte")}
""".strip()

    raw = call_llm(prompt)
    if raw is None:
        return None

    # Nettoyage 
    val = str(raw).strip().strip('"').strip("'")

    for prefix in ["Voici", "La valeur est", "Réponse:", "Valeur:", "Result:"]:
        if val.startswith(prefix):
            val = val[len(prefix):].strip().strip(":").strip()

    if col_type == "integer":
        m = re.findall(r"-?\d+", val)
        return int(m[0]) if m else 0

    if col_type == "float":
        m = re.findall(r"-?\d+(?:[.,]\d+)?", val)
        return float(m[0].replace(",", ".")) if m else 0.0

    if col_type == "boolean":
        return val.lower() in ("true", "1", "yes", "vrai", "oui")

    if col_type in ("date", "datetime"):
        if not val or val.lower() in ("none", "null", ""):
            return None
        return val.strip()

    return val if val else None

# -------------------------------------------------------------------
# Prompt demande au LLM le schéma JSON du dataset
# -------------------------------------------------------------------

def generer_prompt_llm(theme: str, colonnes: list[str], nb_lignes: int, contexte: str = "") -> str:
    """
    Produit le prompt pour demander au LLM une JSON 
    """
    spec = {"theme": theme, "colonnes": colonnes, "nb_lignes": nb_lignes}

    # Optionnel: contexte fourni par user (bon pour guider le LLM).
    contexte_section = f"\nContexte additionnel fourni par l'utilisateur:\n{contexte}\n" if contexte else ""

    return f"""
Tu es un expert senior en modélisation statistique et génération de données synthétiques.

Objectif :
À partir des spécifications suivantes, propose un schéma réaliste pour générer un dataset synthétique HYBRIDE
(texte + nombres + catégories) pour le thème donné.

Spécifications :
- Thème : {spec['theme']}
- Colonnes : {spec['colonnes']}
- Nombre de lignes : {spec['nb_lignes']}{contexte_section}

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
      "type": "integer|float|boolean|string|text|category|date|datetime",
      "distribution": "normal|lognormal|uniform|poisson|none",
      "params": {{}},
      "categories": [],
      "probas": [],

      "text_generator": {{
        "mode": "llm|template|none",
        "instruction": "Instruction courte pour générer ce texte en tenant compte du contexte",
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
  "row_text_prompt": "Instruction globale pour assurer la cohérence des données textuelles sur une ligne (utilise le contexte fourni)"
}}

Règles :
- Toutes les colonnes listées doivent apparaître dans "columns".
- type = "text" ou "string" => tu peux utiliser text_generator.mode="llm" si besoin.
- Si type != "category", laisse "categories" et "probas" vides.
- Si distribution = "none", laisse "params" vide.
- Les probas doivent sommer à 1.
- "derived_from" indique les colonnes dont cette colonne dépend (ex: likes dépend du tweet).
- Les règles doivent être réalistes, cohérentes avec le thème ET le contexte fourni.
- Pour les colonnes de type date/datetime, utilise des params avec "min" et "max" (format: YYYY-MM-DD).
"""

# -------------------------------------------------------------------
# Génération DataFrame 
# -------------------------------------------------------------------

def generate_dataframe_hybride(spec_llm: dict, nb_lignes: int, theme: str, contexte: str = "", seed: int = 42) -> pd.DataFrame:
    """
    1) Pré-génère toutes les colonnes "non LLM" (numériques/catégories/dates)
    2) Génère ensuite les colonnes textuelles via LLM (dépend des valeurs déjà générées)
    3) Génère ensuite les colonnes dérivées via LLM
    """
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

        if col_type in ("date", "datetime"):
            start_date = params.get("min", params.get("start", "2023-01-01"))
            end_date = params.get("max", params.get("end", "2024-12-31"))

            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)

            random_timestamps = np.random.randint(start_ts.value, end_ts.value, size=nb_lignes)
            arr = pd.to_datetime(random_timestamps)

            if col_type == "date":
                arr = arr.strftime('%Y-%m-%d')
            else:
                arr = arr.strftime('%Y-%m-%d %H:%M:%S')

            data[col_name] = arr
            continue

        match dist:
            case "normal":
                mean = params.get("mean", 0)
                std = params.get("std", params.get("stddev", 1))
                arr = np.random.normal(loc=mean, scale=std, size=nb_lignes)

            case "lognormal":
                mu = params.get("mu", params.get("mean", 0))
                sigma = params.get("sigma", params.get("std", 1))
                arr = np.random.lognormal(mean=mu, sigma=sigma, size=nb_lignes)

            case "uniform":
                low = params.get("low", params.get("min", 0))
                high = params.get("high", params.get("max", 1))
                try:
                    low = float(low)
                    high = float(high)
                except (ValueError, TypeError):
                    print(f" Colonne '{col_name}': paramètres invalides pour uniform.")
                    low, high = 0, 1
                arr = np.random.uniform(low=low, high=high, size=nb_lignes)

            case "poisson":
                lam = params.get("lambda", params.get("lam", 1))
                arr = np.random.poisson(lam=lam, size=nb_lignes)

            case _:
                # Fallbacks 
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

        # Standardisation
        if isinstance(arr, np.ndarray) and arr.dtype in [np.float64, np.float32]:
            arr = np.round(arr, 2)

        data[col_name] = arr

    df = pd.DataFrame(data)

    row_text_prompt = (spec_llm.get("row_text_prompt") or "").strip()

    for i in range(nb_lignes):
        row_ctx = df.iloc[i].to_dict()

        if row_text_prompt:
            row_ctx["_row_instruction"] = row_text_prompt
        if contexte:
            row_ctx["_contexte_global"] = contexte

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
        if contexte:
            row_ctx["_contexte_global"] = contexte

        for col_name in derived_cols:
            col_spec = columns[col_name]
            col_type = (col_spec.get("type") or "integer").lower()
            derived_val = _llm_generate_derived_value(theme, row_ctx, col_name, col_type)
            if derived_val is not None:
                df.at[i, col_name] = derived_val

    return df

# -------------------------------------------------------------------
# Parse du JSON renvoyé par le LLM
# -------------------------------------------------------------------

def parse_llm_spec_robuste(raw: str) -> dict | None:
    """
    Transforme une réponse LLM "presque JSON" en dict exploitable.
    """
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

        # Nettoyage des noms de colonnes
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
