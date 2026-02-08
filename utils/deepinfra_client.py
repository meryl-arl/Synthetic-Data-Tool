# ============================================================
# deepinfra_client.py
# Client OpenAI-compatible pour DeepInfra
# Annoté "entreprise" — tu peux copier-coller tel quel.
# ============================================================


import os                     
from openai import OpenAI     
from dotenv import load_dotenv 

load_dotenv()

# ------------------------------------------------------------
# Factory du client DeepInfra
# ------------------------------------------------------------

def make_deepinfra_client() -> OpenAI:
    """
    Crée et retourne un client OpenAI-compatible configuré pour DeepInfra.
    """

    # Récupération de la clé API depuis l’environnement
    api_key = os.environ.get("DEEPINFRA_API_KEY")

    # Fail fast : si la clé n’existe pas, on arrête tout immédiatement
    # (meilleur que de laisser l’API renvoyer des erreurs obscures plus tard)
    if not api_key:
        raise RuntimeError(
            "Set DEEPINFRA_API_KEY in your environment "
            "(via .env or system environment variables)."
        )

    # Création du client OpenAI

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai"
    )
