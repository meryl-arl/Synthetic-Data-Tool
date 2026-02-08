# STENDHAL (Synthetic Tool for Engineering Numeric Data Harvesting And Labeling)

## Présentation du projet

**STENDHAL** est un outil Python permettant de **générer des données synthétiques** à l’aide de **modèles de langage (LLM)**.  
Il permet de produire :

- des **datasets quantitatifs** (nombres, catégories, booléens, dates),
- des **contenus qualitatifs** (textes, articles),
- des **datasets hybrides** combinant données numériques, textuelles et colonnes dérivées.

Les données générées peuvent ensuite être exportées en **CSV**, **PDF** ou **ODT**.

Ce projet a été développé dans le cadre de **mon stage chez Arlequin AI**, avec pour objectif d’explorer des solutions de **génération de données synthétiques réalistes**, exploitables pour des démonstrations, tests, prototypes ou jeux de données fictifs.

---

## Pourquoi les données synthétiques ?

Les **données synthétiques** permettent de :

- tester des pipelines de data sans utiliser de données réelles ou sensibles,
- contourner les problèmes de confidentialité (RGPD, données personnelles),
- générer rapidement de grands volumes de données cohérentes,
- simuler des cas d’usage avant la disponibilité de vraies données,
- créer des datasets de démonstration pour des outils data / IA.

L’outil vise donc à fournir une **alternative flexible et contrôlable** aux datasets réels, tout en conservant une **cohérence sémantique** grâce à l’utilisation de LLM.

---

## Fonctionnement général

Le fonctionnement du projet repose sur trois grandes étapes :

### 1. Génération d’un schéma via LLM
À partir :
- d’un **thème**
- d’une **liste de colonnes**
- d’un **nombre de lignes**

le LLM génère un **schéma JSON** décrivant :
- le type de chaque colonne,
- les distributions statistiques (si quantitatif),
- les règles de génération,
- les colonnes textuelles,
- les colonnes dérivées.

### 2. Génération des données
Selon le mode choisi :
- **Quantitatif** : génération locale via `numpy` / `pandas`
- **Qualitatif** : génération de textes via le LLM
- **Hybride** :
  - données numériques générées localement,
  - données textuelles générées via LLM ligne par ligne,
  - colonnes dérivées calculées à partir du contexte de la ligne

### 3. Export des résultats
Les résultats peuvent être exportés en :
- **CSV** (Excel / analyse),
- **PDF** (présentation),
- **ODT** (documents éditables).

---

## Modes disponibles

- **Quanti** : génération de datasets numériques
- **Quali** : génération de documents texte
- **Hybride** : combinaison des deux avec logique de dépendance entre colonnes

Le tout est accessible via une **CLI** interactive.

---

## Technologies utilisées

- **Python**
- **LLM (DeepInfra – API compatible OpenAI)**
- `numpy`, `pandas`, `polars`
- `click` (CLI)
- `tqdm` (barres de progression)
- `reportlab` (PDF)
- `odfpy` (ODT)
- `python-dotenv` (gestion des clés API)

