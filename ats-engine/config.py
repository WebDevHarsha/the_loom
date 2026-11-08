"""Central configuration for the ATS engine.

Keep every tunable here so evaluation logic stays free of hardcoded values.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # .../ats-engine
PROJECT_ROOT = BASE_DIR.parent                      # project root (F:\RB)

# ---------------------------------------------------------------------------
# Models / caches
# Everything that gets downloaded MUST stay inside this directory.
# ---------------------------------------------------------------------------
MODELS_DIR = BASE_DIR / ".models"                   # sentence-transformers cache
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"

SKILL_TAXONOMY_FILE = DATA_DIR / "skill_taxonomy.json"
SKILL_ALIASES_FILE = DATA_DIR / "skill_aliases.json"
ACTION_VERBS_FILE = DATA_DIR / "action_verbs.json"

# ---------------------------------------------------------------------------
# Matching thresholds
# ---------------------------------------------------------------------------
FUZZY_THRESHOLD = 85          # RapidFuzz partial_ratio >= this -> partial match
FUZZY_MIN_LEN = 4             # terms shorter than this are never fuzzy-matched
SKILL_SECTION_FUZZY = 90      # threshold when canonicalizing raw skills-section tokens

# Semantic similarity remapping: cosine in [FLOOR, CEILING] -> [0, 100]
SEMANTIC_FLOOR = 0.25
SEMANTIC_CEILING = 0.80

# ---------------------------------------------------------------------------
# Resume structure expectations
# ---------------------------------------------------------------------------
EXPECTED_SECTIONS = ("education", "skills")
USEFUL_SECTIONS = ("experience", "projects", "summary")

# Garbled-text tolerance: fraction of non-printable-ASCII chars tolerated
GARBLE_RATIO_LIMIT = 0.05

# ---------------------------------------------------------------------------
# Content quality thresholds
# ---------------------------------------------------------------------------
ACTION_VERB_TARGET = 0.55     # fraction of bullets that should start with an action verb
QUANTIFIED_TARGET = 0.30      # fraction of bullets that should contain a number
LONG_BULLET_WORDS = 45        # bullets longer than this are flagged
WEAK_PHRASES = (
    "responsible for",
    "worked on",
    "helped with",
    "helped to",
    "assisted with",
    "involved in",
    "familiar with",
    "duties included",
    "various tasks",
    "participated in various",
)

# Grammar checking (language_tool_python) is strictly opt-in because it
# downloads a ~200 MB LanguageTool server jar and needs Java on the system.
# Set ATS_ENGINE_GRAMMAR=1 only if you accept that.
GRAMMAR_ENV_FLAG = "ATS_ENGINE_GRAMMAR"
GRAMMAR_ENABLED = False

# ---------------------------------------------------------------------------
# Scoring weights (sum to 100; see scoring/weights.py)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "Keyword Matching": 25.0,
    "Semantic Relevance": 20.0,
    "Experience Relevance": 20.0,
    "Qualifications": 10.0,
    "Resume Structure": 15.0,
    "Content Quality": 10.0,
}
