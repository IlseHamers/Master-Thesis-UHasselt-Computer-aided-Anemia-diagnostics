"""
RAG/KNN retrieval for anemia prompting
Class based KNN with adaptive weights. Each lab value is represented by a direction class (see RAG_prep_thesis.Rmd).
Retrieval finds top 3 most similar reference cases and puts them in the prompt.

use:
    from RAG_retriever import RAGRetriever, build_rag_prompt
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Union

# CONFIG

K_DEFAULT = 3

# feature groups
GROEPEN: dict[str, list[str]] = {
    "CBC": ["Hemoglobine", "MCV", "Erytrocyten", "Trombocyten", "Leucocyten", "Hematocriet"],
    "IJzer": ["Ferritine", "IJzer", "Transf.sat.", "Transferrine"],
    "Megaloblast": ["Vitamine B12", "Foliumzuur"],
    "Hemolyse": ["LD", "Haptoglobine", "Reticulocyten relatief", "Reticulocyten absoluut"],
    "Ontsteking": ["CRP", "Bezinking"],
    "Nier": ["Kreatinine", "eGFR CKD-EPI"],
    "Overig": ["ALAT", "TSH", "NT-proBNP"],
}

# group weights
GROEP_GEWICHT: dict[str, float] = {
    "CBC": 3.0,
    "IJzer": 2.0,
    "Megaloblast": 2.0,
    "Hemolyse": 2.0,
    "Ontsteking": 1.5,
    "Nier": 1.0,
    "Overig": 0.5,
}

# multiplying based on direction class
ABNORM_MULT: dict[int, float] = {0: 1.0, 1: 1.5, 2: 2.5}

# core features CBC, at least 3 should be there
KERN_FEATURES = {"Hemoglobine", "MCV", "Leucocyten", "Trombocyten", "Erytrocyten"}
MIN_KERN_OVERLAP = 3

# small residual for tie breaking based on raw values, max 0.15
MAGN_RES_MAX = 0.15

# Feature → groep lookup
_F2G: dict[str, str] = {feat: grp for grp, feats in GROEPEN.items() for feat in feats}


# HELPER: NaN-check
def _is_nan(val) -> bool:
    """True if value is missing"""
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return val is None or str(val).strip() in ("", "nan", "NaN", "None")


def _safe_float(val) -> float:
    """Return val as float or NaN if missing or non-numeric"""
    if _is_nan(val):
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


# CORE FUNCTiONS
def adaptieve_gewichten(test_klassen: dict[str, float]) -> dict[str, float]:
    """
    Build per-feature weight dictionary from test case:
    --> weight = group_weight * abnormality_multiplier
    Features missing in test case are ignored
    """
    w: dict[str, float] = {}
    for feat, cls in test_klassen.items():
        if math.isnan(cls):
            continue
        grp = _F2G.get(feat, "Overig")
        gw = GROEP_GEWICHT.get(grp, 0.5)
        abs_cls = min(int(abs(cls)), 2)  # cap at 2
        mult = ABNORM_MULT.get(abs_cls, 1.0)
        w[feat] = gw * mult
    return w


def feat_afstand(cls_t: float, cls_r: float, raw_t: float, raw_r: float) -> float:
    """
    Distance contribution of a feature:
      - test value missing -> NaN
      - ref value missing -> max(0.5, |cls_t|)
      - same class -> small residual (max MAGN_RES_MAX)
      - class difference of 1 -> 1.0
      - class difference of 2 -> 2.0
      - class difference of at least 3 -> 3.0
    """
    if math.isnan(cls_t):
        return float("nan")

    if math.isnan(cls_r):
        return max(0.5, abs(cls_t))

    verschil = abs(cls_t - cls_r)

    if verschil == 0:
        # tie-breaking on raw values, scaled to max MAGN_RES_MAX
        if not math.isnan(raw_t) and not math.isnan(raw_r):
            schaal = max(abs(raw_t) + abs(raw_r), 1.0)
            return MAGN_RES_MAX * min(abs(raw_t - raw_r) / schaal, 1.0)
        return 0.0
    elif verschil <= 1:
        return 1.0
    elif verschil <= 2:
        return 2.0
    else:
        return 3.0


def patient_afstand(
    test_klassen: dict[str, float],
    test_raw: dict[str, float],
    ref_klassen: dict[str, float],
    ref_raw: dict[str, float],
    gewichten: dict[str, float],
    min_kern_overlap: int = MIN_KERN_OVERLAP,
) -> float:
    """
    Weighted mean feature distance between test case and reference case.
    NaN if too few CBC features are present.
    """
    som_gewogen = 0.0
    som_gewicht = 0.0
    n_kern = 0

    for feat, w in gewichten.items():
        cls_t = test_klassen.get(feat, float("nan"))
        if math.isnan(cls_t):
            continue

        cls_r = ref_klassen.get(feat, float("nan"))
        raw_t = test_raw.get(feat, float("nan"))
        raw_r = ref_raw.get(feat, float("nan"))

        d = feat_afstand(cls_t, cls_r, raw_t, raw_r)
        if math.isnan(d):
            continue

        som_gewogen += w * d
        som_gewicht += w

        if not math.isnan(cls_r) and feat in KERN_FEATURES:
            n_kern += 1

    if som_gewicht == 0.0 or n_kern < min_kern_overlap:
        return float("nan")

    return som_gewogen / som_gewicht


# RETRIEVER
class RAGRetriever:
    """
    Retrieves top-k nearest reference cases
    """

    def __init__(
        self,
        ref_klasse_path: str,
        ref_raw_path: str,
        top_k: int = K_DEFAULT,
        max_distance: float = 1.5,
        min_kern_overlap: int = MIN_KERN_OVERLAP,
    ):
        self.top_k = top_k
        self.max_distance = max_distance
        self.min_kern_overlap = min_kern_overlap

        self.ref_klasse = pd.read_csv(ref_klasse_path)
        self.ref_raw = pd.read_csv(ref_raw_path)

        # Detect feature names from _klasse columns
        klasse_cols = [c for c in self.ref_klasse.columns if c.endswith("_klasse")]
        self.features = [c.removesuffix("_klasse") for c in klasse_cols]

        self._ref_ids: list[str] = self.ref_klasse["UniekLabnummer"].astype(str).tolist()

    # ----------------------------------------------------------
    def _rij_naar_klassen(self, rij: Union[pd.Series, dict]) -> dict[str, float]:
        """Extracts {feature: class_value} from a row."""
        result = {}
        for feat in self.features:
            col = f"{feat}_klasse"
            val = rij.get(col) if isinstance(rij, dict) else rij.get(col)
            result[feat] = _safe_float(val)
        return result

    def _rij_naar_raw(self, rij: Union[pd.Series, dict]) -> dict[str, float]:
        """Extracts {feature: raw_value} from a row."""
        result = {}
        for feat in self.features:
            col = f"{feat}_raw"
            val = rij.get(col) if isinstance(rij, dict) else rij.get(col)
            result[feat] = _safe_float(val)
        return result

    # ----------------------------------------------------------
    def retrieve(self, testcase_rij: Union[pd.Series, dict]) -> list[str]:
        """
        Return UniekLabnummers of top-k nearest ref cases.
        """
        test_klassen = self._rij_naar_klassen(testcase_rij)
        test_raw = self._rij_naar_raw(testcase_rij)
        gewichten = adaptieve_gewichten(test_klassen)

        afstanden: list[float] = []
        for i in range(len(self.ref_klasse)):
            ref_rij = self.ref_klasse.iloc[i]
            ref_klassen = self._rij_naar_klassen(ref_rij)
            ref_raw_ = self._rij_naar_raw(ref_rij)
            d = patient_afstand(
                test_klassen,
                test_raw,
                ref_klassen,
                ref_raw_,
                gewichten,
                self.min_kern_overlap,
            )
            afstanden.append(d)

        # sort by distance, NaN at the end
        sorted_idx = sorted(
            range(len(afstanden)),
            key=lambda i: afstanden[i] if not math.isnan(afstanden[i]) else float("inf"),
        )

        selected: list[str] = []
        for idx in sorted_idx:
            d = afstanden[idx]
            if math.isnan(d) or d > self.max_distance:
                break
            selected.append(self._ref_ids[idx])
            if len(selected) >= self.top_k:
                break

        return selected

    # ----------------------------------------------------------
    def get_distances(self, testcase_rij: Union[pd.Series, dict]) -> dict[str, float]:
        """
        Returns a dictionary of distances to all reference cases. Used for logging/debugging.
        """
        test_klassen = self._rij_naar_klassen(testcase_rij)
        test_raw = self._rij_naar_raw(testcase_rij)
        gewichten = adaptieve_gewichten(test_klassen)

        result: dict[str, float] = {}
        for i in range(len(self.ref_klasse)):
            ref_rij = self.ref_klasse.iloc[i]
            ref_klassen = self._rij_naar_klassen(ref_rij)
            ref_raw_ = self._rij_naar_raw(ref_rij)
            d = patient_afstand(
                test_klassen,
                test_raw,
                ref_klassen,
                ref_raw_,
                gewichten,
                self.min_kern_overlap,
            )
            uid = self._ref_ids[i]
            result[uid] = d
        return result

    # ----------------------------------------------------------
    def format_case_as_markdown(self, row: pd.Series) -> str:
        """
        Format a reference case as a Markdown table with conclusion (Beschrijving).
        """
        skip = {"UniekLabnummer", "Source", "Beschrijving", "Anemie protocol"}
        tabel_cols = [c for c in row.index if c not in skip]

        lines = ["| Naam | Resultaat |", "|-----------|--------|"]
        for col in tabel_cols:
            val = row[col]
            val_str = "NA" if _is_nan(val) else str(val)
            lines.append(f"| {col} | {val_str} |")

        tabel = "\n".join(lines)

        beschrijving = row.get("Beschrijving", "")
        if _is_nan(beschrijving):
            beschrijving = ""

        return f"{tabel}\n\n**Conclusie referentiecasus:** {beschrijving}"

    # ----------------------------------------------------------
    def get_context(self, testcase_rij: Union[pd.Series, dict]) -> str:
        """
        Returns a Markdown string with the top-k reference cases for the given test case to put into prompt.
        Empty string if no matches.
        """
        ids = self.retrieve(testcase_rij)
        if not ids:
            return ""

        raw_rows = self.ref_raw[self.ref_raw["UniekLabnummer"].astype(str).isin(ids)].copy()

        # nearest first
        order = {uid: i for i, uid in enumerate(ids)}
        raw_rows["_sort"] = raw_rows["UniekLabnummer"].astype(str).map(order)
        raw_rows = raw_rows.sort_values("_sort").drop(columns="_sort")

        blokken = []
        for _, row in raw_rows.iterrows():
            ref_id = row["UniekLabnummer"]
            blok = f"### Referentiecasus: {ref_id}\n\n{self.format_case_as_markdown(row)}"
            blokken.append(blok)

        return "\n\n---\n\n".join(blokken)


# PROMPT BUILDER
def build_rag_prompt(
    user_input: str,
    rag_context: str,
    system_prompt: str,
) -> list[dict]:
    if rag_context.strip():
        content = (
            "### CONTEXT: REFERENTIE DATABASE\n"
            "Gebruik de onderstaande informatie ENKEL om vergelijkbare medische redeneringen te begrijpen. "
            "Let op: de waarden in deze casussen zijn van ANDERE patiënten.\n\n"
            f"{rag_context}\n"
            "\n"
            "### HUIDIGE PATIËNTGEGEVENS (Prioriteit)\n"
            "VOER DE VOLGENDE STAPPEN UIT:\n"
            "1. Analyseer de onderstaande laboratoriumwaarden van de HUIDIGE patiënt.\n"
            "2. Pas de regels uit de System Prompt toe op DEZE waarden.\n\n"
            "GEGEVENS HUIDIGE PATIËNT:\n"
            "--------------------------------------------------\n"
            f"{user_input}\n"
            "--------------------------------------------------\n"
        )
    else:
        content = user_input

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
