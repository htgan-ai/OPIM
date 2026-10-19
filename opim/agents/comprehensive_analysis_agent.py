"""Multimodal fusion agent used by the public OPIM release."""
from copy import deepcopy
import math
from typing import Dict, List, Any

try:
    from opim.models.shared_memory import SharedMemoryPool
    from opim.opim_metrics import OPIM_SCORE_MAX
except ImportError:
    from models.shared_memory import SharedMemoryPool
    from opim_metrics import OPIM_SCORE_MAX


class ComprehensiveAnalysisAgent:
    def __init__(self):
        self.name = "ComprehensiveAnalysisAgent"
        self.shared_memory = SharedMemoryPool()
        # The paper specifies modality direction, while these numeric defaults
        # are release-level hyperparameters and can be changed by callers.
        self.dimension_weights = {
            "Sincerity": {"text_weight": 0.6, "image_weight": 0.4},
            "Excitement": {"text_weight": 0.4, "image_weight": 0.6},
            "Professionalism": {"text_weight": 0.6, "image_weight": 0.4},
            "Sophistication": {"text_weight": 0.6, "image_weight": 0.4},
            "Attractiveness": {"text_weight": 0.6, "image_weight": 0.4},
            "Materialism": {"text_weight": 0.4, "image_weight": 0.6},
        }

    @staticmethod
    def _valid_score(value: Any) -> bool:
        try:
            # OPIM trait scores (Eqs. 1-3) range over [0, OPIM_SCORE_MAX], not [0, 1].
            return math.isfinite(float(value)) and 0.0 <= float(value) <= OPIM_SCORE_MAX + 1e-9
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _evidence_list(results: Dict, key: str) -> List[str]:
        evidence = results.get("evidence", {}) if isinstance(results, dict) else {}
        if not isinstance(evidence, dict):
            return []
        values = evidence.get(key, [])
        if isinstance(values, str):
            return [values]
        return [str(value) for value in values] if isinstance(values, list) else []

    def analyze(self, text_results: Dict, image_results: Dict) -> Dict:
        text_results = text_results or {}
        image_results = image_results or {}
        output = {
            "dimension_scores": self._integrate_scores(
                self._modal_scores(text_results, "text") or text_results.get("dimension_scores", {}),
                self._modal_scores(image_results, "image") or image_results.get("dimension_scores", {}),
            ),
            "evidence": {
                "text_evidence": self._evidence_list(text_results, "text_evidence"),
                "image_evidence": self._evidence_list(image_results, "image_evidence"),
            },
            "failed": list(text_results.get("failed", [])) + list(image_results.get("failed", [])),
        }
        return output

    @staticmethod
    def _modal_scores(results: Any, modality: str) -> Dict:
        """Read modality scores from both supported result shapes.

        The compact release exposes ``dimension_scores`` directly, while the
        older controller stored the same values under ``text_scores`` and
        ``image_scores`` (or under ``*_results``).  Keeping the extraction in
        one place prevents consistency checks from assuming one shape and
        raising ``KeyError`` on the other.
        """
        if not isinstance(results, dict):
            return {}
        score_key = "%s_scores" % modality
        result_key = "%s_results" % modality
        candidates = [results.get(score_key), results.get(result_key), results.get(modality)]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            nested = candidate.get("dimension_scores")
            if isinstance(nested, dict):
                return nested
            # A bare dimension-to-score mapping is also accepted.
            known_dimensions = {
                "Sincerity", "Excitement", "Professionalism",
                "Sophistication", "Attractiveness", "Materialism",
            }
            if any(dimension in candidate for dimension in known_dimensions):
                return candidate
        return {}

    def _integrate_scores(self, text_scores: Dict, image_scores: Dict) -> Dict:
        integrated_scores = {}
        text_scores = text_scores if isinstance(text_scores, dict) else {}
        image_scores = image_scores if isinstance(image_scores, dict) else {}
        for dimension, weights in self.dimension_weights.items():
            available = []
            if self._valid_score(text_scores.get(dimension)):
                available.append((float(text_scores[dimension]), weights["text_weight"]))
            if self._valid_score(image_scores.get(dimension)):
                available.append((float(image_scores[dimension]), weights["image_weight"]))
            if not available:
                integrated_scores[dimension] = None
                continue
            total_weight = sum(weight for _, weight in available)
            integrated_scores[dimension] = sum(score * weight for score, weight in available) / total_weight
        return integrated_scores

    def _integrate_analysis(self, text_results: Dict, image_results: Dict, target: str) -> Dict:
        return {
            "dimension_scores": self._integrate_scores(
                self._modal_scores(text_results or {}, "text") or (text_results or {}).get("dimension_scores", {}),
                self._modal_scores(image_results or {}, "image") or (image_results or {}).get("dimension_scores", {}),
            ),
            "evidence": {
                "text_evidence": self._evidence_list(text_results or {}, "text_evidence"),
                "image_evidence": self._evidence_list(image_results or {}, "image_evidence"),
            },
            "metadata": {
                "target": target,
                "text_metadata": (text_results or {}).get("metadata", {}),
                "image_metadata": (image_results or {}).get("metadata", {}),
            },
            "failed": list((text_results or {}).get("failed", [])) + list((image_results or {}).get("failed", [])),
        }

    def _check_consistency(self, results: Dict) -> Dict:
        discrepancies = []
        missing = []
        if not isinstance(results, dict):
            results = {}
        # Legacy callers pass ``brand``/``user`` objects.  Direct callers may
        # pass one pair of ``text_results``/``image_results`` at the root.
        target_names = [name for name in ("brand", "user") if isinstance(results.get(name), dict)]
        if not target_names and any(key in results for key in ("text_results", "image_results", "text_scores", "image_scores")):
            target_names = ["root"]
        for target in target_names:
            target_results = results if target == "root" else results.get(target, {})
            text_scores = self._modal_scores(target_results, "text")
            image_scores = self._modal_scores(target_results, "image")
            for dimension in self.dimension_weights:
                text_score = text_scores.get(dimension) if isinstance(text_scores, dict) else None
                image_score = image_scores.get(dimension) if isinstance(image_scores, dict) else None
                if not self._valid_score(text_score) or not self._valid_score(image_score):
                    missing.append({"target": target, "dimension": dimension})
                    continue
                difference = abs(float(text_score) - float(image_score))
                if difference > 0.3:
                    discrepancies.append({
                        "target": target, "dimension": dimension,
                        "text_score": float(text_score), "image_score": float(image_score),
                        "difference": difference,
                    })
        return {"is_consistent": not discrepancies, "discrepancies": discrepancies, "missing": missing}

    def _resolve_discrepancies(self, results: Dict, discrepancies: List[Dict]) -> Dict:
        resolved_results = deepcopy(results) if isinstance(results, dict) else {}
        for discrepancy in discrepancies or []:
            if not isinstance(discrepancy, dict):
                continue
            target = discrepancy.get("target")
            dimension = discrepancy.get("dimension")
            if not isinstance(dimension, str) or dimension not in self.dimension_weights:
                continue
            target_results = resolved_results if target == "root" else resolved_results.get(target)
            if not isinstance(target_results, dict):
                # A missing target cannot be resolved safely.  Leave the
                # original object unchanged instead of fabricating a branch.
                continue
            evidence = target_results.get("evidence", {})
            text_evidence = evidence.get("text_evidence", []) if isinstance(evidence, dict) else []
            image_evidence = evidence.get("image_evidence", []) if isinstance(evidence, dict) else []
            text_count = len(text_evidence) if isinstance(text_evidence, list) else bool(text_evidence)
            image_count = len(image_evidence) if isinstance(image_evidence, list) else bool(image_evidence)
            total = text_count + image_count
            text_score = discrepancy.get("text_score")
            image_score = discrepancy.get("image_score")
            if not self._valid_score(text_score) or not self._valid_score(image_score):
                continue
            if total:
                dimension_scores = target_results.setdefault("dimension_scores", {})
                if not isinstance(dimension_scores, dict):
                    dimension_scores = {}
                    target_results["dimension_scores"] = dimension_scores
                dimension_scores[dimension] = (
                    text_count * float(text_score) + image_count * float(image_score)
                ) / total
        return resolved_results
