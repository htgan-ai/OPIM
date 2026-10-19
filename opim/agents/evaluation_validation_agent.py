from typing import Any, Dict, List, Optional
import numpy as np
from datetime import datetime
import json
import os
import requests
import time
import random
import itertools
import warnings
from collections import defaultdict

# 配置 NumPy 环境
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from opim.api_config import api_endpoint, api_key
except ImportError:
    from api_config import api_endpoint, api_key
try:
    from opim.opim_metrics import OPIM_COMPONENT_MAX, OPIM_SCORE_MAX
except ImportError:
    from opim_metrics import OPIM_COMPONENT_MAX, OPIM_SCORE_MAX


class EvaluationValidationAgent:
    # These names identify pipeline stages.  They are deliberately kept
    # separate from the OPIM output metrics (perceptual_energy,
    # perceptual_stability, and perceptual_coverage).  A final score file does
    # not contain enough information to validate stage weights, so stage data
    # must be supplied explicitly under one of the containers handled below.
    _STAGE_ALIASES = {
        "memory": "memory_stream",
        "memory_stream": "memory_stream",
        "reflection": "reflection",
        "reaction": "reaction",
    }

    def __init__(self):
        self.name = "EvaluationValidationAgent"
        
        # 品牌特征记录
        self.brand_characteristics = {
            'Chanel': {},
            'LouisVuitton': {},
            'Hermes': {}  # 可以继续添加其他品牌
        }
        
        # 跨品牌验证数据
        self.cross_brand_validation = {
            'weight_history': defaultdict(list),  # 记录不同品牌的权重
            'performance_metrics': defaultdict(list)  # 记录性能指标
        }
        
        # API配置
        self.api_key = api_key()
        self.api_url = api_endpoint()
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # 初始权重配置
        self.current_weights = {
            'memory_stream': {
                'evidence_quality': 0.4,  # 证据质量权重
                'reliability': 0.3,       # 可靠性权重
                'consistency': 0.3        # 一致性权重
            },
            'reflection': {
                'evidence_quality': 0.3,  # 证据质量权重
                'pattern_relevance': 0.3, # 模式关联权重
                'reliability': 0.2,       # 可靠性权重
                'consistency': 0.2        # 一致性权重
            },
            'reaction': {
                'pattern_relevance': 0.3, # 模式关联权重
                'evidence_quality': 0.3,  # 证据质量权重
                'reliability': 0.2,       # 可靠性权重
                'consistency': 0.2        # 一致性权重
            }
        }
        
        # 学习历史记录
        self.learning_history = []
        
        # 自适应学习参数
        self.learning_rate = 0.1
        self.decay_rate = 0.95
        self.min_learning_rate = 0.01
        self.performance_threshold = 0.7

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Return a finite Python float for report/JSON generation."""
        try:
            if value is None:
                return default
            converted = float(value)
            return converted if np.isfinite(converted) else default
        except (TypeError, ValueError, OverflowError):
            return default

    @classmethod
    def _to_builtin(cls, value: Any) -> Any:
        """Recursively convert NumPy/set values to JSON-compatible values."""
        if isinstance(value, dict):
            return {str(k): cls._to_builtin(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_builtin(v) for v in value]
        if isinstance(value, set):
            return [cls._to_builtin(v) for v in sorted(value, key=str)]
        if isinstance(value, np.ndarray):
            return [cls._to_builtin(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _optional_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        """Return a finite Python float, or ``None`` when unavailable."""
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return converted if np.isfinite(converted) else default

    @classmethod
    def _extract_stage_records(cls, results: Dict, stage: str) -> List[Dict]:
        """Extract explicitly labelled stage records from a result payload.

        ``evaluation_result.json`` contains final OPIM metrics only.  It must
        not be interpreted as memory/reflection/reaction evidence.  This
        helper therefore reads only explicit stage containers or trait-level
        stage objects and intentionally ignores ``perceptual_*`` fields.
        """
        if not isinstance(results, dict):
            return []
        canonical = cls._STAGE_ALIASES.get(stage, stage)
        candidates = []
        containers = ("stages", "stage_results", "pipeline", "analysis_stages")
        for container_name in containers:
            container = results.get(container_name)
            if isinstance(container, dict):
                for key in dict.fromkeys((canonical, stage)):
                    if key in container:
                        candidates.append(container[key])
        # Accept direct, explicitly named stage payloads for callers that do
        # not wrap them in a ``stages`` object.
        for key in dict.fromkeys((canonical, stage)):
            if key in results:
                candidates.append(results[key])

        # A raw memory model state uses these explicit names.
        if canonical == "memory_stream" and isinstance(results.get("memory_pool"), dict):
            pool = results["memory_pool"]
            if isinstance(pool.get("trait_memories"), dict):
                candidates.append(pool["trait_memories"])
        if canonical == "reflection" and isinstance(results.get("reflection_results"), dict):
            candidates.append(results["reflection_results"])

        # Stage objects can also be attached to individual traits.  Do not
        # infer stages from final ``score`` or ``perceptual_*`` fields.
        dimensions = results.get("dimensions")
        if isinstance(dimensions, dict):
            for dimension in dimensions.values():
                if not isinstance(dimension, dict):
                    continue
                traits = dimension.get("traits", dimension)
                if not isinstance(traits, dict):
                    continue
                for trait in traits.values():
                    if not isinstance(trait, dict):
                        continue
                    for key in dict.fromkeys((canonical, stage)):
                        if isinstance(trait.get(key), (dict, list)):
                            candidates.append(trait[key])

        records: List[Dict] = []

        def flatten(value: Any) -> None:
            if isinstance(value, dict):
                # A stage payload may itself wrap dimensions or traits.
                if isinstance(value.get("dimensions"), dict):
                    flatten(value["dimensions"])
                    return
                if isinstance(value.get("traits"), dict):
                    flatten(value["traits"])
                    return
                # Treat dictionaries with explicit numeric/evidence fields as
                # records; otherwise recurse through keyed trait maps.
                record_keys = {
                    "score", "final_score", "evidence_quality",
                    "pattern_relevance", "reliability", "consistency",
                    "metrics", "evidence_summary", "evidence_count",
                }
                if record_keys.intersection(value):
                    records.append(value)
                else:
                    for child in value.values():
                        flatten(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    flatten(child)

        for candidate in candidates:
            flatten(candidate)
        return records

    def adaptive_weight_learning(self, evaluation_results: Dict) -> Dict:
        """自适应权重学习"""
        # 分析当前性能
        performance_metrics = self._analyze_performance(evaluation_results)
        
        # 记录学习历史
        self.learning_history.append({
            'weights': self.current_weights.copy(),
            'performance': performance_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        # 根据历史表现调整学习率
        self._adjust_learning_rate()
        
        # 计算权重调整
        weight_adjustments = self._calculate_weight_adjustments(performance_metrics)

        # 应用权重调整
        old_weights = self._to_builtin(self.current_weights)
        new_weights = self._apply_weight_adjustments(weight_adjustments)

        return self._to_builtin({
            'old_weights': old_weights,
            'new_weights': new_weights,
            'performance': performance_metrics,
            'learning_rate': self.learning_rate
        })

    def _analyze_performance(self, results: Dict) -> Dict:
        """分析模型性能"""
        metrics = {
            'evidence_quality': [],
            'reliability': [],
            'consistency': [],
            'pattern_relevance': []
        }
        
        # Read metric values from explicitly labelled stage records.  The
        # final result's perceptual_energy/stability/coverage are OPIM output
        # metrics, not evidence-quality weights, and must not be substituted.
        for stage in sorted(set(self._STAGE_ALIASES.values())):
            for record in self._extract_stage_records(results, stage):
                source = record.get('metrics', record)
                if not isinstance(source, dict):
                    continue
                for metric in metrics:
                    value = self._optional_float(source.get(metric))
                    if value is not None:
                        metrics[metric].append(value)
        
        # Missing stage evidence is represented as ``None``.  A neutral 0.5
        # would look like an observed score and would cause adaptive learning
        # to change weights even when no stage records were supplied.
        return {
            metric: float(np.mean(scores)) if scores else None
            for metric, scores in metrics.items()
        }

    def _adjust_learning_rate(self):
        """根据历史表现调整学习率"""
        if len(self.learning_history) < 2:
            return
            
        recent_performance = [h['performance'] for h in self.learning_history[-2:]]
        improvements = [
            recent_performance[1][metric] - recent_performance[0][metric]
            for metric in recent_performance[0]
            if recent_performance[0].get(metric) is not None
            and recent_performance[1].get(metric) is not None
        ]
        if not improvements:
            return
        avg_improvement = float(np.mean(improvements))
        
        if avg_improvement < 0:  # 性能下降
            self.learning_rate *= self.decay_rate
        elif avg_improvement < 0.01:  # 性能提升很小
            self.learning_rate *= self.decay_rate
        
        self.learning_rate = max(self.learning_rate, self.min_learning_rate)

    def _calculate_weight_adjustments(self, performance: Dict) -> Dict:
        """计算权重调整"""
        adjustments = {stage: {} for stage in self.current_weights}
        
        for stage, weights in self.current_weights.items():
            for metric, weight in weights.items():
                if metric in performance:
                    if performance[metric] is None:
                        adjustments[stage][metric] = 0.0
                        continue
                    if performance[metric] < self.performance_threshold:
                        # 性能不佳时增加权重
                        adjustments[stage][metric] = self.learning_rate
                    else:
                        # 性能良好时微调
                        adjustments[stage][metric] = -self.learning_rate * 0.5
        
        return adjustments

    def _apply_weight_adjustments(self, adjustments: Dict) -> Dict:
        """应用权重调整"""
        new_weights = {
            stage: {
                metric: max(0.1, min(0.5, weight + adjustments[stage].get(metric, 0)))
                for metric, weight in weights.items()
            }
            for stage, weights in self.current_weights.items()
        }
        
        # 归一化权重
        for stage in new_weights:
            total = sum(new_weights[stage].values())
            new_weights[stage] = {
                metric: weight / total
                for metric, weight in new_weights[stage].items()
            }
        
        self.current_weights = new_weights
        return new_weights

    def analyze_model_weights(self, official_results: Dict, user_results: Dict, comparison_results: Dict) -> Dict:
        """分析模型权重的合理性"""
        # 直接进行本地分析，不依赖API
        
        # 分析各阶段的实际表现
        memory_performance = self._analyze_stage_performance(official_results, user_results, 'memory')
        reflection_performance = self._analyze_stage_performance(official_results, user_results, 'reflection')
        reaction_performance = self._analyze_stage_performance(official_results, user_results, 'reaction')
        
        # 分析差异识别能力
        gap_detection = self._analyze_gap_detection(comparison_results)
        
        # Generate suggestions only from explicit stage records.  A final
        # evaluation file does not expose stage evidence, so retaining the
        # configured weights is safer than deriving pseudo-adjustments from
        # final OPIM metrics.
        performances = {
            'memory_stream': memory_performance,
            'reflection': reflection_performance,
            'reaction': reaction_performance,
        }
        weight_suggestions = {}
        for stage, base in self.current_weights.items():
            perf = performances[stage]
            weight_suggestions[stage] = {}
            for metric, current in base.items():
                adjustment = perf.get(f'{metric}_adj', 0.0) if perf.get('available') else 0.0
                weight_suggestions[stage][metric] = round(
                    (self._optional_float(current, 0.0) or 0.0)
                    + (self._optional_float(adjustment, 0.0) or 0.0), 2
                )
        
        # 归一化权重建议
        for stage in weight_suggestions:
            total = sum(weight_suggestions[stage].values())
            weight_suggestions[stage] = {
                metric: round(weight / total, 2)
                for metric, weight in weight_suggestions[stage].items()
            }
        
        # 生成改进建议
        improvement_suggestions = []
        
        def below(performance: Dict, key: str, threshold: float = 0.6) -> bool:
            value = performance.get(key) if isinstance(performance, dict) else None
            return value is not None and value < threshold

        # 基于记忆流阶段分析
        if memory_performance.get('available') and below(memory_performance, 'evidence_quality_score'):
            improvement_suggestions.append(
                f"记忆流阶段的证据质量评估需要加强 (当前得分: {memory_performance['evidence_quality_score']:.2f}), "
                "建议增加对文本关键信息的提取和筛选"
            )
        if memory_performance.get('available') and below(memory_performance, 'reliability_score'):
            improvement_suggestions.append(
                f"记忆流阶段的可靠性评估不足 (当前得分: {memory_performance['reliability_score']:.2f}), "
                "建议优化评估标准的一致性"
            )
        
        # 基于反思阶段分析
        if reflection_performance.get('available') and below(reflection_performance, 'pattern_relevance_score'):
            improvement_suggestions.append(
                f"反思阶段的模式关联分析不够深入 (当前得分: {reflection_performance['pattern_relevance_score']:.2f}), "
                "建议加强特征词间的关联分析"
            )
        if reflection_performance.get('available') and below(reflection_performance, 'consistency_score'):
            improvement_suggestions.append(
                f"反思阶段的一致性评估需要改进 (当前得分: {reflection_performance['consistency_score']:.2f}), "
                "建议优化评估标准的统一性"
            )
        
        # 基于反应阶段分析
        if gap_detection.get('score') is not None and gap_detection['score'] < 0.6:
            improvement_suggestions.append(
                f"反应阶段对品牌-用户差异的识别不够敏感 (当前得分: {gap_detection['score']:.2f}), "
                "建议加强差异分析的深度"
            )
        
        # 如果没有发现具体问题，添加一般性建议
        if not improvement_suggestions:
            if not any(perf.get('available') for perf in performances.values()):
                improvement_suggestions.append(
                    "未提供显式 memory_stream/reflection/reaction 阶段记录；未执行阶段权重验证。"
                )
            else:
                improvement_suggestions.append("当前权重配置总体合理，建议继续观察模型表现，适时微调")

        available_performances = [
            perf for perf in performances.values() if perf.get('available')
        ]
        def aggregate(metric: str):
            values = [perf.get(metric) for perf in available_performances]
            values = [value for value in values if value is not None]
            return float(np.mean(values)) if values else None

        progression = self._calculate_progression_score(
            memory_performance, reflection_performance, reaction_performance
        )
        
        return self._to_builtin({
            "balance_score": round(aggregate('overall_score'), 2) if aggregate('overall_score') is not None else None,
            "progression_score": round(progression, 2) if progression is not None else None,
            "evidence_focus_score": round(aggregate('evidence_quality_score'), 2) if aggregate('evidence_quality_score') is not None else None,
            "consistency_score": round(aggregate('consistency_score'), 2) if aggregate('consistency_score') is not None else None,
            "gap_detection_score": round(gap_detection['score'], 2) if gap_detection.get('score') is not None else None,
            "stage_availability": {
                stage: bool(perf.get('available')) for stage, perf in performances.items()
            },
            "stage_performance": performances,
            "gap_detection_status": gap_detection.get('status', 'available'),
            "metric_source": "explicit stage records only",
            "weight_suggestions": weight_suggestions,
            "improvement_suggestions": improvement_suggestions
        })

    def _analyze_stage_performance(self, official_results: Dict, user_results: Dict, stage: str) -> Dict:
        """Analyze one explicitly labelled pipeline stage.

        Older code treated ``final_score`` and ``weights[stage]`` in a final
        result as stage evidence.  Public OPIM results expose only final
        perceptual metrics, so that interpretation was invalid.  Missing
        stage records are reported as unavailable and never replaced by zero
        or by a final metric from another stage.
        """
        canonical = self._STAGE_ALIASES.get(stage, stage)
        metric_names = ('evidence_quality', 'pattern_relevance', 'reliability', 'consistency')

        def summarize(results: Dict) -> Dict:
            records = self._extract_stage_records(results, canonical)
            metric_values = {metric: [] for metric in metric_names}
            scores = []
            for record in records:
                source = record.get('metrics', record)
                if not isinstance(source, dict):
                    source = record
                score = self._optional_float(source.get('score', source.get('final_score')))
                if score is not None and 0.0 <= score <= OPIM_SCORE_MAX + 1e-9:
                    scores.append(score)
                for metric in metric_names:
                    value = self._optional_float(source.get(metric))
                    if value is not None and 0.0 <= value <= OPIM_COMPONENT_MAX + 1e-9:
                        metric_values[metric].append(value)

            summary = {
                f'{metric}_score': (float(np.mean(values)) if values else None)
                for metric, values in metric_values.items()
            }
            explicit_metric_values = [value for value in summary.values() if value is not None]
            if scores:
                summary['overall_score'] = float(np.mean(scores))
            elif explicit_metric_values:
                summary['overall_score'] = float(np.mean(explicit_metric_values))
            else:
                summary['overall_score'] = None
            summary['available'] = bool(records)
            summary['sample_count'] = len(records)
            return summary

        official = summarize(official_results)
        user = summarize(user_results)
        combined = {}
        for metric in metric_names:
            values = [official.get(f'{metric}_score'), user.get(f'{metric}_score')]
            values = [value for value in values if value is not None]
            combined[f'{metric}_score'] = float(np.mean(values)) if values else None

            # A weight adjustment is meaningful only when both payloads expose
            # the same explicit metric.
            if official.get(f'{metric}_score') is not None and user.get(f'{metric}_score') is not None:
                gap = abs(official[f'{metric}_score'] - user[f'{metric}_score'])
                combined[f'{metric}_adj'] = 0.1 if gap > 0.2 and official[f'{metric}_score'] < user[f'{metric}_score'] else (-0.1 if gap > 0.2 else 0.0)
            else:
                combined[f'{metric}_adj'] = 0.0

        overall_values = [official.get('overall_score'), user.get('overall_score')]
        overall_values = [value for value in overall_values if value is not None]
        combined['overall_score'] = float(np.mean(overall_values)) if overall_values else None
        combined['available'] = bool(official['available'] or user['available'])
        combined['sample_count'] = official['sample_count'] + user['sample_count']
        combined['stage'] = canonical
        return combined

    def _analyze_gap_detection(self, comparison_results: Dict) -> Dict:
        """Evaluate textual gap descriptions, including SummaryAgent output.

        ``SummaryAgent.compare_evaluations`` nests descriptions under
        ``dimension_analysis[*].significant_differences[*].explanation``;
        looking only at top-level ``expression_gap`` fields silently returned
        a fabricated default score.  When no description is present, mark the
        result unavailable instead.
        """
        if not isinstance(comparison_results, dict):
            return {'score': None, 'adjustment': 0.0, 'status': 'unavailable', 'sample_count': 0}

        descriptions: List[str] = []
        for key in ('expression_gap', 'focus_gap', 'emotion_gap'):
            value = comparison_results.get(key)
            if isinstance(value, str) and value.strip():
                descriptions.append(value.strip())

        dimension_analysis = comparison_results.get('dimension_analysis')
        if isinstance(dimension_analysis, dict):
            for dimension in dimension_analysis.values():
                if not isinstance(dimension, dict):
                    continue
                for item in dimension.get('significant_differences', []):
                    if not isinstance(item, dict):
                        continue
                    explanation = item.get('explanation')
                    if not isinstance(explanation, dict):
                        continue
                    basic = explanation.get('basic_analysis')
                    if isinstance(basic, str) and basic.strip():
                        descriptions.append(basic.strip())
                    detailed = explanation.get('detailed_analysis')
                    if isinstance(detailed, dict):
                        for key in ('expression_gap', 'focus_gap', 'emotion_gap'):
                            value = detailed.get(key)
                            if isinstance(value, str) and value.strip():
                                descriptions.append(value.strip())

        gap_scores = []
        key_phrases = ('而', '但', '然而', '相比', '差异', '不同', '高于', '低于')
        for text in descriptions:
            # Character count works for both Chinese and whitespace-tokenized
            # English descriptions and avoids assuming Chinese word spaces.
            specificity = min(len(text) / 60.0, 1.0)
            depth = sum(1 for phrase in key_phrases if phrase in text) / len(key_phrases)
            gap_scores.append((specificity + depth) / 2)

        if not gap_scores:
            return {'score': None, 'adjustment': 0.0, 'status': 'unavailable', 'sample_count': 0}
        score = float(np.mean(gap_scores))
        return {
            'score': score,
            'adjustment': (score - 0.5) * 0.1,
            'status': 'available',
            'sample_count': len(gap_scores),
        }

    def _calculate_progression_score(self, memory_perf: Dict, reflection_perf: Dict, reaction_perf: Dict) -> Optional[float]:
        """计算阶段进展性得分"""
        if not all(perf.get('available') and perf.get('overall_score') is not None
                   for perf in (memory_perf, reflection_perf, reaction_perf)):
            return None
        # 评估三个阶段的逻辑连贯性
        memory_to_reflection = reflection_perf['overall_score'] - memory_perf['overall_score']
        reflection_to_reaction = reaction_perf['overall_score'] - reflection_perf['overall_score']
        
        # 理想情况下，应该呈现递进关系
        progression_score = (
            (memory_to_reflection > 0) * 0.3 +  # 反思阶段有提升
            (reflection_to_reaction > 0) * 0.3 +  # 反应阶段有提升
            (abs(memory_to_reflection - reflection_to_reaction) < 0.2) * 0.4  # 进展平稳
        )
        
        return progression_score

    def _evaluate_evidence_quality(self, official_results: Dict, user_results: Dict, stage: str) -> Dict:
        """评估显式阶段记录中的证据质量。"""
        values = []
        for results in (official_results, user_results):
            for record in self._extract_stage_records(results, stage):
                source = record.get('metrics', record)
                if isinstance(source, dict):
                    value = self._optional_float(source.get('evidence_quality'))
                    if value is not None and 0.0 <= value <= 1.0:
                        values.append(value)
        score = float(np.mean(values)) if values else None
        return {'score': score, 'adjustment': ((score - 0.5) * 0.1 if score is not None else 0.0), 'status': 'available' if values else 'unavailable'}

    def _evaluate_reliability(self, official_results: Dict, user_results: Dict, stage: str) -> Dict:
        """评估显式阶段记录中的可靠性。"""
        values = []
        for results in (official_results, user_results):
            for record in self._extract_stage_records(results, stage):
                source = record.get('metrics', record)
                if isinstance(source, dict):
                    value = self._optional_float(source.get('reliability'))
                    if value is not None and 0.0 <= value <= 1.0:
                        values.append(value)
        score = float(np.mean(values)) if values else None
        return {'score': score, 'adjustment': ((score - 0.5) * 0.1 if score is not None else 0.0), 'status': 'available' if values else 'unavailable'}

    def _evaluate_consistency(self, memories: List[Dict]) -> float:
        """改进一致性评估"""
        
        def standardize_scores(scores: List[float]) -> List[float]:
            """标准化评分"""
            if not scores:
                return []
            mean = np.mean(scores)
            std = np.std(scores)
            if std == 0:
                return [0.5] * len(scores)
            return [(s - mean) / std for s in scores]
        
        def check_scoring_reliability(memories: List[Dict]) -> float:
            """检验评分可靠性"""
            if len(memories) < 2:
                return 0.0
            
            # 评估时间序列上的一致性
            time_series = sorted(memories, key=lambda x: x['batch_id'])
            score_changes = [
                abs(time_series[i]['score'] - time_series[i-1]['score'])
                for i in range(1, len(time_series))
            ]
            
            return 1.0 - min(np.mean(score_changes), 0.5) * 2
        
        def verify_cross_dimension_consistency(memories: List[Dict]) -> float:
            """验证跨维度一致性"""
            dimension_scores = defaultdict(list)
            for memory in memories:
                if 'memory_key' in memory:
                    dimension = memory['memory_key'].split('_')[0]
                    dimension_scores[dimension].append(memory['score'])
            
            if len(dimension_scores) < 2:
                return 0.0
            
            # 计算维度间的评分相关性
            correlations = []
            for dim1, dim2 in itertools.combinations(dimension_scores.keys(), 2):
                if dimension_scores[dim1] and dimension_scores[dim2]:
                    correlation = np.corrcoef(
                        standardize_scores(dimension_scores[dim1]),
                        standardize_scores(dimension_scores[dim2])
                    )[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
            
            return np.mean(correlations) if correlations else 0.0
        
        # 综合评估
        reliability_score = check_scoring_reliability(memories)
        cross_dim_score = verify_cross_dimension_consistency(memories)
        
        return (reliability_score * 0.6 + cross_dim_score * 0.4)

    def _evaluate_pattern_relevance(self, official_results: Dict, user_results: Dict, stage: str) -> Dict:
        """评估显式阶段记录中的模式关联性。"""
        values = []
        for results in (official_results, user_results):
            for record in self._extract_stage_records(results, stage):
                source = record.get('metrics', record)
                if isinstance(source, dict):
                    value = self._optional_float(source.get('pattern_relevance'))
                    if value is not None and 0.0 <= value <= 1.0:
                        values.append(value)
        score = float(np.mean(values)) if values else None
        return {'score': score, 'adjustment': ((score - 0.5) * 0.1 if score is not None else 0.0), 'status': 'available' if values else 'unavailable'}

    def generate_validation_report(self, analysis_results: Dict) -> str:
        """生成验证报告"""
        analysis_results = analysis_results if isinstance(analysis_results, dict) else {}
        report = ["# 模型权重评估报告\n"]
        report.append(f"生成时间：{datetime.now().isoformat()}\n")

        def display(value: Any) -> str:
            converted = self._optional_float(value)
            return f"{converted:.2f}" if converted is not None else "不可用"
        
        # 总体评分
        report.append("## 总体评分")
        report.append(f"- 权重平衡性：{display(analysis_results.get('balance_score'))}")
        report.append(f"- 阶段进展性：{display(analysis_results.get('progression_score'))}")
        report.append(f"- 证据关注度：{display(analysis_results.get('evidence_focus_score'))}")
        report.append(f"- 一致性评分：{display(analysis_results.get('consistency_score'))}")
        report.append(f"- 差异识别能力：{display(analysis_results.get('gap_detection_score'))}")
        
        # 权重调整建议
        report.append("\n## 权重调整建议")
        
        # 确保使用正确的指标名称
        metric_mapping = {
            'evidence_quality': '证据质量',
            'reliability': '可靠性',
            'consistency': '一致性',
            'pattern_relevance': '模式关联'  # 修正这里的名称
        }
        
        for stage, weights in (analysis_results.get('weight_suggestions') or {}).items():
            report.append(f"\n### {stage}")
            report.append("当前 -> 建议:")
            for metric, weight in weights.items():
                if stage in self.current_weights and metric in self.current_weights[stage]:  # 确保指标存在
                    current = self.current_weights[stage][metric]
                    change = weight - current
                    direction = "增加" if change > 0 else "减少"
                    metric_name = metric_mapping.get(metric, metric)  # 使用中文名称
                    report.append(f"- {metric_name}: {current:.2f} -> {weight:.2f} ({direction} {abs(change):.2f})")
        
        # 改进建议
        report.append("\n## 改进建议")
        for suggestion in (analysis_results.get('improvement_suggestions') or []):
            report.append(f"- {suggestion}")
        
        return "\n".join(report)

    def generate_learning_report(self) -> str:
        """生成学习过程报告"""
        report = ["# 权重学习过程报告\n"]
        
        # 学习历史分析
        report.append("## 学习历史")
        for i, record in enumerate(self.learning_history):
            report.append(f"\n### 迭代 {i+1}")
            report.append(f"时间: {record['timestamp']}")
            report.append("\n性能指标:")
            for metric, value in record['performance'].items():
                # Stage metrics are optional: a final OPIM result may not
                # contain explicit memory/reflection/reaction evidence.
                # Keep that absence visible instead of formatting ``None``
                # as a number or silently replacing it with zero.
                numeric = self._optional_float(value)
                rendered = f"{numeric:.3f}" if numeric is not None else "不可用"
                report.append(f"- {metric}: {rendered}")
        
        # 学习效果分析
        if len(self.learning_history) > 1:
            first = self.learning_history[0]['performance']
            last = self.learning_history[-1]['performance']
            
            report.append("\n## 学习效果")
            for metric in first:
                first_value = self._optional_float(first.get(metric))
                last_value = self._optional_float(last.get(metric))
                if first_value is None or last_value is None:
                    report.append(f"- {metric}: 不可用（缺少两个迭代的有效指标）")
                    continue
                change = last_value - first_value
                direction = "提升" if change > 0 else "下降"
                report.append(f"- {metric}: {abs(change):.3f} ({direction})")
        
        return "\n".join(report)

    def _analyze_single_brand(self, results: Dict) -> Dict:
        """Normalize one brand payload for model-weight and learning stages."""
        if not isinstance(results, dict):
            raise TypeError("brand results must be a dictionary")
        official = results.get("official_results", results.get("official", {})) or {}
        user = results.get("user_results", results.get("user", {})) or {}
        comparison = results.get("comparison_results", results.get("comparison", {})) or {}
        if not isinstance(official, dict) or not isinstance(user, dict) or not isinstance(comparison, dict):
            raise TypeError("official, user, and comparison results must be dictionaries")
        if not official.get("dimensions") and not user.get("dimensions"):
            raise ValueError("insufficient brand evaluation dimensions")
        return {"official_results": official, "user_results": user, "comparison_results": comparison}

    def analyze_multi_brand_weights(self, brand_results: Dict[str, Dict]) -> Dict:
        """多品牌权重分析的主入口"""
        try:
            # 1. 分别分析每个品牌
            brand_analyses = {}
            for brand, results in brand_results.items():
                normalized = self._analyze_single_brand(results)
                single_analysis = self.analyze_model_weights(
                    normalized["official_results"],
                    normalized["user_results"],
                    normalized["comparison_results"],
                )
                # Adaptive learning requires explicit stage metrics.  Final
                # OPIM dimensions alone are insufficient and must not be fed
                # into the stage-weight learner as if they were its inputs.
                learning_source = normalized["official_results"] if normalized["official_results"].get("dimensions") else normalized["user_results"]
                if any(single_analysis.get('stage_availability', {}).values()):
                    learning_results = self.adaptive_weight_learning(learning_source)
                else:
                    learning_results = {
                        'status': 'unavailable',
                        'reason': 'explicit memory_stream/reflection/reaction records are missing',
                    }
                brand_analyses[brand] = {**single_analysis, "learning_results": learning_results}
            
            # 2. 计算跨品牌一致性
            cross_brand_consistency = self._calculate_cross_brand_consistency(brand_analyses)
            
            # 3. 识别共同模式
            common_patterns = self._identify_common_patterns(brand_analyses)
            
            # 4. 生成优化建议
            weight_suggestions = self._generate_weight_suggestions(
                brand_analyses,
                cross_brand_consistency,
                common_patterns
            )
            
            return self._to_builtin({
                'brand_analyses': brand_analyses,
                'cross_brand_consistency': cross_brand_consistency,
                'common_patterns': common_patterns,
                'weight_suggestions': weight_suggestions
            })
        except Exception as e:
            print(f"多品牌分析错误: {str(e)}")
            return {}

    def _calculate_cross_brand_consistency(self, brand_analyses: Dict) -> Dict:
        """计算跨品牌一致性"""
        brand_analyses = brand_analyses if isinstance(brand_analyses, dict) else {}
        consistencies = {
            'evidence_quality': None,
            'reliability': None,
            'consistency': None,
            'pattern_relevance': None
        }

        # Read only the explicit stage-performance fields produced above.
        # Aggregate final ``balance_score``/``evidence_focus_score`` values are
        # intentionally not used as substitutes for stage metrics.
        values_by_metric = defaultdict(list)
        for brand_analysis in brand_analyses.values():
            if not isinstance(brand_analysis, dict):
                continue
            stage_performance = brand_analysis.get('stage_performance', {})
            if not isinstance(stage_performance, dict):
                continue
            for performance in stage_performance.values():
                if not isinstance(performance, dict) or not performance.get('available'):
                    continue
                for metric in consistencies:
                    value = self._optional_float(performance.get(f'{metric}_score'))
                    if value is not None:
                        values_by_metric[metric].append(value)

        for metric in consistencies:
            scores = values_by_metric.get(metric, [])
            if scores:
                mean = float(np.mean(scores))
                std = float(np.std(scores))
                cv = std / abs(mean) if mean else 0.0
                consistencies[metric] = max(0.0, 1.0 - min(cv, 1.0))

        return self._to_builtin(consistencies)

    def _identify_common_patterns(self, brand_analyses: Dict) -> Dict:
        """识别共同模式"""
        patterns = {
            'high_performance_weights': {},  # 高性能品牌共同的权重特征
            'improvement_areas': [],        # 共同需要改进的领域
            'stable_metrics': []            # 跨品牌表现稳定的指标
        }
        
        # 分析高性能品牌的共同特征
        brand_analyses = brand_analyses if isinstance(brand_analyses, dict) else {}
        high_performance_brands = {
            brand: analysis
            for brand, analysis in brand_analyses.items()
            if isinstance(analysis, dict)
            and self._safe_float(analysis.get('balance_score')) > 0.7
        }
        
        if high_performance_brands:
            # 提取共同的权重特征
            for stage in ['memory_stream', 'reflection', 'reaction']:
                patterns['high_performance_weights'][stage] = {}
                for metric in ['evidence_quality', 'reliability', 'consistency', 'pattern_relevance']:
                    weights = []
                    for analysis in high_performance_brands.values():
                        weight_suggestions = analysis.get('weight_suggestions') or {}
                        if not isinstance(weight_suggestions, dict):
                            continue
                        stage_weights = weight_suggestions.get(stage) or {}
                        if not isinstance(stage_weights, dict):
                            continue
                        weight = stage_weights.get(metric)
                        if weight is not None:
                            weights.append(weight)
                    
                    if weights:
                        patterns['high_performance_weights'][stage][metric] = self._safe_float(np.mean(weights))

        # The source implementation reserved these fields but did not fill
        # them.  Keep the same semantics while representing the old sets as
        # JSON-compatible lists.
        return self._to_builtin(patterns)

    def _generate_weight_suggestions(self, brand_analyses: Dict, 
                                   cross_brand_consistency: Dict,
                                   common_patterns: Dict) -> Dict:
        """生成权重优化建议"""
        if not isinstance(cross_brand_consistency, dict):
            cross_brand_consistency = {}
        if not isinstance(common_patterns, dict):
            common_patterns = {}
        # 基于多品牌分析结果生成权重调整建议
        suggested_weights = {
            'memory_stream': {},
            'reflection': {},
            'reaction': {}
        }
        
        # 1. 从高性能品牌学习
        # Start from the complete current schema, then overlay any learned
        # high-performance values.  This prevents a partial pattern dict from
        # causing a KeyError for a missing stage or metric.
        base_weights = {
            stage: dict(weights)
            for stage, weights in self.current_weights.items()
        }
        for stage, weights in (common_patterns.get('high_performance_weights') or {}).items():
            if isinstance(weights, dict) and stage in base_weights:
                base_weights[stage].update(weights)
        
        # 2. 考虑跨品牌一致性
        for stage in suggested_weights:
            suggested_weights[stage] = {}
            for metric, current_weight in base_weights[stage].items():
                raw_consistency = cross_brand_consistency.get(metric)
                # No cross-brand stage data means no evidence for changing a
                # configured weight.  Preserve it exactly in that case.
                if raw_consistency is None:
                    suggested_weights[stage][metric] = self._safe_float(current_weight)
                    continue
                consistency_score = self._safe_float(raw_consistency, 0.5)
                consistency_score = min(max(consistency_score, 0.0), 1.0)
                
                # 如果一致性高，保持当前权重
                # 如果一致性低，向平均值调整
                if consistency_score > 0.8:
                    suggested_weights[stage][metric] = self._safe_float(current_weight)
                else:
                    # 向0.25-0.35的中间值调整
                    target = 0.3
                    adjustment = (target - current_weight) * (1 - consistency_score)
                    suggested_weights[stage][metric] = round(
                        self._safe_float(current_weight) + adjustment, 2
                    )
        
        # 3. 归一化权重
        for stage in suggested_weights:
            total = sum(suggested_weights[stage].values())
            if total <= 0:
                suggested_weights[stage] = dict(self.current_weights.get(stage, {}))
            else:
                suggested_weights[stage] = {
                    metric: round(weight / total, 2)
                    for metric, weight in suggested_weights[stage].items()
                }

        return self._to_builtin(suggested_weights)

    def generate_multi_brand_report(self, analysis_results: Dict) -> str:
        """生成多品牌分析报告"""
        # ``analyze_multi_brand_weights`` may return an empty payload after a
        # failed/empty input.  A report should still be renderable and explain
        # that no analysis data was available instead of raising KeyError.
        if not isinstance(analysis_results, dict):
            analysis_results = {}
        report = ["# 多品牌权重评估报告\n"]
        report.append(f"生成时间：{datetime.now().isoformat()}\n")
        
        # 1. 品牌间一致性分析
        report.append("## 品牌间一致性分析")
        cross_brand_consistency = analysis_results.get('cross_brand_consistency') or {}
        if isinstance(cross_brand_consistency, dict):
            for metric, score in cross_brand_consistency.items():
                rendered = self._optional_float(score)
                report.append(f"- {metric}: {rendered:.2f}" if rendered is not None else f"- {metric}: 不可用")
        else:
            report.append("- 暂无可用数据")
        
        # 2. 各品牌独立分析
        report.append("\n## 品牌独立分析")
        brand_analyses = analysis_results.get('brand_analyses') or {}
        if not isinstance(brand_analyses, dict):
            brand_analyses = {}
        for brand, analysis in brand_analyses.items():
            analysis = analysis if isinstance(analysis, dict) else {}
            report.append(f"\n### {brand}")
            for label, key in (("权重平衡性", "balance_score"), ("证据关注度", "evidence_focus_score"), ("一致性评分", "consistency_score")):
                rendered = self._optional_float(analysis.get(key))
                report.append(f"- {label}：{rendered:.2f}" if rendered is not None else f"- {label}：不可用")
            
            # 添加学习结果
            if 'learning_results' in analysis:
                report.append("\n#### 学习效果")
                learning = analysis['learning_results']
                if not isinstance(learning, dict):
                    learning = {}
                report.append(f"- 学习率：{self._safe_float(learning.get('learning_rate')):.3f}")
                performance = learning.get("performance", {})
                if isinstance(performance, dict):
                    values = [self._safe_float(v) for v in performance.values()]
                else:
                    values = []
                mean_performance = float(np.mean(values)) if values else 0.0
                report.append(f"- 平均性能指标：{mean_performance:.3f}")
        
        # 3. 权重优化建议
        report.append("\n## 权重优化建议")
        suggestions = analysis_results.get('weight_suggestions') or {}
        if not isinstance(suggestions, dict):
            suggestions = {}
        for stage, weights in suggestions.items():
            report.append(f"\n### {stage}")
            if isinstance(weights, dict):
                for metric, weight in weights.items():
                    report.append(f"- {metric}: {self._safe_float(weight):.2f}")
        
        # 4. 添加学习报告
        if analysis_results.get('learning_report'):
            report.append("\n## 学习过程分析")
            report.append(str(analysis_results['learning_report']))
        
        return "\n".join(report)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate OPIM stage weights from JSON results.")
    parser.add_argument("--official", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with open(args.official, encoding="utf-8") as f:
        official_results = json.load(f)
    with open(args.user, encoding="utf-8") as f:
        user_results = json.load(f)
    with open(args.comparison, encoding="utf-8") as f:
        comparison_results = json.load(f)
    validator = EvaluationValidationAgent()
    analysis_results = validator.analyze_model_weights(official_results, user_results, comparison_results)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "model_weights_validation.md"), "w", encoding="utf-8") as f:
        f.write(validator.generate_validation_report(analysis_results))
    with open(os.path.join(args.output_dir, "model_weights_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(validator._to_builtin(analysis_results), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
