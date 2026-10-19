"""
记忆流-反思-反应模型
专门用于文本分析的评分累积和反思系统
基于光学模型实现的文本评价框架
"""

from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict
from datetime import datetime
import sys
import os
import itertools

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from opim.models.shared_memory import SharedMemoryHub
except ImportError:
    from models.shared_memory import SharedMemoryHub
try:
    from opim_metrics import OPIM_WEIGHTS, compute_opim_metrics
except ImportError:  # package import
    from ..opim_metrics import OPIM_WEIGHTS, compute_opim_metrics

class MemoryReflectionModel:
    def __init__(self, luxury_standards: Dict[str, List[str]],
                 shared_memory: Optional[SharedMemoryHub] = None,
                 participant_name: str = "memory_model_text",
                 time_index: str = "batch"):
        self.luxury_standards = luxury_standards
        self.shared_memory = shared_memory or SharedMemoryHub()
        self.shared_memory.add_participant(participant_name)

        # 记忆池
        self.memory_pool = {
            'trait_memories': defaultdict(list),  # 存储特征词评分记忆
            'batch_summaries': [],               # 存储batch级别总结
            'reflection_results': {},            # 存储反思结果
            'all_texts': [],                     # 存储所有文本
            'total_texts': 0,                    # 文本总数
            'reflection_series': {}              # per-batch interference-stage quantities
        }

        # Time index of Eq. (1)-(3): "batch" follows the paper (t = input batch);
        # "observation" uses one time step per observation instead.
        if time_index not in {"batch", "observation"}:
            raise ValueError("time_index must be 'batch' or 'observation'")
        self.time_index = time_index
        # Measurement-stage weights (paper: 40% / 30% / 30%).
        self.reaction_weights = dict(OPIM_WEIGHTS)

    @staticmethod
    def _ordered(memories: List[Dict]) -> List[Dict]:
        return [memory for _, memory in sorted(enumerate(memories), key=lambda item: (item[1].get('batch_id', 0), item[0]))]

    def _paper_metrics(self, memory_key: str, memories: List[Dict], reflection: Dict) -> Dict[str, float]:
        """Evaluate the paper's E_p, S_p, C_p (Eqs. 1-3) for one trait.

        ``time_index == "batch"`` (default, paper): t runs over the input
        batches that contain evidence for the trait.  A(t), L(t), rho(t) are the
        batch means of the propagation-stage quantities; C(t), S(t), R(t) are the
        interference-stage quantities computed on the memory accumulated through
        batch t (``reflection_series``).  Batches whose interference quantities
        are undefined (for example the first batch, where no trend exists yet)
        are excluded instead of being filled with a later aggregate.

        ``time_index == "observation"``: one t per observation, using
        per-observation prefix values with the trait aggregate as fallback.
        """
        ordered = self._ordered(memories)
        if self.time_index == "observation":
            return dict(compute_opim_metrics(
                amplitude=[m.get('wave_amplitude', 0.0) for m in ordered],
                clarity=[m.get('wavelength_purity', 0.0) for m in ordered],
                density=[m.get('photon_density', 0.0) for m in ordered],
                consistency=[m.get('coherence') if m.get('coherence') is not None else reflection.get('coherence', 0.0) for m in ordered],
                salience=[m.get('fringe_clarity') if m.get('fringe_clarity') is not None else reflection.get('fringe_clarity', 0.0) for m in ordered],
                recognizability=[m.get('diffraction_resolution') if m.get('diffraction_resolution') is not None else reflection.get('diffraction_resolution', 0.0) for m in ordered],
            ))
        series = {row['batch_id']: row for row in self.memory_pool.get('reflection_series', {}).get(memory_key, [])}
        groups: Dict = {}
        for memory in ordered:
            groups.setdefault(memory.get('batch_id', 0), []).append(memory)
        A, L, rho, C, S, R = [], [], [], [], [], []
        for batch_id, batch_memories in groups.items():
            row = series.get(batch_id)
            if row is None or any(row.get(name) is None for name in ('coherence', 'fringe_clarity', 'diffraction_resolution')):
                continue
            A.append(float(np.mean([m.get('wave_amplitude', 0.0) for m in batch_memories])))
            L.append(float(np.mean([m.get('wavelength_purity', 0.0) for m in batch_memories])))
            rho.append(float(np.mean([m.get('photon_density', 0.0) for m in batch_memories])))
            C.append(float(row['coherence'])); S.append(float(row['fringe_clarity'])); R.append(float(row['diffraction_resolution']))
        if not A:
            raise ValueError(f"{memory_key}: no batch with complete interference-stage quantities")
        return dict(compute_opim_metrics(amplitude=A, clarity=L, density=rho, consistency=C, salience=S, recognizability=R))

    def memory_stream(self, batch_results: List[Dict]) -> Dict:
        """记忆流处理 - 实现光学模型的记忆流阶段"""
        try:
            if not batch_results:
                raise ValueError("批次结果为空")

            # 保存所有文本
            if 'all_texts' not in self.memory_pool:
                self.memory_pool['all_texts'] = []
                self.memory_pool['total_texts'] = 0

            self.memory_pool['all_texts'].extend(batch_results)
            self.memory_pool['total_texts'] += len(batch_results)

            batch_summary = self._calculate_batch_summary(batch_results)
            self.memory_pool['batch_summaries'].append(batch_summary)

            # 初始化计数器，用于统计参数来源
            api_wave_count = 0
            api_wavelength_count = 0
            missing_param_count = 0
            skipped_memories = 0
            total_traits = 0

            # 更新记忆池 - 使用API返回的光学模型评分
            for result in batch_results:
                if not isinstance(result.get('dimension_scores'), dict):
                    continue

                # 获取API返回的光学模型评分
                optical_scores = result.get('optical_model_scores', {})

                for dimension, scores in result['dimension_scores'].items():
                    if not isinstance(scores, dict):
                        continue

                    # 提取特征词
                    matching_traits = scores.get('matching_traits', [])
                    if not matching_traits and scores.get('explanation'):
                        # 从解释中提取特征词
                        for trait in self.luxury_standards.get(dimension, []):
                            if trait.lower() in scores['explanation'].lower():
                                matching_traits.append(trait)

                    for trait in matching_traits:
                        memory_key = f"{dimension}_{trait}"
                        total_traits += 1

                        # 与图像模型保持一致：wave_amplitude 和
                        # wavelength_purity 必须来自 API。缺失时跳过该记忆，
                        # 不把隐式估计写成可复核的观测值。
                        api_optical_scores = optical_scores.get(memory_key, {})
                        memory_stream_scores = api_optical_scores.get('memory_stream', {})
                        if not isinstance(memory_stream_scores, dict):
                            memory_stream_scores = {}
                        if 'wave_amplitude' not in memory_stream_scores or 'wavelength_purity' not in memory_stream_scores:
                            missing_param_count += 2
                            skipped_memories += 1
                            continue
                        wave_amplitude = float(memory_stream_scores['wave_amplitude'])
                        wavelength_purity = float(memory_stream_scores['wavelength_purity'])
                        if (not np.isfinite(wave_amplitude) or not 0 <= wave_amplitude <= 1 or
                                not np.isfinite(wavelength_purity) or not 0 <= wavelength_purity <= 1):
                            missing_param_count += 2
                            skipped_memories += 1
                            continue
                        api_wave_count += 1
                        api_wavelength_count += 1

                        # 直接计算光子密度
                        photon_density = self._evaluate_photon_density(scores)
                        if photon_density is None:
                            skipped_memories += 1
                            continue

                        # 对反思阶段参数，现在初始化为None，等待反思阶段进行真正计算
                        coherence = None
                        fringe_clarity = None
                        diffraction_resolution = None

                        memory_data = {
                            'score': float(scores.get('score', 0)),
                            'wave_amplitude': wave_amplitude,
                            'wavelength_purity': wavelength_purity,
                            'photon_density': photon_density,
                            'coherence': coherence,
                            'fringe_clarity': fringe_clarity,
                            'diffraction_resolution': diffraction_resolution,
                            'content': result.get('content', ''),
                            'evidence': str(scores.get('evidence', '')),
                            'explanation': str(scores.get('explanation', '')),
                            'batch_id': len(self.memory_pool['batch_summaries']),
                            'memory_key': memory_key
                        }

                        # 只保存有效的记忆
                        if memory_data['score'] >= 0 and (memory_data['evidence'] or memory_data['explanation']):
                            self.memory_pool['trait_memories'][memory_key].append(memory_data)

            # 输出统计信息
            if total_traits > 0:
                print("\n光学参数来源统计:")
                print(f"  wave_amplitude: API={api_wave_count}/{total_traits} ({api_wave_count/total_traits*100:.1f}%), 缺失={missing_param_count/2}/{total_traits} ({missing_param_count/2/total_traits*100:.1f}%)")
                print(f"  wavelength_purity: API={api_wavelength_count}/{total_traits} ({api_wavelength_count/total_traits*100:.1f}%), 缺失={missing_param_count/2}/{total_traits} ({missing_param_count/2/total_traits*100:.1f}%)")
                print(f"  photon_density: API=0/{total_traits} (0.0%), 计算={total_traits}/{total_traits} (100.0%)")
                print(f"  反思阶段参数: 全部使用直接计算")
                print(f"  因缺失必要参数跳过的记忆: {skipped_memories}/{total_traits}")

            # 存储记忆流结果
            self.shared_memory.store(
                key=f"memory_stream_{datetime.now().isoformat()}",
                value=batch_summary,
                metadata={'type': 'memory_stream', 'model': 'optical'}
            )

            return batch_summary

        except Exception as e:
            print(f"记忆流处理错误: {str(e)}")
            return {
                'batch_id': len(self.memory_pool['batch_summaries']) + 1,
                'dimensions': {}
            }

    def reflection(self) -> Dict:
        """Interference stage: coherence C, fringe visibility S, diffraction resolution R.

        The quantities are computed on growing prefixes of each trait's memory,
        ordered by (batch, arrival).  With ``time_index="batch"`` there is one
        prefix per input batch (ending at the batch boundary), matching the
        paper's t index; the ``"observation"`` mode uses one prefix per
        observation.  Quantities that are undefined for a prefix (for example a
        trend over a single batch) stay undefined; nothing is filled with 0.5.
        """
        reflection_results = {}
        reflection_series = {}

        for memory_key, memories in self.memory_pool['trait_memories'].items():
            if not memories:
                continue
            ordered = self._ordered(memories)
            if self.time_index == "batch":
                boundaries = []
                for index, memory in enumerate(ordered):
                    if index + 1 == len(ordered) or ordered[index + 1].get('batch_id', 0) != memory.get('batch_id', 0):
                        boundaries.append(index + 1)
            else:
                boundaries = list(range(1, len(ordered) + 1))
            reflection_values = []
            prefix_updates = []
            series_rows = []
            for end in boundaries:
                prefix = ordered[:end]
                scores = [float(m.get('score', 0.0)) for m in prefix]
                mean_score = float(np.mean(scores)) if scores else 0.0
                std_score = float(np.std(scores)) if len(scores) > 1 else 0.0
                variation_coef = std_score / max(mean_score, 0.01)
                coherence = 1.0 - min(variation_coef, 1.0)
                values = {
                    'coherence': coherence,
                    'fringe_clarity': self._calculate_fringe_clarity(prefix),
                    'diffraction_resolution': self._calculate_diffraction_resolution(prefix),
                }
                for name, value in values.items():
                    if value is None:
                        continue
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        values[name] = None
                        continue
                    if not np.isfinite(value):
                        values[name] = None
                        continue
                    values[name] = float(min(max(value, 0.0), 1.0))
                reflection_values.append(values)
                batch_id = ordered[end - 1].get('batch_id', 0)
                series_rows.append({'batch_id': batch_id, 'prefix_size': end, **values})
                if self.time_index == "batch":
                    prefix_updates.extend((memory, values) for memory in prefix if memory.get('batch_id', 0) == batch_id)
                else:
                    prefix_updates.append((ordered[end - 1], values))

            metric_values = {
                name: [values[name] for values in reflection_values if values.get(name) is not None]
                for name in ('coherence', 'fringe_clarity', 'diffraction_resolution')
            }
            missing_metrics = {name for name, values in metric_values.items() if not values}
            reflection_series[memory_key] = series_rows
            if missing_metrics:
                reflection_results[memory_key] = {
                    'status': 'partial',
                    'missing_metrics': sorted(missing_metrics),
                    'memory_count': len(ordered),
                    'time_index': self.time_index,
                }
                continue

            for memory, values in prefix_updates:
                memory.update({name: value for name, value in values.items() if value is not None})

            coherence = float(np.mean(metric_values['coherence']))
            fringe_clarity = float(np.mean(metric_values['fringe_clarity']))
            diffraction_resolution = float(np.mean(metric_values['diffraction_resolution']))
            evidence_quality = self._evaluate_evidence_quality(memories)
            reliability = self._evaluate_reliability(memories)
            consistency = self._evaluate_consistency(memories)
            pattern_relevance = self._evaluate_pattern_relevance(memories)

            reflection_results[memory_key] = {
                'status': 'complete',
                'memory_count': len(ordered),
                'time_index': self.time_index,
                'time_points': len(boundaries),
                'available_prefixes': {name: len(values) for name, values in metric_values.items()},
                'coherence': float(round(coherence, 2)),
                'fringe_clarity': float(round(fringe_clarity, 2)),
                'diffraction_resolution': float(round(diffraction_resolution, 2)),
                'evidence_quality': float(round(evidence_quality, 2)),
                'reliability': float(round(reliability, 2)),
                'consistency': float(round(consistency, 2)),
                'pattern_relevance': float(round(pattern_relevance, 2)),
                'importance_weight': float(round(coherence * 0.6 + fringe_clarity * 0.4, 2))
            }

        self.memory_pool['reflection_results'] = reflection_results
        self.memory_pool['reflection_series'] = reflection_series

        self.shared_memory.store(
            key=f"reflection_{datetime.now().isoformat()}",
            value=reflection_results,
            metadata={'type': 'reflection', 'model': 'optical', 'time_index': self.time_index}
        )

        return reflection_results

    def response(self) -> Dict:
        """反应处理 - 生成最终评估结果
        基于光学模型并兼容原方法的接口
        """
        try:
            # 使用光学模型生成评估结果
            optical_result = self.optical_reaction()

            # 转换为原接口格式
            final_evaluation = {
                'timestamp': optical_result.get('timestamp', datetime.now().isoformat()),
                'dimensions': {},
                'summary': {
                    'total_texts': optical_result.get('summary', {}).get('total_texts', 0),
                    'analyzed_dimensions': optical_result.get('summary', {}).get('analyzed_dimensions', 0),
                    'key_traits': optical_result.get('summary', {}).get('key_traits', [])
                }
            }

            # 转换维度数据
            for dimension, data in optical_result.get('dimensions', {}).items():
                if data.get('evidence_count', 0) == 0:
                    continue

                dimension_result = {
                    'score': data.get('score', 0),
                    'perceptual_energy': data.get('perceptual_energy', 0),
                    'perceptual_stability': data.get('perceptual_stability', 0),
                    'perceptual_coverage': data.get('perceptual_coverage', 0),
                    'evidence_count': data.get('evidence_count', 0),
                    'traits': {}
                }

                # 处理特征词数据
                for trait, trait_data in data.get('traits', {}).items():
                    if trait_data.get('count', 0) > 0:
                        dimension_result['traits'][trait] = {
                            'score': trait_data.get('score', 0),
                            'count': trait_data.get('count', 0),
                            'perceptual_energy': trait_data.get('perceptual_energy', 0),
                            'perceptual_stability': trait_data.get('perceptual_stability', 0),
                            'perceptual_coverage': trait_data.get('perceptual_coverage', 0),
                            'best_evidence': trait_data.get('best_evidence', {})
                        }

                final_evaluation['dimensions'][dimension] = dimension_result

            # 存储评估结果
            self.shared_memory.store(
                key=f"response_{datetime.now().isoformat()}",
                value=final_evaluation,
                metadata={'type': 'response', 'model': 'optical'}
            )

            return final_evaluation

        except Exception as e:
            print(f"生成最终评估结果时出错: {str(e)}")

            # 如果光学模型处理失败，尝试使用原始方法
            fallback_result = self._fallback_response()

            # 存储评估结果
            self.shared_memory.store(
                key=f"response_fallback_{datetime.now().isoformat()}",
                value=fallback_result,
                metadata={'type': 'response_fallback', 'model': 'original'}
            )

            return fallback_result

    def _fallback_response(self) -> Dict:
        """Compatibility fallback that still uses the paper equations."""
        return self.calculate_final_scores()

    def calculate_final_scores(self) -> Dict:
        """计算最终得分，使用光学模型的公式"""
        try:
            final_scores = {
                'timestamp': datetime.now().isoformat(),
                'total_texts': len(self.memory_pool.get('all_texts', [])),
                'dimensions': {}
            }

            # 获取反思结果
            reflection_results = self.memory_pool.get('reflection_results', {})

            # 处理每个维度
            for dim_name, traits in self.luxury_standards.items():
                final_scores['dimensions'][dim_name] = {}

                # 处理每个特征词
                for trait in traits:
                    memory_key = f"{dim_name}_{trait}"
                    memories = self.memory_pool['trait_memories'].get(memory_key, [])

                    final_scores['dimensions'][dim_name][trait] = {
                        'final_score': None,
                        'evidence_summary': '',
                        'weights': {},
                        'optical_metrics': {}
                    }

                    if (memories and memory_key in reflection_results and
                            reflection_results[memory_key].get('status') == 'complete'):
                        reflection = reflection_results[memory_key]
                        metrics = self._paper_metrics(memory_key, memories, reflection)
                        wave_amplitude = np.mean([m.get('wave_amplitude', 0.0) for m in memories])
                        wavelength_purity = np.mean([m.get('wavelength_purity', 0.0) for m in memories])
                        photon_density = np.mean([m.get('photon_density', 0.0) for m in memories])
                        coherence = reflection.get('coherence', 0.0)
                        fringe_clarity = reflection.get('fringe_clarity', 0.0)
                        diffraction_resolution = reflection.get('diffraction_resolution', 0.0)
                        final_score = metrics['score']

                        # 构建证据摘要
                        sorted_memories = sorted(
                            memories,
                            key=lambda x: len(x.get('evidence', '')),
                            reverse=True
                        )
                        evidence_summary = '; '.join(
                            m.get('evidence', '')
                            for m in sorted_memories[:3]
                            if m.get('evidence')
                        )

                        # 保存计算结果
                        final_scores['dimensions'][dim_name][trait].update({
                            'final_score': round(final_score, 6),
                            'evidence_summary': evidence_summary,
                            'weights': dict(self.reaction_weights),
                            'optical_metrics': {
                                'memory_stream': {
                                    'wave_amplitude': round(wave_amplitude, 2),
                                    'wavelength_purity': round(wavelength_purity, 2),
                                    'photon_density': round(photon_density, 2)
                                },
                                'reflection': {
                                    'coherence': round(coherence, 2),
                                    'fringe_clarity': round(fringe_clarity, 2),
                                    'diffraction_resolution': round(diffraction_resolution, 2)
                                },
                                'reaction': {
                                    'perceptual_energy': round(metrics['perceptual_energy'], 4),
                                    'perceptual_stability': round(metrics['perceptual_stability'], 4),
                                    'perceptual_coverage': round(metrics['perceptual_coverage'], 4),
                                }
                            }
                        })

            return final_scores

        except Exception as e:
            print(f"计算最终得分时出错: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'total_texts': len(self.memory_pool.get('all_texts', [])),
                'dimensions': {}
            }

    def _evaluate_photon_density(self, scores: Dict) -> Optional[float]:
        """计算光子密度 - 评估信息密度和质量"""
        try:
            # 提取所需数据
            evidence = str(scores.get('evidence', ''))
            explanation = str(scores.get('explanation', ''))
            matching_traits = scores.get('matching_traits', [])

            # 有效信息密度
            if not evidence and not explanation:
                effective_density = 0.5
            else:
                combined_text = (evidence + ' ' + explanation).lower()
                words = combined_text.split()

                # 统计实质性词汇 (长度>3的非停用词)
                content_words = [w for w in words if len(w) > 3]
                content_ratio = len(content_words) / max(len(words), 1)

                # 计算有效信息密度
                effective_density = min(content_ratio * 1.5, 1.0)

            # 信息质量因子
            trait_count = len(matching_traits)
            trait_quality = min(trait_count / 3, 1.0) if trait_count > 0 else 0

            # 证据质量
            evidence_quality = min(len(evidence) / 50, 1.0) if evidence else 0

            # 信息质量因子 = 特征词质量 * 0.6 + 证据质量 * 0.4
            info_quality_factor = trait_quality * 0.6 + evidence_quality * 0.4

            # 光子密度 = 有效信息密度 * 0.7 + 信息质量因子 * 0.3
            photon_density = effective_density * 0.7 + info_quality_factor * 0.3

            return min(photon_density, 1.0)  # 确保结果在0-1范围内

        except Exception as e:
            print(f"计算光子密度时出错: {str(e)}")
            return None

    def _calculate_fringe_clarity(self, memories: List[Dict]) -> float:
        """计算干涉条纹清晰度 - 模式显著性

        公式: 条纹清晰度 = 趋势强度 × 模式重复指数
        """
        if not memories or len(memories) < 2:
            return None

        try:
            # 根据batch_id排序记忆
            sorted_memories = sorted(memories, key=lambda m: m.get('batch_id', 0))

            # 计算趋势强度
            batches = [m.get('batch_id', 0) for m in sorted_memories]
            scores = [m.get('score', 0) for m in sorted_memories]

            # 使用线性回归计算趋势
            if len(set(batches)) >= 2:  # 至少需要两个不同的batch
                trend_coefficient = np.polyfit(batches, scores, 1)[0]
                trend_strength = min(abs(trend_coefficient) * 10, 1.0)
            else:
                return None

            # 计算模式重复指数 (简化为分数的一致性)
            score_std = np.std(scores) if len(scores) > 1 else 0
            score_mean = np.mean(scores) if scores else 0.5
            consistency = 1 - min(score_std / max(score_mean, 0.01), 1.0)
            pattern_repetition = consistency

            # 应用公式
            return trend_strength * pattern_repetition

        except Exception as e:
            print(f"计算干涉条纹清晰度时出错: {str(e)}")
            return None

    def _calculate_diffraction_resolution(self, memories: List[Dict]) -> float:
        """计算衍射分辨率 - 信息区分能力

        公式: 衍射分辨率 = 语义边界清晰度 × 特征区分度
        """
        if not memories:
            return None

        try:
            # 语义边界清晰度 - 评估解释中的概念区分度
            descriptions = [
                text for memory in memories
                for text in (memory.get('explanation', ''), memory.get('evidence', ''))
                if text
            ]
            if not descriptions:
                return None

            # 简化为解释文本的平均长度与理想长度的比值
            avg_length = sum(len(text) for text in descriptions) / len(descriptions)
            semantic_boundary = min(avg_length / 200, 1.0)

            # 特征区分度 - 评分的方差
            scores = [m.get('score', 0) for m in memories]
            score_variance = np.var(scores) if len(scores) > 1 else 0.0
            feature_distinction = min(score_variance * 4, 1.0)

            # 应用公式
            return semantic_boundary * feature_distinction

        except Exception as e:
            print(f"计算衍射分辨率时出错: {str(e)}")
            return None

    def _calculate_batch_summary(self, batch_results: List[Dict]) -> Dict:
        """计算batch的阶段性总结
        保持原接口实现
        """
        summary = {
            'batch_id': len(self.memory_pool['batch_summaries']) + 1,
            'dimensions': defaultdict(lambda: {
                'traits': defaultdict(list),
                'sample_count': 0
            })
        }

        try:
            # 收集batch中的评分和证据
            for result in batch_results:
                for dimension, scores in result.get('dimension_scores', {}).items():
                    summary['dimensions'][dimension]['sample_count'] += 1

                    # 确保scores是字典类型
                    if isinstance(scores, dict):
                        matching_traits = scores.get('matching_traits', [])
                        for trait in matching_traits:
                            summary['dimensions'][dimension]['traits'][trait].append({
                                'score': scores.get('score', 0),
                                'content': result.get('content', ''),
                                'evidence': scores.get('evidence', ''),
                                'explanation': scores.get('explanation', '')
                            })

        except Exception as e:
            print(f"计算batch总结时出错: {str(e)}")

        return summary

    def _evaluate_evidence_quality(self, memories: List[Dict]) -> float:
        """评估证据质量 - 保持与原接口兼容"""
        if not memories:
            return 0.0

        quality_scores = []
        for memory in memories:
            explanation_text = memory.get('explanation', '')

            # 具体性评分
            specificity = min(len(explanation_text.split()) / 10, 1.0)

            # 相关性评分
            relevance = 1.0 if explanation_text and memory.get('content') else 0.0

            # 完整性评分
            completeness = 1.0 if all(k in memory for k in ['score', 'content', 'explanation']) else 0.5

            quality_score = (specificity * 0.4 + relevance * 0.3 + completeness * 0.3)
            quality_scores.append(quality_score)

        return sum(quality_scores) / len(quality_scores)

    def _evaluate_reliability(self, memories: List[Dict]) -> float:
        """评估可靠性 - 保持与原接口兼容"""
        if not memories:
            return 0.0

        scores = [m['score'] for m in memories]
        scores_array = np.array(scores)

        # 分数稳定性
        score_stability = 1.0 - min(np.std(scores_array), 0.5) * 2

        # 样本充分性
        sample_confidence = min(len(scores) / 10, 1.0)

        # 解释充分性
        explanation_strength = np.mean([
            len(m.get('explanation', '').split()) / 20
            for m in memories
        ])

        return (score_stability * 0.4 +
                sample_confidence * 0.3 +
                explanation_strength * 0.3)

    def _evaluate_consistency(self, memories: List[Dict]) -> float:
        """评估一致性 - 保持与原接口兼容"""
        if not memories:
            return 0.0

        # 解释表述一致性
        explanation_texts = [m.get('explanation', '') for m in memories]
        unique_concepts = set()
        for text in explanation_texts:
            unique_concepts.update(text.split())
        concept_consistency = min(len(unique_concepts) / (len(explanation_texts) * 5), 1.0)

        # 评分标准一致性
        scores = np.array([m['score'] for m in memories])
        score_consistency = 1.0 - min(np.std(scores), 0.5) * 2

        # 解释支持度一致性
        support_levels = [
            len(m.get('explanation', '').split()) / len(m.get('content', '').split())
            if m.get('content') and m.get('explanation')
            else 0.0
            for m in memories
        ]
        support_consistency = np.mean(support_levels) if support_levels else 0.0

        return (concept_consistency * 0.4 +
                score_consistency * 0.3 +
                support_consistency * 0.3)

    def _identify_patterns(self) -> Dict:
        """识别特征词之间的关联模式 - 保持原接口实现"""
        patterns = {}

        # 遍历所有特征词对
        for key1, memories1 in self.memory_pool['trait_memories'].items():
            patterns[key1] = {}
            for key2, memories2 in self.memory_pool['trait_memories'].items():
                if key1 != key2:
                    # 计算共现率
                    cooccurrence = self._calculate_cooccurrence(memories1, memories2)
                    # 计算相关性
                    correlation = self._calculate_correlation(memories1, memories2)
                    patterns[key1][key2] = {
                        'cooccurrence': cooccurrence,
                        'correlation': correlation
                    }

        return patterns

    def _evaluate_pattern_relevance(self, memories: List[Dict]) -> float:
        """评估模式关联性 - 保持原接口实现"""
        if not memories:
            return 0.0

        try:
            # 1. 维度内特征词关联分析
            trait_correlations = []
            for dimension, traits in self.luxury_standards.items():
                trait_memories = {
                    trait: [m for m in memories if f"{dimension}_{trait}" in m.get('memory_key', '')]
                    for trait in traits
                }

                # 计算特征词间的相关性
                for t1, t2 in itertools.combinations(traits, 2):
                    if trait_memories[t1] and trait_memories[t2]:
                        correlation = self._calculate_correlation(
                            trait_memories[t1],
                            trait_memories[t2]
                        )
                        trait_correlations.append(correlation)

            # 2. 维度间关联分析
            dimension_correlations = []
            for dim1, dim2 in itertools.combinations(self.luxury_standards.keys(), 2):
                dim1_memories = [m for m in memories if dim1 in m.get('memory_key', '')]
                dim2_memories = [m for m in memories if dim2 in m.get('memory_key', '')]

                if dim1_memories and dim2_memories:
                    correlation = self._calculate_correlation(
                        dim1_memories,
                        dim2_memories
                    )
                    dimension_correlations.append(correlation)

            # 3. 时序模式分析
            temporal_patterns = self._analyze_temporal_patterns(memories)

            # 计算综合得分
            trait_score = np.mean(trait_correlations) if trait_correlations else 0
            dimension_score = np.mean(dimension_correlations) if dimension_correlations else 0

            return (
                trait_score * 0.4 +           # 特征词关联权重
                dimension_score * 0.4 +       # 维度关联权重
                temporal_patterns * 0.2        # 时序模式权重
            )

        except Exception as e:
            print(f"评估模式关联性时出错: {str(e)}")
            return 0.0

    def _analyze_temporal_patterns(self, memories: List[Dict]) -> float:
        """分析时序模式 - 保持原接口实现"""
        try:
            # 按batch_id分组
            batch_groups = defaultdict(list)
            for memory in memories:
                batch_groups[memory['batch_id']].append(memory)

            if len(batch_groups) < 2:
                return 0.0

            # 计算批次间的趋势
            batch_scores = []
            for batch_id in sorted(batch_groups.keys()):
                batch_avg = np.mean([m['score'] for m in batch_groups[batch_id]])
                batch_scores.append(batch_avg)

            # 评估趋势的稳定性
            trend_stability = 1.0 - min(np.std(batch_scores), 0.5) * 2

            # 评估趋势的方向性
            trend_direction = np.polyfit(range(len(batch_scores)), batch_scores, 1)[0]
            direction_score = min(abs(trend_direction) * 5, 1.0)  # 归一化趋势强度

            return (trend_stability * 0.6 + direction_score * 0.4)

        except Exception as e:
            print(f"分析时序模式时出错: {str(e)}")
            return 0.0

    def _calculate_cooccurrence(self, memories1: List[Dict], memories2: List[Dict]) -> float:
        """计算两组记忆的共现率 - 保持原接口实现"""
        if not memories1 or not memories2:
            return 0.0

        # 获取两组记忆的batch_ids
        batches1 = set(m['batch_id'] for m in memories1)
        batches2 = set(m['batch_id'] for m in memories2)

        # 计算共现率
        cooccurrence = len(batches1 & batches2) / len(batches1 | batches2)
        return cooccurrence

    def _calculate_correlation(self, memories1: List[Dict], memories2: List[Dict]) -> float:
        """计算两组记忆的相关性 - 保持原接口实现"""
        if not memories1 or not memories2:
            return 0.0

        # 获取共同的batch_ids
        common_batches = set(m1['batch_id'] for m1 in memories1) & set(m2['batch_id'] for m2 in memories2)
        if len(common_batches) < 2:  # 需要至少2个样本才能计算相关性
            return 0.0

        try:
            # 计算相关性
            scores1 = []
            scores2 = []
            for batch_id in common_batches:
                batch_scores1 = [m['score'] for m in memories1 if m['batch_id'] == batch_id]
                batch_scores2 = [m['score'] for m in memories2 if m['batch_id'] == batch_id]

                if batch_scores1 and batch_scores2:  # 确保两组都有分数
                    scores1.append(np.mean(batch_scores1))
                    scores2.append(np.mean(batch_scores2))

            # 检查样本数量
            if len(scores1) < 2 or len(scores2) < 2:
                return 0.0

            # 检查是否有变化
            if np.std(scores1) == 0 or np.std(scores2) == 0:
                return 0.0

            # 计算相关系数
            correlation = np.corrcoef(scores1, scores2)[0, 1]

            # 处理无效值
            if np.isnan(correlation) or np.isinf(correlation):
                return 0.0

            return max(0, correlation)  # 只考虑正相关

        except Exception as e:
            print(f"计算相关性时出错: {str(e)}")
            return 0.0

    def reset(self):
        """重置模型状态，用于处理新的文件"""
        # 重置记忆池
        self.memory_pool = {
            'trait_memories': defaultdict(list),  # 存储特征词评分记忆
            'batch_summaries': [],               # 存储batch级别总结
            'reflection_results': {},            # 存储反思结果
            'all_texts': [],                     # 存储所有文本
            'total_texts': 0,                    # 文本总数
            'reflection_series': {}
        }

        # 通知共享内存
        self.shared_memory.store(
            key=f"model_reset_{datetime.now().isoformat()}",
            value={"status": "reset_completed", "model": "optical"},
            metadata={'type': 'model_reset'}
        )

        return True

    def optical_reaction(self) -> Dict:
        """光学模型的反应阶段处理

        生成基于论文 E_p、S_p、C_p 方程的最终评估结果。
        """
        optical_evaluation = {
            'timestamp': datetime.now().isoformat(),
            'dimensions': {},
            'summary': {
                'total_texts': self.memory_pool['total_texts'],
                'analyzed_dimensions': 0,
                'key_traits': []
            }
        }

        try:
            # 确保反思阶段已完成
            if not self.memory_pool['reflection_results']:
                self.reflection()

            # 按维度计算指标
            for dimension, traits in self.luxury_standards.items():
                dimension_data = {
                    'traits': {},
                    'perceptual_energy': 0,
                    'perceptual_stability': 0,
                    'perceptual_coverage': 0,
                    'score': 0,
                    'evidence_count': 0
                }

                valid_traits = []
                total_amplitude = 0
                trait_intensities = []

                # 计算每个特征词的指标
                for trait in traits:
                    memory_key = f"{dimension}_{trait}"
                    memories = self.memory_pool['trait_memories'].get(memory_key, [])
                    reflection = self.memory_pool['reflection_results'].get(memory_key, {})

                    if not memories or not reflection:
                        continue

                    if reflection.get('status') != 'complete':
                        continue
                    try:
                        metrics = self._paper_metrics(memory_key, memories, reflection)
                    except ValueError:
                        continue
                    trait_score = metrics['score']

                    # 找到最佳证据
                    best_evidence = None
                    if memories:
                        best_memory = max(memories, key=lambda m: m.get('wave_amplitude', 0))
                        best_evidence = {
                            'content': best_memory.get('content', ''),
                            'evidence': best_memory.get('evidence', ''),
                            'explanation': best_memory.get('explanation', '')
                        }

                    # 保存特征词数据
                    trait_data = {
                        'score': round(trait_score, 2),
                        'count': len(memories),
                        'perceptual_energy': round(metrics['perceptual_energy'], 4),
                        'perceptual_stability': round(metrics['perceptual_stability'], 4),
                        'perceptual_coverage': round(metrics['perceptual_coverage'], 4),
                        'best_evidence': best_evidence
                    }

                    dimension_data['traits'][trait] = trait_data
                    dimension_data['evidence_count'] += len(memories)

                    # 累积维度数据
                    valid_traits.append(trait)
                    trait_intensities.append(trait_score)

                # 计算维度总体指标
                if valid_traits:
                    dimension_data['perceptual_energy'] = round(np.mean([
                        data['perceptual_energy'] for data in dimension_data['traits'].values()
                    ]), 4)

                    dimension_data['perceptual_stability'] = round(np.mean([
                        data['perceptual_stability'] for data in dimension_data['traits'].values()
                    ]), 4)
                    dimension_data['perceptual_coverage'] = round(np.mean([
                        data['perceptual_coverage'] for data in dimension_data['traits'].values()
                    ]), 4)
                    dimension_data['score'] = round(np.mean(trait_intensities), 4)

                optical_evaluation['dimensions'][dimension] = dimension_data

            # 更新摘要统计
            analyzed_dimensions = [d for d, data in optical_evaluation['dimensions'].items()
                                  if data['evidence_count'] > 0]
            optical_evaluation['summary']['analyzed_dimensions'] = len(analyzed_dimensions)

            # 计算关键特征词（按得分排序）
            all_traits = []
            for dimension, data in optical_evaluation['dimensions'].items():
                for trait, trait_data in data['traits'].items():
                    if trait_data['count'] > 0:
                        all_traits.append({
                            'dimension': dimension,
                            'trait': trait,
                            'score': trait_data['score'],
                            'count': trait_data['count']
                        })

            # 按分数降序排序并取前10个
            key_traits = sorted(all_traits, key=lambda x: x['score'], reverse=True)[:10]
            optical_evaluation['summary']['key_traits'] = key_traits

            # 存储到共享内存
            self.shared_memory.store(
                key=f"optical_reaction_{datetime.now().isoformat()}",
                value=optical_evaluation,
                metadata={'type': 'optical_reaction', 'model': 'optical'}
            )

            return optical_evaluation

        except Exception as e:
            print(f"光学模型反应阶段处理错误: {str(e)}")
            return optical_evaluation
