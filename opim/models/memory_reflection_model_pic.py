"""
记忆流-反思-反应模型
专门用于图像分析的评分累积和反思系统
基于光学模型实现的图像评价框架
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
                 participant_name: str = "memory_model_pic",
                 time_index: str = "batch"):
        self.luxury_standards = luxury_standards
        self.shared_memory = shared_memory or SharedMemoryHub()
        self.shared_memory.add_participant(participant_name)

        # 记忆池
        self.memory_pool = {
            'trait_memories': defaultdict(list),  # 存储特征词评分记忆
            'batch_summaries': [],               # 存储batch级别总结
            'reflection_results': {},            # 存储反思结果
            'all_images': [],                     # 存储所有图像
            'total_images': 0,                    # 图像总数
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
        """记忆流处理 - 实现光学模型的记忆流阶段

        处理批次结果，从API获取光学参数或通过计算补充，并存储到记忆池。
        wave_amplitude和wavelength_purity只能从API获取，photon_density始终通过计算获得。
        """
        try:
            if not batch_results:
                raise ValueError("批次结果为空")

            # 保存所有图像
            if 'all_images' not in self.memory_pool:
                self.memory_pool['all_images'] = []
                self.memory_pool['total_images'] = 0

            self.memory_pool['all_images'].extend(batch_results)
            self.memory_pool['total_images'] += len(batch_results)

            batch_summary = self._calculate_batch_summary(batch_results)
            self.memory_pool['batch_summaries'].append(batch_summary)

            # 初始化计数器，用于统计参数来源
            api_param_count = 0
            missing_param_count = 0
            total_params = 0
            skipped_memories = 0
            total_memories = 0

            # 更新记忆池 - 使用API返回的光学模型评分并补充计算
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
                        total_memories += 1
                        memory_key = f"{dimension}_{trait}"

                        # 从API获取光学模型评分
                        api_optical_scores = optical_scores.get(memory_key, {})
                        memory_stream_scores = api_optical_scores.get('memory_stream', {})

                        # 提取API返回的wave_amplitude和wavelength_purity
                        wave_amplitude = None
                        wavelength_purity = None

                        # 必须从API获取光学参数
                        if memory_stream_scores:
                            if 'wave_amplitude' in memory_stream_scores:
                                wave_amplitude = float(memory_stream_scores.get('wave_amplitude'))
                                api_param_count += 1
                                total_params += 1
                            else:
                                missing_param_count += 1
                                total_params += 1
                                print(f"警告: API未返回{memory_key}的必要参数wave_amplitude")

                            if 'wavelength_purity' in memory_stream_scores:
                                wavelength_purity = float(memory_stream_scores.get('wavelength_purity'))
                                api_param_count += 1
                                total_params += 1
                            else:
                                missing_param_count += 1
                                total_params += 1
                                print(f"警告: API未返回{memory_key}的必要参数wavelength_purity")
                        else:
                            missing_param_count += 2
                            total_params += 2
                            print(f"警告: API未返回{memory_key}的必要光学参数")

                        # 跳过没有必要光学参数的记忆；与文本模型一致，越界或非有限值也视为缺失
                        if wave_amplitude is None or wavelength_purity is None:
                            skipped_memories += 1
                            continue
                        if (not np.isfinite(wave_amplitude) or not 0 <= wave_amplitude <= 1 or
                                not np.isfinite(wavelength_purity) or not 0 <= wavelength_purity <= 1):
                            print(f"警告: {memory_key} 的 wave_amplitude/wavelength_purity 不在 [0,1] 内，已跳过")
                            missing_param_count += 2
                            skipped_memories += 1
                            continue

                        # photon_density始终使用计算方法获取
                        photon_density = self._evaluate_photon_density(scores)
                        if photon_density is None:
                            skipped_memories += 1
                            continue

                        # 反思阶段参数初始化为None，等待反思阶段计算
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
                            'image_path': result.get('image_path', ''),
                            'image_url': result.get('image_url', ''),
                            'evidence': str(scores.get('evidence', '')),
                            'explanation': str(scores.get('explanation', '')),
                            'batch_id': len(self.memory_pool['batch_summaries']),
                            'memory_key': memory_key,
                            'api_provided': True  # 所有光学参数现在都必须由API提供
                        }

                        # 只保存有效的记忆
                        if memory_data['score'] >= 0 and (memory_data['evidence'] or memory_data['explanation']):
                            self.memory_pool['trait_memories'][memory_key].append(memory_data)

            # 输出统计信息
            if total_params > 0:
                print("\n光学参数来源统计:")
                print(f"  API提供的参数: {api_param_count}/{total_params} ({api_param_count/total_params*100:.1f}%)")
                print(f"  缺失的参数: {missing_param_count}/{total_params} ({missing_param_count/total_params*100:.1f}%)")
                print(f"  反思阶段参数: 100% 通过直接计算得出")

            if total_memories > 0:
                print("\n记忆处理统计:")
                print(f"  成功处理的记忆: {total_memories - skipped_memories}/{total_memories} ({(total_memories - skipped_memories)/total_memories*100:.1f}%)")
                print(f"  因缺少必要光学参数而跳过的记忆: {skipped_memories}/{total_memories} ({skipped_memories/total_memories*100:.1f}%)")

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
                amplitudes = [float(m.get('wave_amplitude', 0.0)) for m in prefix]
                coherence = 1.0 - min(variation_coef, 1.0)
                if len(amplitudes) > 1:
                    coherence *= 1.0 - min(float(np.var(amplitudes)), 0.5)
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

            reflection_results[memory_key] = {
                'status': 'complete',
                'memory_count': len(ordered),
                'time_index': self.time_index,
                'time_points': len(boundaries),
                'available_prefixes': {name: len(values) for name, values in metric_values.items()},
                'coherence': float(round(coherence, 2)),
                'fringe_clarity': float(round(fringe_clarity, 2)),
                'diffraction_resolution': float(round(diffraction_resolution, 2)),
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
        基于光学模型
        """
        try:
            # 使用光学模型生成评估结果
            optical_result = self.optical_reaction()

            # 转换为标准接口格式
            final_evaluation = {
                'timestamp': optical_result.get('timestamp', datetime.now().isoformat()),
                'dimensions': {},
                'summary': {
                    'total_images': optical_result.get('summary', {}).get('total_images', 0),
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
            print(f"尝试重新计算光学反应")

            try:
                # 如果出现错误，尝试重新计算
                self.reflection()  # 确保反思阶段已完成
                optical_result = self.optical_reaction()

                # 简化输出，仅返回原始的optical_result
                self.shared_memory.store(
                    key=f"response_minimal_{datetime.now().isoformat()}",
                    value=optical_result,
                    metadata={'type': 'response_minimal', 'model': 'optical'}
                )

                return optical_result
            except Exception as inner_e:
                print(f"再次尝试失败: {str(inner_e)}")
                return {
                    'timestamp': datetime.now().isoformat(),
                    'dimensions': {},
                    'error': f"光学模型处理失败: {str(e)}, 再次尝试失败: {str(inner_e)}"
                }

    def calculate_final_scores(self) -> Dict:
        """计算最终得分，使用光学模型的公式

        结合API提供的光学参数和计算方法得出最终得分
        """
        try:
            final_scores = {
                'timestamp': datetime.now().isoformat(),
                'total_images': len(self.memory_pool.get('all_images', [])),
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
                        'optical_metrics': {},
                        'api_contribution': 0.0  # 记录API贡献度
                    }

                    if (memories and memory_key in reflection_results and
                            reflection_results[memory_key].get('status') == 'complete'):
                        reflection = reflection_results[memory_key]
                        metrics = self._paper_metrics(memory_key, memories, reflection)
                        final_score = metrics['score']
                        wave_amplitude = np.mean([m.get('wave_amplitude', 0.0) for m in memories])
                        wavelength_purity = np.mean([m.get('wavelength_purity', 0.0) for m in memories])
                        photon_density = np.mean([m.get('photon_density', 0.0) for m in memories])
                        coherence = reflection.get('coherence', 0.0)
                        fringe_clarity = reflection.get('fringe_clarity', 0.0)
                        diffraction_resolution = reflection.get('diffraction_resolution', 0.0)

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
                            'api_contribution': 0.0,
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
                'total_images': len(self.memory_pool.get('all_images', [])),
                'dimensions': {}
            }

    def _evaluate_photon_density(self, scores: Dict) -> Optional[float]:
        """计算光子密度 - 评估图像信息密度和视觉质量

        光子密度代表图像信息的密度和视觉质量，始终通过计算获得。

        公式: 光子密度 = 视觉信息密度 * 0.7 + 图像质量因子 * 0.3
        """
        try:
            # 提取所需数据
            evidence = str(scores.get('evidence', ''))
            explanation = str(scores.get('explanation', ''))
            matching_traits = scores.get('matching_traits', [])

            # 视觉信息密度
            if not evidence and not explanation:
                visual_density = 0.5
            else:
                combined_text = (evidence + ' ' + explanation).lower()
                words = combined_text.split()

                # 视觉关键词列表
                visual_keywords = ['color', 'image', 'shape', 'light', 'pattern',
                                  'texture', 'design', '颜色', '图像', '形状', '光线',
                                  '图案', '纹理', '设计']

                # 统计视觉相关词汇
                visual_words = [w for w in words if len(w) > 3 and any(vt in w.lower() for vt in visual_keywords)]
                visual_ratio = len(visual_words) / max(len(words), 1)

                # 计算视觉信息密度
                visual_density = min(visual_ratio * 2.0, 1.0)

            # 图像质量因子
            trait_count = len(matching_traits)
            trait_quality = min(trait_count / 3, 1.0) if trait_count > 0 else 0

            # 证据视觉质量 - 更偏重于视觉描述的质量
            visual_keywords = ['color', 'image', 'shape', 'light', 'pattern',
                              'texture', 'design', '颜色', '图像', '形状', '光线',
                              '图案', '纹理', '设计']

            visual_evidence_words = [w for w in evidence.split()
                                   if len(w) > 3 and any(vt in w.lower() for vt in visual_keywords)]

            visual_evidence_quality = min(len(visual_evidence_words) / 10, 1.0) if evidence else 0

            # 图像质量因子 = 特征词质量 * 0.6 + 视觉证据质量 * 0.4
            image_quality_factor = trait_quality * 0.6 + visual_evidence_quality * 0.4

            # 光子密度 = 视觉信息密度 * 0.7 + 图像质量因子 * 0.3
            photon_density = visual_density * 0.7 + image_quality_factor * 0.3

            return min(photon_density, 1.0)  # 确保结果在0-1范围内

        except Exception as e:
            print(f"计算光子密度时出错: {str(e)}")
            return None

    def _calculate_fringe_clarity(self, memories: List[Dict]) -> float:
        """计算干涉条纹清晰度 - 图像模式显著性

        干涉条纹清晰度代表图像模式的显著性和一致性，通过计算获得。
        公式: 条纹清晰度 = 视觉趋势强度 × 图像模式重复指数

        参数:
            memories: 包含图像记忆数据的列表

        返回:
            float: 干涉条纹清晰度评分 (0.0-1.0)
        """
        if not memories or len(memories) < 2:
            return None

        try:
            # 根据batch_id排序记忆
            sorted_memories = sorted(memories, key=lambda m: m.get('batch_id', 0))

            # 计算视觉趋势强度
            batches = [m.get('batch_id', 0) for m in sorted_memories]
            scores = [m.get('score', 0) for m in sorted_memories]

            # 使用线性回归计算趋势
            if len(set(batches)) >= 2:  # 至少需要两个不同的batch
                trend_coefficient = np.polyfit(batches, scores, 1)[0]
                trend_strength = min(abs(trend_coefficient) * 10, 1.0)
            else:
                return None

            # 计算图像模式重复指数 (评估图像描述的一致性)
            # 提取所有解释和证据中的视觉关键词
            visual_keywords = set()
            visual_word_categories = [
                'color', 'image', 'shape', 'light', 'pattern', 'texture', 'design',
                '颜色', '图像', '形状', '光线', '图案', '纹理', '设计'
            ]

            for memory in memories:
                explanation = memory.get('explanation', '').lower()
                evidence = memory.get('evidence', '').lower()

                # 提取与视觉相关的关键词
                visual_words = [word for word in (explanation + ' ' + evidence).split()
                              if any(vt in word.lower() for vt in visual_word_categories)]
                visual_keywords.update(visual_words)

            # 计算重复指数。没有视觉词就无法从文字证据审计模式重复性，
            # 因此标记为不可计算，而不是填入中性值。
            if not visual_keywords:
                return None
            keyword_density = min(len(visual_keywords) / 20, 1.0)  # 关键词密度
            score_std = np.std(scores) if len(scores) > 1 else 0
            score_mean = np.mean(scores) if scores else 0.0
            score_consistency = 1 - min(score_std / max(score_mean, 0.01), 1.0)

            pattern_repetition = 0.7 * score_consistency + 0.3 * keyword_density

            # 应用公式
            return trend_strength * pattern_repetition

        except Exception as e:
            print(f"计算干涉条纹清晰度时出错: {str(e)}")
            return None

    def _calculate_diffraction_resolution(self, memories: List[Dict]) -> float:
        """计算衍射分辨率 - 图像信息区分能力

        衍射分辨率代表图像信息的区分能力和边界清晰度，通过计算获得。
        公式: 衍射分辨率 = 视觉边界清晰度 × 图像特征区分度

        参数:
            memories: 包含图像记忆数据的列表

        返回:
            float: 衍射分辨率评分 (0.0-1.0)
        """
        if not memories:
            return None

        try:
            # 视觉边界清晰度 - 评估图像描述的边界清晰度
            explanations = [m.get('explanation', '') for m in memories if m.get('explanation')]
            evidences = [m.get('evidence', '') for m in memories if m.get('evidence')]

            if not (explanations or evidences):
                return None

            # 计算视觉边界词的使用频率
            boundary_words = ['clear', 'sharp', 'distinct', 'defined', 'outline', 'edge', 'border', 'contrast',
                            'boundary', 'margin', 'silhouette', 'profile', 'contour',
                            '清晰', '锐利', '明确', '轮廓', '边缘', '边界', '对比', '界限', '轮廓', '外形']

            # 计算视觉描述文本中使用边界词的比例
            all_text = ' '.join(explanations + evidences).lower()
            all_words = all_text.split()
            boundary_count = sum(1 for word in all_words if any(bw in word for bw in boundary_words))

            # 计算视觉边界清晰度指数
            visual_boundary = min(boundary_count / 5, 1.0) if all_words else 0.0

            # 图像特征区分度 - 评分的方差和视觉描述的多样性
            scores = [m.get('score', 0) for m in memories]
            score_variance = np.var(scores) if len(scores) > 1 else 0.0

            # 计算视觉描述的多样性
            visual_diversity = 0.0
            if explanations or evidences:
                # 提取所有视觉特征词
                visual_features = []
                visual_categories = ['color', 'shape', 'texture', 'pattern', 'design', 'image',
                                   '颜色', '形状', '纹理', '图案', '设计', '图像']

                for text in explanations + evidences:
                    visual_features.extend([word for word in text.lower().split()
                                          if any(vt in word for vt in visual_categories)])

                # 计算多样性指数 - 基于不同视觉特征词的比例
                unique_features = set(visual_features)
                if visual_features:
                    visual_diversity = min(len(unique_features) / len(visual_features) * 2, 1.0)

            # 计算综合图像特征区分度
            feature_distinction = (score_variance * 0.5 + visual_diversity * 0.5) * 2
            feature_distinction = min(feature_distinction, 1.0)

            # 应用公式
            return visual_boundary * feature_distinction

        except Exception as e:
            print(f"计算衍射分辨率时出错: {str(e)}")
            return None

    def _calculate_batch_summary(self, batch_results: List[Dict]) -> Dict:
        """计算batch的阶段性总结"""
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
                                'image_path': result.get('image_path', ''),
                                'image_url': result.get('image_url', ''),
                                'evidence': scores.get('evidence', ''),
                                'explanation': scores.get('explanation', '')
                            })

        except Exception as e:
            print(f"计算batch总结时出错: {str(e)}")

        return summary

    def reset(self):
        """重置模型状态，用于处理新的文件"""
        # 重置记忆池
        self.memory_pool = {
            'trait_memories': defaultdict(list),  # 存储特征词评分记忆
            'batch_summaries': [],               # 存储batch级别总结
            'reflection_results': {},            # 存储反思结果
            'all_images': [],                     # 存储所有图像
            'total_images': 0,                    # 图像总数
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
                'total_images': self.memory_pool['total_images'],
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
                            'image_path': best_memory.get('image_path', ''),
                            'image_url': best_memory.get('image_url', ''),
                            'evidence': best_memory.get('evidence', ''),
                            'explanation': best_memory.get('explanation', '')
                        }

                    # 保存特征词数据
                    trait_data = {
                        'score': round(trait_score, 4),
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
