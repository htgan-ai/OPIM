import os
import warnings

# 配置 NumPy 环境
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # 限制 OpenBLAS 线程数
os.environ['MKL_NUM_THREADS'] = '1'       # 限制 MKL 线程数

# 忽略特定警告
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._distributor_init')
warnings.filterwarnings('ignore', message='loaded more than 1 DLL from .libs')

# 然后再导入其他库
import numpy as np
import json
from typing import Dict, List
from datetime import datetime
import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import time
import random

try:
    from opim.api_config import api_endpoint, api_key, api_model
except ImportError:
    from api_config import api_endpoint, api_key, api_model

class SummaryAgent:
    def __init__(self):
        self.name = "SummaryAgent"
        
        # API配置
        self.api_key = api_key()
        self.api_url = api_endpoint()
        self.api_model = api_model()
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # 维度标准及其关键词
        self.luxury_standards = {
            'Sincerity': ['Laidback', 'Simple', 'Gentle', 'Family-oriented', 
                         'Down-to-earth', 'Sensitive', 'Thoughtful', 'Warm'],
            'Excitement': ['Energetic', 'Exciting', 'Adventurous', 'Fun', 
                         'Daring', 'Outgoing', 'Cool', 'Colorful'],
            'Professionalism': ['Mature', 'Professional', 'Intelligent', 
                              'Reliable', 'Refined'],
            'Sophistication': ['Upper-class', 'Wealthy', 'Status-conscious', 
                             'Stylish', 'Sophisticated'],
            'Attractiveness': ['Beautiful', 'Attractive', 'Good-looking', 
                             'Artistic', 'Gorgeous'],
            'Materialism': ['Selfish', 'Materialistic', 'Stuck-up', 
                          'Pretentious', 'Showy']
        }
        
        self.session = self._create_retry_session()

    def _create_retry_session(self):
        """创建带重试机制的session"""
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504, 429])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('https://', adapter)
        return session

    def analyze_with_api(self, dimension: str, keyword: str, official_evidence: str, user_evidence: str) -> Dict:
        """使用API分析差异"""
        if not self.api_key:
            # Do not send an unauthenticated request with an empty bearer
            # token.  Keep the established failure schema so callers can
            # report the optional summary stage as unavailable.
            return {
                "status": "failed",
                "error": "OPIM_API_KEY is required for SummaryAgent",
                "expression_gap": "",
                "focus_gap": "",
                "emotion_gap": "",
                "suggestions": [],
            }
        prompt = f"""
        分析该品牌在{dimension}维度下{keyword}特征的品牌传播与用户感知差异：

        品牌表达证据：{official_evidence}

        用户反馈证据：{user_evidence}

        请分析表达方式、内容重点、情感倾向的差异，并给出改进建议。
        严格按照以下JSON格式返回，确保JSON格式完整且不包含任何其他内容：
        {{
            "expression_gap": "限制在100字以内",
            "focus_gap": "限制在100字以内",
            "emotion_gap": "限制在100字以内",
            "suggestions": ["每条建议限制在50字以内", "最多3条建议"]
        }}
        """
        
        try:
            payload = {
                "model": self.api_model,
                "messages": [
                    {"role": "system", "content": "你是一个品牌分析专家。请确保返回完整且格式正确的JSON。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000  # 限制返回长度
            }
            
            response = self.session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        content = response_json["choices"][0]["message"]["content"]
                        # 清理内容，移除可能的格式标记
                        content = content.replace("```json", "").replace("```", "").strip()
                        try:
                            parsed = json.loads(content)
                            required = ("expression_gap", "focus_gap", "emotion_gap", "suggestions")
                            if not isinstance(parsed, dict) or any(key not in parsed for key in required) or not isinstance(parsed["suggestions"], list):
                                raise ValueError("API response does not match the summary schema")
                            return parsed
                        except (json.JSONDecodeError, ValueError) as e:
                            print(f"JSON解析错误: {str(e)}")
                            print(f"原始内容: {content}")
                    else:
                        print(f"API响应格式异常: {response_json}")
                except json.JSONDecodeError:
                    print(f"API响应解析失败: {response.text[:200]}")
            else:
                print(f"API请求失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text[:200]}")
            
            return {
                "status": "failed",
                "error": f"HTTP状态码: {response.status_code}",
                "expression_gap": "",
                "focus_gap": "",
                "emotion_gap": "",
                "suggestions": [],
            }
            
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "expression_gap": "",
                "focus_gap": "",
                "emotion_gap": "",
                "suggestions": [],
            }

    def compare_evaluations(self, official_file: str, user_file: str, output_dir: str) -> Dict:
        """比较官方和用户评估结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载评估结果
        with open(official_file, 'r', encoding='utf-8') as f:
            official_data = json.load(f)
        with open(user_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        
        def _mapping(value):
            return value if isinstance(value, dict) else {}

        def _finite_score(entry):
            entry = _mapping(entry)
            raw = entry.get('final_score')
            try:
                score = float(raw)
            except (TypeError, ValueError):
                return None
            return score if np.isfinite(score) else None

        official_data = _mapping(official_data)
        user_data = _mapping(user_data)
        official_dimensions = _mapping(official_data.get('dimensions'))
        user_dimensions = _mapping(user_data.get('dimensions'))
        analysis = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_summary': {
                'official_records': official_data.get('total_texts', official_data.get('total_images')),
                'user_records': user_data.get('total_texts', user_data.get('total_images'))
            },
            'dimension_analysis': {}
        }
        
        # 分析每个维度
        for dimension, keywords in self.luxury_standards.items():
            dimension_analysis = {
                'keywords_comparison': [],
                'overall_gap': 0,
                'significant_differences': []
            }
            
            # 分析每个关键词
            valid_scores = 0
            total_gap = 0
            
            for keyword in keywords:
                official_dimension = _mapping(official_dimensions.get(dimension))
                user_dimension = _mapping(user_dimensions.get(dimension))
                official_entry = _mapping(official_dimension.get(keyword))
                user_entry = _mapping(user_dimension.get(keyword))
                official_score = _finite_score(official_entry)
                user_score = _finite_score(user_entry)
                official_evidence = str(official_entry.get("evidence_summary", "") or "")
                user_evidence = str(user_entry.get("evidence_summary", "") or "")
                
                # 记录所有关键词的状态
                keyword_analysis = {
                    'keyword': keyword,
                    'official_score': official_score,
                    'user_score': user_score,
                    'status': 'complete',  # 默认状态
                    'gap': None,
                    'evidence': {
                        'official': self._summarize_evidence(official_evidence),
                        'user': self._summarize_evidence(user_evidence)
                    }
                }
                
                if official_score is None and user_score is None:
                    keyword_analysis['status'] = 'both_missing'
                elif official_score is None:
                    keyword_analysis['status'] = 'official_missing'
                elif user_score is None:
                    keyword_analysis['status'] = 'user_missing'
                else:
                    gap = official_score - user_score
                    total_gap += gap
                    valid_scores += 1
                    keyword_analysis['gap'] = gap
                    
                    # 记录显著差异 (阈值改为0.3)
                    if abs(gap) > 0.3:
                        dimension_analysis['significant_differences'].append({
                            'keyword': keyword,
                            'gap': gap,
                            'explanation': self._analyze_difference(
                                dimension,
                                keyword,
                                gap,
                                official_evidence,
                                user_evidence
                            )
                        })
                
                dimension_analysis['keywords_comparison'].append(keyword_analysis)
            
            # 计算整体差异 (只考虑有效得分)
            if valid_scores > 0:
                dimension_analysis['overall_gap'] = total_gap / valid_scores
            else:
                dimension_analysis['overall_gap'] = None
            
            analysis['dimension_analysis'][dimension] = dimension_analysis
        
        # 生成报告
        self._generate_report(analysis, output_dir)
        return analysis

    def _summarize_evidence(self, evidence: str) -> str:
        """提取和总结关键证据"""
        if not evidence:
            return ""
            
        # 将证据按分号分割
        examples = evidence.split(';')
        
        # 选择2-3个最具代表性的例子
        selected = examples[:2] if len(examples) > 2 else examples
        
        # 添加总结性话语
        summary = "; ".join(selected).strip()
        if len(examples) > 2:
            summary += "; 等。"
            
        return summary

    def _analyze_difference(self, dimension: str, keyword: str, gap: float, official_evidence: str, user_evidence: str) -> Dict:
        """分析差异原因"""
        direction = "高于" if gap > 0 else "低于"
        magnitude = "显著" if abs(gap) > 0.4 else "略微"
        
        basic_analysis = f"品牌传播{magnitude}{direction}用户感知 ({abs(gap):.2f}分)"
        
        # 如果有证据，使用API进行深入分析
        if official_evidence and user_evidence:
            api_analysis = self.analyze_with_api(
                dimension,
                keyword,
                official_evidence,
                user_evidence
            )
            time.sleep(random.uniform(1, 2))
            
            return {
                'basic_analysis': basic_analysis,
                'detailed_analysis': api_analysis
            }
        
        return {
            'basic_analysis': basic_analysis,
            'detailed_analysis': None
        }

    def _generate_report(self, analysis: Dict, output_dir: str):
        """生成Markdown格式的分析报告"""
        report_path = os.path.join(output_dir, 'evaluation_comparison.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 品牌形象评估对比分析\n\n")
            f.write(f"生成时间：{analysis['timestamp']}\n\n")
            
            f.write("## 数据概况\n")
            f.write(f"- 官方记录数量：{analysis['data_summary']['official_records']}\n")
            f.write(f"- 用户记录数量：{analysis['data_summary']['user_records']}\n\n")
            
            f.write("## 维度分析\n")
            for dimension, results in analysis['dimension_analysis'].items():
                f.write(f"\n### {dimension}\n")
                if results['overall_gap'] is not None:
                    f.write(f"整体差异：{results['overall_gap']:.2f} (基于{len([k for k in results['keywords_comparison'] if k['gap'] is not None])}个有效特征词)\n\n")
                else:
                    f.write("整体差异：无法计算 (缺少有效数据)\n\n")
                
                f.write("#### 关键词得分对比\n")
                for keyword in results['keywords_comparison']:
                    f.write(f"\n**{keyword['keyword']}**\n")
                    
                    if keyword['status'] == 'both_missing':
                        f.write("状态：官方和用户数据均缺失\n")
                    elif keyword['status'] == 'official_missing':
                        f.write("状态：缺少官方数据\n")
                        f.write(f"用户得分：{keyword['user_score']:.2f}\n")
                    elif keyword['status'] == 'user_missing':
                        f.write("状态：缺少用户数据\n")
                        f.write(f"官方得分：{keyword['official_score']:.2f}\n")
                    else:
                        f.write(f"- 官方得分：{keyword['official_score']:.2f}\n")
                        f.write(f"- 用户得分：{keyword['user_score']:.2f}\n")
                        f.write(f"- 差异：{keyword['gap']:.2f}\n")
                    
                    if keyword['evidence']['official'] or keyword['evidence']['user']:
                        f.write("\n典型证据：\n")
                        if keyword['evidence']['official']:
                            f.write(f"- 品牌表达：{keyword['evidence']['official']}\n")
                        if keyword['evidence']['user']:
                            f.write(f"- 用户反馈：{keyword['evidence']['user']}\n")
                    
                    explanation = next((item["explanation"] for item in results["significant_differences"] if item["keyword"] == keyword["keyword"]), None)
                    api_analysis = explanation.get("detailed_analysis") if isinstance(explanation, dict) else None
                    if isinstance(api_analysis, dict) and api_analysis.get("status") != "failed":
                        f.write("\nAPI分析结果：\n")
                        f.write(f"- 表达差异：{api_analysis.get('expression_gap', '')}\n")
                        f.write(f"- 重点差异：{api_analysis.get('focus_gap', '')}\n")
                        f.write(f"- 情感差异：{api_analysis.get('emotion_gap', '')}\n")
                        for suggestion in api_analysis.get("suggestions", []):
                            f.write(f"- 建议：{suggestion}\n")
                
                if results['significant_differences']:
                    f.write("\n#### 显著差异分析\n")
                    for diff in results['significant_differences']:
                        f.write(f"\n**{diff['keyword']}** (差异：{diff['gap']:.2f})\n")
                        detail_analysis = diff['explanation'].get('detailed_analysis')
                        if isinstance(detail_analysis, dict) and detail_analysis.get("status") != "failed":
                            f.write("- 表达差异：" + str(detail_analysis.get('expression_gap', '')) + "\n")
                            f.write("- 重点差异：" + str(detail_analysis.get('focus_gap', '')) + "\n")
                            f.write("- 情感差异：" + str(detail_analysis.get('emotion_gap', '')) + "\n")
                            for suggestion in detail_analysis.get('suggestions', []):
                                f.write(f"- 建议：{suggestion}\n")
                        else:
                            f.write(diff['explanation']['basic_analysis'] + "\n")
            
            # 保存原始数据
            json_path = os.path.join(output_dir, 'evaluation_comparison.json')
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(analysis, jf, ensure_ascii=False, indent=2)
            
        print(f"\n分析完成！报告已保存到：{report_path}")

    def analyze_gaps(self, official_data: Dict, user_data: Dict) -> Dict:
        """Summarize score gaps while preserving missing-data status.

        This helper reports dimension-score differences only. It does not claim
        to compute semantic, emotional, or topic statistics that are absent
        from the released source data.
        """
        def values(data: Dict) -> Dict[str, float]:
            output = {}
            for dimension, traits in data.get("dimensions", {}).items():
                if not isinstance(traits, dict):
                    continue
                scores = []
                for trait in traits.values():
                    if not isinstance(trait, dict):
                        continue
                    raw = trait.get("final_score", trait.get("score"))
                    try:
                        score = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(score):
                        scores.append(score)
                if scores:
                    output[dimension] = float(np.mean(scores))
            return output

        official = values(official_data)
        user = values(user_data)
        differences = {}
        paired_gaps = []
        for dimension in sorted(set(official) | set(user)):
            if dimension in official and dimension in user:
                difference = user[dimension] - official[dimension]
                differences[dimension] = {
                    "official": official[dimension], "user": user[dimension],
                    "difference": difference, "status": "complete",
                }
                paired_gaps.append(difference)
            elif dimension in official:
                differences[dimension] = {
                    "official": official[dimension], "user": None,
                    "difference": None, "status": "user_missing",
                }
            else:
                differences[dimension] = {
                    "official": None, "user": user[dimension],
                    "difference": None, "status": "official_missing",
                }
        return {
            "dimension_differences": differences,
            "metrics": {
                "mean_absolute_dimension_gap": float(np.mean(np.abs(paired_gaps))) if paired_gaps else None,
                "dimension_gap_std": float(np.std(paired_gaps)) if paired_gaps else None,
                "max_dimension_gap": float(max((abs(value) for value in paired_gaps), default=0.0)) if paired_gaps else None,
            },
        }

    def batch_compare_evaluations(self, official_base_dir: str, user_base_dir: str, output_base_dir: str):
        """
        批量比较多个品牌的官方和用户评估结果
        
        Args:
            official_base_dir: 官方评估结果文件夹路径
            user_base_dir: 用户评估结果文件夹路径
            output_base_dir: 输出结果的基础文件夹路径
        """
        # 获取官方数据中所有品牌子文件夹
        brand_folders = sorted(
            f for f in os.listdir(official_base_dir)
            if os.path.isdir(os.path.join(official_base_dir, f))
        )
        
        print(f"找到 {len(brand_folders)} 个品牌文件夹需要处理")
        
        for brand in brand_folders:
            official_brand_dir = os.path.join(official_base_dir, brand)
            user_brand_dir = os.path.join(user_base_dir, brand)
            
            # 检查用户数据中是否存在对应的品牌文件夹
            if not os.path.exists(user_brand_dir):
                print(f"警告: 品牌 '{brand}' 在用户数据中不存在，跳过")
                continue
            
            # 找到评估结果文件
            official_file = os.path.join(official_brand_dir, "evaluation_result.json")
            user_file = os.path.join(user_brand_dir, "evaluation_result.json")
            
            # 检查文件是否存在
            if not os.path.exists(official_file):
                print(f"警告: 官方评估结果文件在 '{brand}' 文件夹中不存在，跳过")
                continue
                
            if not os.path.exists(user_file):
                print(f"警告: 用户评估结果文件在 '{brand}' 文件夹中不存在，跳过")
                continue
            
            # 创建输出目录
            output_dir = os.path.join(output_base_dir, f"{brand}_analysis_results")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"处理品牌: {brand}")
            try:
                # 执行比较分析
                self.compare_evaluations(official_file, user_file, output_dir)
                print(f"品牌 '{brand}' 分析完成，结果保存至: {output_dir}")
            except Exception as e:
                print(f"处理品牌 '{brand}' 时出错: {str(e)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare official and user evaluation JSON files.")
    parser.add_argument("--official-base-dir", required=True)
    parser.add_argument("--user-base-dir", required=True)
    parser.add_argument("--output-base-dir", required=True)
    args = parser.parse_args()
    SummaryAgent().batch_compare_evaluations(
        args.official_base_dir, args.user_base_dir, args.output_base_dir
    )

if __name__ == "__main__":
    main()
