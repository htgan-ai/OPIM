from __future__ import annotations

import html
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Set, Optional, Any, Tuple
import glob
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 配置 NumPy 环境
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # 限制 OpenBLAS 线程数
os.environ['MKL_NUM_THREADS'] = '1'       # 限制 MKL 线程数

# 忽略特定警告
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._distributor_init')
warnings.filterwarnings('ignore', message='loaded more than 1 DLL from .libs')

# 然后再导入 numpy
import numpy as np
import requests
from requests.adapters import HTTPAdapter

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from opim.api_config import api_endpoint, api_key, api_model
except ImportError:
    from api_config import api_endpoint, api_key, api_model

try:
    from opim.models.memory_reflection_model_text import MemoryReflectionModel
    from opim.models.shared_memory import SharedMemoryHub
except ImportError:
    from models.memory_reflection_model_text import MemoryReflectionModel
    from models.shared_memory import SharedMemoryHub

class TextAnalysisAgent:
    def __init__(self, project_root: Optional[str] = None, name: str = "TextAnalysisAgent", shared_memory: Optional[SharedMemoryHub] = None):
        # 初始化代理
        self.name = name
        project_root = project_root or os.getenv("OPIM_OUTPUT_DIR")
        self._external_shared_memory = shared_memory is not None
        self.shared_memory = shared_memory or SharedMemoryHub()
        self.shared_memory.add_participant(self.name)
        
        # 加载奢侈品标准维度
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
        
        # API配置
        self.api_key = api_key()
        self.api_url = api_endpoint()
        self.api_model = api_model()
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # 记忆流相关
        self.memory_stream = []
        self.reflection_threshold = 3
        
        # 添加批量处理相关配置
        self.batch_size = 10
        # OPIM_MAX_RETRIES is the complete per-record budget.  The batch
        # loop owns semantic/JSON retries; _call_api performs one transport
        # attempt per batch-loop iteration so the budget cannot multiply.
        self.max_retries = max(1, int(os.getenv("OPIM_MAX_RETRIES", "6")))
        self.retry_delay = float(os.getenv("OPIM_RETRY_DELAY", "5"))
        # Interaction-count filtering is opt-in (OPIM_MIN_INTERACTION) and is
        # recorded in the run summary when configured.
        raw_min_interaction = os.getenv("OPIM_MIN_INTERACTION", "").strip()
        try:
            self.min_interaction = int(raw_min_interaction) if raw_min_interaction else None
        except ValueError as exc:
            raise ValueError("OPIM_MIN_INTERACTION must be an integer when provided") from exc
        if self.min_interaction is not None and self.min_interaction < 0:
            raise ValueError("OPIM_MIN_INTERACTION must be non-negative")
        self.preprocessing_summary = {}
        
        # 添加文件路径配置
        self.process_file = (os.path.join(project_root, 'tests', 'results', 'evaluation_results', 'evaluation_process.json')
                             if project_root else None)
        self.result_file = (os.path.join(project_root, 'tests', 'results', 'evaluation_results', 'evaluation_result.json')
                            if project_root else None)
        
        # 添加模型文件路径 (解决引用错误)
        self.model_file = (os.path.join(project_root, 'models', 'memory_reflection_model_text.py')
                           if project_root else None)
        self.standard_file = (os.path.join(project_root, 'data', 'luxury_standards.json')
                              if project_root else None)
        
        # 确保目录存在
        if self.process_file:
            os.makedirs(os.path.dirname(self.process_file), exist_ok=True)
        if self.standard_file:
            os.makedirs(os.path.dirname(self.standard_file), exist_ok=True)
        
        # 尝试从文件加载奢侈品标准
        self._load_luxury_standards()
        
        # 初始化记忆流-反思模型
        self.memory_model = MemoryReflectionModel(self.luxury_standards, shared_memory=self.shared_memory, participant_name=self.name + ".memory")
        self.last_failures = []
        self._active_session = False
        self._run_id = None
        
        # 保存奢侈品标准到文件
        self._save_luxury_standards()
        
    def _reset_session(self) -> None:
        """Start an isolated model session while preserving an injected hub."""
        self.memory_model = MemoryReflectionModel(
            self.luxury_standards,
            shared_memory=self.shared_memory,
            participant_name=self.name + ".memory",
        )
        self.last_failures = []
        self._run_id = uuid.uuid4().hex
        self._active_session = True
        # Process files written in this session; the first write of a session
        # starts a new log instead of appending to a previous run's file.
        self._session_process_files = set()

    @staticmethod
    def _record_datetime(record: Dict) -> Optional[datetime]:
        for field in ("publish_date_date", "created_at", "publish_time", "publish_date", "timestamp", "date"):
            value = record.get(field) if isinstance(record, dict) else None
            if not value:
                continue
            text = str(value).strip().replace("Z", "+00:00")
            for parser in (datetime.fromisoformat,):
                try:
                    parsed = parser(text)
                    if parsed.tzinfo is not None:
                        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                    return parsed
                except (TypeError, ValueError):
                    pass
            for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
                try:
                    return datetime.strptime(text, fmt)
                except (TypeError, ValueError):
                    continue
        return None

    @classmethod
    def _sort_records(cls, records: List[Dict]) -> List[Dict]:
        indexed = list(enumerate(records or []))
        return [record for _, record in sorted(
            indexed,
            key=lambda item: (cls._record_datetime(item[1]) is None,
                              cls._record_datetime(item[1]) or datetime.max,
                              item[0]),
        )]

    @staticmethod
    def _interaction_value(record: Dict) -> Optional[int]:
        """Read the released interaction count without guessing another metric."""
        if not isinstance(record, dict):
            return None
        value = record.get("interaction_cnt", record.get("interaction_count"))
        try:
            parsed = int(float(value))
            return parsed if parsed >= 0 else None
        except (TypeError, ValueError, OverflowError):
            return None

    def _apply_optional_preprocessing(self, records: List[Dict]) -> List[Dict]:
        """Apply only an explicitly configured interaction threshold.

        Private advertising/symbol filters and Gemini correction logs were not
        recovered. They are not inferred here; this helper exposes only the
        released interaction field and records exactly what it did.
        """
        records = list(records or [])
        threshold = self.min_interaction
        if threshold is None:
            self.preprocessing_summary = {
                "interaction_filter": {"enabled": False, "threshold": None},
                "input_records": len(records), "kept_records": len(records),
                "dropped_records": 0,
                "missing_interaction_count": sum(self._interaction_value(r) is None for r in records),
            }
            return records
        kept, missing = [], 0
        for record in records:
            count = self._interaction_value(record)
            if count is None:
                missing += 1
                continue
            if count >= threshold:
                kept.append(record)
        self.preprocessing_summary = {
            "interaction_filter": {"enabled": True, "threshold": threshold, "missing_values_dropped": True},
            "input_records": len(records), "kept_records": len(kept),
            "dropped_records": len(records) - len(kept),
            "missing_interaction_count": missing,
        }
        return kept

    def _record_prompt_text(self, record: Dict) -> str:
        """Build model input from title, body, and released keywords."""
        record = record if isinstance(record, dict) else {}
        title = self._clean_text_content(str(record.get("title", "") or ""))
        content = self._clean_text_content(str(record.get("content", "") or ""))
        keywords = record.get("keywords", "")
        if isinstance(keywords, (list, tuple, set)):
            keywords = ", ".join(self._clean_text_content(str(item)) for item in keywords if item is not None)
        else:
            keywords = self._clean_text_content(str(keywords or ""))
        sections = []
        if title:
            sections.append("标题:\n" + title)
        if keywords:
            sections.append("关键词:\n" + keywords)
        if content:
            sections.append("正文:\n" + content)
        return "\n\n".join(sections)

    def analyze_batch(self, texts: List[Dict], output_folder: str = None, reset: bool = True) -> List[Dict]:
        """分析一个batch的文本，并将结果保存到指定文件夹"""
        try:
            new_session = reset or not self._active_session
            if new_session:
                self._reset_session()
            texts = self._sort_records(list(texts or []))
            # Direct callers commonly use analyze_batch as the public entry
            # point.  Apply the opt-in interaction filter once for a new
            # session; process_single_file applies it before its per-batch
            # loop and therefore passes reset=False here.
            if reset:
                texts = self._apply_optional_preprocessing(texts)
            batch_results = []
            failed_texts = []
            
            # 1. 获取每条文本的API分析结果
            for text in texts:
                success = False
                last_error = ""
                for attempt in range(self.max_retries):
                    try:
                        prompt_text = self._record_prompt_text(text)
                        if not prompt_text:
                            raise ValueError("text record has no title, keywords, or content")
                        prompt = self._create_batch_prompt(prompt_text)
                        response = self._call_api(prompt)
                        
                        result = None
                        if response:
                            result = self._process_api_response(response)
                        
                        # API/解析失败时重试；失败记录不会伪装成有效评分。
                        if not result or 'results' not in result:
                            raise ValueError("API response did not contain a valid results list")

                        # 处理API响应并提取光学模型评分
                        processed_result = self._process_batch_response(result, [text])
                        if processed_result:
                            batch_results.extend(processed_result)
                            success = True
                            break
                        
                        # 如果处理失败且不是最后一次尝试，则等待后重试
                        if attempt < self.max_retries - 1:
                            print(f"处理失败，等待{self.retry_delay}秒后重试...")
                            time.sleep(self.retry_delay)
                            
                    except Exception as e:
                        last_error = str(e)
                        print(f"处理文本失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                
                if not success:
                    failed_texts.append(text)
                    self.last_failures.append({"record": text, "error": last_error or "API response or result parsing failed"})
            
            # 2. 记忆流处理
            if batch_results:
                batch_summary = self.memory_model.memory_stream(batch_results)
                
                # 3. 保存评估过程（使用指定的输出文件夹）
                process_file = self.process_file
                if output_folder:
                    process_file = os.path.join(output_folder, os.path.basename(self.process_file or "evaluation_process.json"))
                self._save_evaluation_process(batch_results, batch_summary, process_file)
                
                # 4. 存储批次结果
                self.shared_memory.store(
                    key=f"{self.name}:{self._run_id}:batch_{len(self.memory_model.memory_pool['batch_summaries'])}",
                    value={
                        'results': batch_results,
                        'summary': batch_summary
                    },
                    metadata={'type': 'batch_analysis', 'output_folder': output_folder, 'run_id': self._run_id}
                )
            
            return batch_results
            
        except Exception as e:
            print(f"批次分析错误: {str(e)}")
            self.last_failures.append({"scope": "batch", "error": str(e)})
            return []

    def analyze(self, texts: List[Dict], output_folder: Optional[str] = None) -> Dict:
        """Run the complete three-stage evaluation for in-memory records.

        Records are processed in chronological order in batches of
        ``self.batch_size``; each batch is one time step t of the interference
        stage, so the trend-based quantities need at least two batches.
        """
        self._reset_session()
        texts = self._apply_optional_preprocessing(self._sort_records(list(texts or [])))
        batch_size = max(1, int(self.batch_size))
        results: List[Dict] = []
        for start in range(0, len(texts), batch_size):
            results.extend(self.analyze_batch(texts[start:start + batch_size], output_folder=output_folder, reset=False))
        if results:
            self.memory_model.reflection()
        final = self.memory_model.calculate_final_scores()
        dimension_scores = {}
        evidence = {"text_evidence": []}
        for dimension, traits in final.get("dimensions", {}).items():
            values = [item.get("final_score") for item in traits.values() if item.get("final_score") is not None]
            if values:
                dimension_scores[dimension] = float(np.mean(values))
            evidence["text_evidence"].extend(
                item.get("evidence_summary", "") for item in traits.values() if item.get("evidence_summary")
            )
        return {
            "dimension_scores": dimension_scores,
            "evidence": evidence,
            "results": results,
            "failed": list(self.last_failures),
            "final_scores": final,
        }

    def _call_api(self, prompt: str) -> Dict:
        """调用API进行文本分析"""
        if not self.api_key:
            raise RuntimeError("OPIM_API_KEY is required for TextAnalysisAgent")
        # Keep transport to one request here.  analyze_batch retries the
        # complete record (request + parse + validation) at most max_retries.
        api_attempts = 1
        for attempt in range(api_attempts):
            try:
                print(f"API调用尝试 {attempt + 1}/{api_attempts}")
                
                payload = {
                    "model": self.api_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一个专业的文本分析助手，请严格按照JSON格式返回分析结果。"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4096,
                    "top_p": 0.95
                }

                session = requests.Session()
                # urllib3 retries would silently multiply the explicit
                # per-record budget, so transport retries are disabled.
                adapter = requests.adapters.HTTPAdapter(max_retries=0)
                session.mount('http://', adapter)
                session.mount('https://', adapter)

                print(f"发送API请求，目标URL: {self.api_url}")
                
                # 设置更长的超时时间
                response = session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120  # 增加超时时间到120秒
                )
                
                print(f"收到API响应，状态码: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        response_json = response.json()
                        # 检查响应格式
                        if not isinstance(response_json, dict):
                            print(f"API响应不是字典: {type(response_json)}")
                            if attempt == api_attempts - 1:
                                print(f"响应内容: {response_json}")
                            continue
                            
                        choices = response_json.get('choices', [])
                        if not choices:
                            print("API响应中没有choices")
                            if attempt == api_attempts - 1:
                                print(f"完整响应: {response_json}")
                            continue
                            
                        message = choices[0].get('message', {})
                        if not message:
                            print("API响应中没有message")
                            if attempt == api_attempts - 1:
                                print(f"choices内容: {choices}")
                            continue
                            
                        content = message.get('content', '')
                        if not content:
                            print("API响应中没有content")
                            if attempt == api_attempts - 1:
                                print(f"message内容: {message}")
                            continue
                        
                        # 打印完整的响应内容用于调试
                        print("\nAPI完整响应:")
                        print("=" * 50)
                        print(json.dumps(response_json, ensure_ascii=False, indent=2))
                        print("=" * 50)
                        
                        # 检查content是否包含完整的JSON
                        if not ('{' in content and '}' in content):
                            print("Content不包含完整的JSON")
                            if attempt == api_attempts - 1:
                                # 最后一次尝试时打印完整content内容
                                print(f"完整content内容:\n{content}")
                            
                            # 对于不是最后一次尝试，可以调整提示词并重试
                            if attempt < api_attempts - 1:
                                # 修改提示词，更强调返回JSON格式
                                prompt = prompt + "\n\n请确保返回有效的JSON格式，不要添加任何额外的解释。必须包含大括号{}并采用JSON格式。"
                                print(f"提示词已调整，将在{self.retry_delay}秒后重试")
                                time.sleep(self.retry_delay)
                            continue
                        
                        return response_json
                        
                    except json.JSONDecodeError as e:
                        print(f"API响应JSON解析错误: {str(e)}")
                        print(f"原始响应: {response.text[:500]}...")  # 只打印前500个字符
                        
                        # 对于不是最后一次尝试，可以调整提示词并重试
                        if attempt < api_attempts - 1:
                            # 修改提示词，更强调返回有效的JSON
                            prompt = prompt + "\n\n警告：上一次返回的不是有效的JSON格式。请确保返回可解析的JSON结构，不要有任何语法错误。"
                            print(f"提示词已调整，将在{self.retry_delay}秒后重试")
                            time.sleep(self.retry_delay * 2)  # 增加等待时间
                        continue
                else:
                    error_message = f"API错误状态码: {response.status_code}"
                    if response.text:
                        error_message += f"\n错误响应: {response.text[:500]}..."  # 只打印前500个字符
                    print(error_message)
                    
                    # 根据状态码调整重试策略
                    if response.status_code >= 500:  # 服务器错误
                        wait_time = self.retry_delay * (attempt + 1)  # 递增等待时间
                        if attempt < api_attempts - 1:
                            print(f"服务器错误，将在{wait_time}秒后重试")
                            time.sleep(wait_time)
                    elif response.status_code == 429:  # 请求过多
                        wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                        if attempt < api_attempts - 1:
                            print(f"请求过多，将在{wait_time}秒后重试")
                            time.sleep(wait_time)
                    elif response.status_code >= 400 and response.status_code < 500:  # 客户端错误
                        # 对于客户端错误，可能需要检查提示词或API密钥
                        print("客户端错误，检查提示词和API配置")
                        if attempt < api_attempts - 1:
                            # 简化提示词
                            if len(prompt) > 5000:
                                prompt = prompt[:5000] + "...[内容已截断]"
                                print("提示词已截断，将重试")
                            time.sleep(self.retry_delay)
                
            except requests.exceptions.RequestException as e:
                print(f"API请求异常 (尝试 {attempt + 1}/{api_attempts}): {str(e)}")
                if "timeout" in str(e).lower():
                    if attempt < api_attempts - 1:
                        print(f"请求超时，将在{self.retry_delay}秒后重试")
                elif "connection" in str(e).lower():
                    if attempt < api_attempts - 1:
                        print(f"连接错误，将在{self.retry_delay * 2}秒后重试")
                        time.sleep(self.retry_delay * 2)  # 连接错误需要更长的等待时间
                    continue
                
                if attempt < api_attempts - 1:
                    time.sleep(self.retry_delay)
                continue
            except Exception as e:
                print(f"API调用其他错误 (尝试 {attempt + 1}/{api_attempts}): {str(e)}")
                print(f"错误类型: {type(e).__name__}")
                if attempt < api_attempts - 1:
                    time.sleep(self.retry_delay)
                continue
        
        print(f"API调用失败，已尝试{api_attempts}次")
        return None

    def _process_api_response(self, response: Dict) -> Dict:
        """处理API响应，提取JSON内容"""
        try:
            if not isinstance(response, dict):
                print(f"无效的API响应格式: {type(response)}")
                return None
            
            choices = response.get('choices', [])
            if not choices:
                print("API响应中没有choices")
                return None
            
            message = choices[0].get('message', {})
            content = message.get('content', '')
            if not content:
                print("API响应中没有content")
                return None
            
            # 打印完整内容以便调试
            print(f"API返回的原始内容: {content}")
            
            # 从content中提取JSON
            json_content = None
            
            # 尝试查找markdown代码块中的JSON
            json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            json_blocks = re.findall(json_block_pattern, content)
            
            if json_blocks:
                # 使用第一个找到的JSON代码块
                json_content = json_blocks[0].strip()
                print(f"找到JSON代码块，长度: {len(json_content)}")
            else:
                # 尝试找到大括号包围的JSON对象
                json_pattern = r'(\{[\s\S]*\})'
                json_matches = re.findall(json_pattern, content)
                
                if json_matches:
                    # 使用第一个找到的JSON对象
                    json_content = json_matches[0].strip()
                    print(f"找到JSON对象，长度: {len(json_content)}")
                else:
                    print("无法找到JSON内容")
                    return None
            
            if not json_content:
                print("JSON内容为空")
                return None
            
            # 清理JSON字符串
            # 移除所有换行符和不必要的空格，但保留必要的空格
            json_content = re.sub(r'\n\s*', ' ', json_content)
            
            # 移除转义字符
            json_content = json_content.replace('\\n', ' ').replace('\\r', ' ')
            
            print(f"清理后的JSON (前200字符): {json_content[:200]}...")
                
            try:
                # 尝试解析JSON
                result = json.loads(json_content)
                print("JSON解析成功")
                return result
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}")
                
                # 尝试使用更安全的方法修复可能的JSON格式问题
                try:
                    # 安全的eval环境，只用于解析JSON
                    import ast
                    # 将单引号转换为双引号
                    fixed_json = json_content.replace("'", "\"")
                    # 使用ast.literal_eval解析JSON，更安全
                    result = json.loads(fixed_json)
                    print("JSON修复并解析成功")
                    return result
                except Exception as inner_e:
                    print(f"修复JSON失败: {str(inner_e)}")
                    # 不再返回默认结构，而是返回None，让光学模型来计算
                    print("解析失败，将使用光学模型计算")
                    return None
            
        except Exception as e:
            print(f"处理API响应时出错: {str(e)}")
            print(f"错误位置: {e.__traceback__.tb_lineno}")
            return None

    def _standardize_results(self, result: Dict) -> Optional[Dict]:
        """Validate source-schema text results without default scores."""
        if not isinstance(result, dict) or not isinstance(result.get("results"), list):
            return None
        standardized = []
        for item in result["results"]:
            if not isinstance(item, dict) or not isinstance(item.get("dimension_scores"), dict):
                continue
            dimensions = {}
            try:
                for dimension, scores in item["dimension_scores"].items():
                    if dimension not in self.luxury_standards or not isinstance(scores, dict):
                        raise ValueError("unknown or malformed dimension")
                    score = float(scores["score"])
                    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                        raise ValueError("dimension score must be finite and in [0,1]")
                    traits = scores.get("matching_traits", [])
                    if not isinstance(traits, list) or any(trait not in self.luxury_standards[dimension] for trait in traits):
                        raise ValueError("matching_traits contains an invalid trait")
                    dimensions[dimension] = {
                        "score": score, "matching_traits": list(traits),
                        "evidence": str(scores.get("evidence", "")),
                        "explanation": str(scores.get("explanation", "")),
                    }
            except (KeyError, TypeError, ValueError):
                continue
            optical = item.get("optical_scores", item.get("optical_model_scores", {}))
            if not isinstance(optical, dict):
                continue
            complete = True
            for dimension, scores in dimensions.items():
                for trait in scores.get("matching_traits", []):
                    values = optical.get(f"{dimension}_{trait}", {})
                    memory = values.get("memory_stream", values) if isinstance(values, dict) else {}
                    if not isinstance(memory, dict) or any(name not in memory for name in ("wave_amplitude", "wavelength_purity")):
                        complete = False
                        break
                if not complete:
                    break
            if dimensions and complete:
                standardized.append({"dimension_scores": dimensions, "optical_scores": optical})
        return {"results": standardized} if standardized else None

    def _create_batch_prompt(self, texts: str) -> str:
        """创建批量分析提示词，简化结构以方便API处理"""
        prompt = f"""分析以下文本内容，返回指定格式的JSON。

文本内容:
{texts}

评估维度及特征词:
{json.dumps(self.luxury_standards, indent=2, ensure_ascii=False)}

请按以下JSON格式返回结果:
```json
{{
  "results": [
    {{
      "dimension_scores": {{
        "维度名": {{
          "score": 0.0-1.0,
          "matching_traits": ["特征词1", "特征词2"],
          "evidence": "支持证据",
          "explanation": "解释"
        }}
      }},
      "optical_scores": {{
        "维度名_特征词": {{
          "wave_amplitude": 0.0-1.0,
          "wavelength_purity": 0.0-1.0
        }}
      }}
    }}
  ]
}}
```

评分重点说明:
- wave_amplitude (光波振幅): 必须提供，信号强度和证据质量的综合评分
- wavelength_purity (波长纯度): 必须提供，主题聚焦度和语义清晰度的综合评分

注意:
1. 仅返回JSON格式，不要添加任何解释文本
2. 只有在文本中有明确证据支持的特征词才评分
3. 每个维度的分数范围是0-1之间
4. 匹配特征词为空时使用空数组[]
5. 请确保对于每个匹配的特征词，至少提供wave_amplitude和wavelength_purity参数，这些对后续计算非常重要"""
        return prompt

    def _process_batch_response(self, response_data: Dict, original_texts: List[Dict]) -> List[Dict]:
        """Validate source-schema text results without inventing failed scores."""
        if not isinstance(response_data, dict) or not isinstance(response_data.get("results"), list):
            return []
        results = response_data["results"]
        # Each request contains exactly one source record.  Do not silently
        # truncate or align a malformed response by position.
        if len(results) != len(original_texts):
            return []
        processed_results = []
        for index, result in enumerate(results):
            if not isinstance(result, dict) or not isinstance(result.get("dimension_scores"), dict):
                continue
            dimensions = {}
            try:
                for dimension, scores in result["dimension_scores"].items():
                    if dimension not in self.luxury_standards or not isinstance(scores, dict):
                        raise ValueError("unknown or malformed dimension")
                    score = float(scores["score"])
                    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                        raise ValueError("dimension score must be finite and in [0,1]")
                    traits = scores.get("matching_traits", [])
                    if not isinstance(traits, list) or any(trait not in self.luxury_standards[dimension] for trait in traits):
                        raise ValueError("matching_traits contains an invalid trait")
                    dimensions[dimension] = {
                        "score": score,
                        "matching_traits": list(traits),
                        "evidence": str(scores.get("evidence", "")),
                        "explanation": str(scores.get("explanation", "")),
                    }
                if not dimensions:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            optical = result.get("optical_scores", result.get("optical_model_scores", {}))
            if not isinstance(optical, dict):
                optical = {}
            optical_out = {}
            for key, values in optical.items():
                if not isinstance(values, dict):
                    continue
                memory = values.get("memory_stream", values)
                reflection = values.get("reflection", values)
                if not isinstance(memory, dict) or not isinstance(reflection, dict):
                    continue
                memory_out = {}
                reflection_out = {}
                for name in ("wave_amplitude", "wavelength_purity", "photon_density"):
                    if name in memory:
                        value = float(memory[name])
                        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                            raise ValueError("invalid optical score")
                        memory_out[name] = value
                for name in ("coherence", "fringe_clarity", "diffraction_resolution"):
                    if name in reflection:
                        value = float(reflection[name])
                        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                            raise ValueError("invalid optical score")
                        reflection_out[name] = value
                optical_out[key] = {"memory_stream": memory_out, "reflection": reflection_out}
            text = original_texts[index]
            # Every evidence-backed trait must carry the two API optical
            # parameters used by the paper's memory-stream stage.  Missing
            # parameters are a failed item, never an implicit derived score.
            for dimension, scores in dimensions.items():
                for trait in scores.get("matching_traits", []):
                    key = f"{dimension}_{trait}"
                    values = optical_out.get(key)
                    if not isinstance(values, dict):
                        raise ValueError(f"missing optical scores for {key}")
                    memory_values = values.get("memory_stream", {})
                    if any(name not in memory_values for name in ("wave_amplitude", "wavelength_purity")):
                        raise ValueError(f"incomplete optical scores for {key}")
            processed_results.append({
                "content": text.get("content", ""),
                # Preserve the exact released source fields that were sent to
                # the model so downstream reports can audit title/keyword use.
                "title": text.get("title", ""),
                "keywords": text.get("keywords", ""),
                "dimension_scores": dimensions,
                "optical_model_scores": optical_out,
                "metadata": {
                    **(text.get("metadata", {}) if isinstance(text.get("metadata", {}), dict) else {}),
                    **({"record_id": text.get("record_id")} if text.get("record_id") else {}),
                },
            })
        return processed_results

    @staticmethod
    def _build_evidence_weights(evidences: List[Dict]) -> Tuple[List[str], Dict[str, float]]:
        """Build unique contribution keys and weights that sum to one.

        Multiple images can refer to the same source text ``record_id``.  A
        plain ``record_id -> weight`` mapping would overwrite earlier entries,
        so duplicate ids receive deterministic suffixes.
        """
        if not evidences:
            return [], {}
        lengths = [max(0, int(item.get('evidence_length') or 0)) for item in evidences]
        total_length = sum(lengths)
        if total_length <= 0:
            # Empty evidence/explanation is still a valid contribution.
            lengths = [1] * len(evidences)
            total_length = len(evidences)

        contribution_ids: List[str] = []
        weights: Dict[str, float] = {}
        used = set()
        running = 0.0
        for index, (item, length) in enumerate(zip(evidences, lengths)):
            base = str(item.get('record_id') or f'record_{index + 1}')
            contribution_id = base
            suffix = 2
            while contribution_id in used:
                contribution_id = f'{base}#{suffix}'
                suffix += 1
            used.add(contribution_id)
            contribution_ids.append(contribution_id)

            if index == len(evidences) - 1:
                weight = max(0.0, 1.0 - running)
            else:
                weight = float(length) / float(total_length)
                running += weight
            weights[contribution_id] = weight
        return contribution_ids, weights

    def _save_evaluation_process(self, batch_results: List[Dict], batch_summary: Dict = None, 
                               process_file: str = None):
        """保存评估过程到指定的process_file"""
        if not process_file:
            process_file = self.process_file
        if not process_file:
            return True

        try:
            # 确保目录存在（失败时与写入失败一样记录为输出错误）
            os.makedirs(os.path.dirname(process_file) or ".", exist_ok=True)
            # 1. 读取现有数据（仅限本次会话已写入的文件；新会话从空日志开始）
            all_data = {"batches": [], "run_id": getattr(self, "_run_id", None)}
            session_files = getattr(self, "_session_process_files", None)
            if session_files is None:
                session_files = self._session_process_files = set()
            first_write = process_file not in session_files
            session_files.add(process_file)
            if os.path.exists(process_file) and not first_write:
                try:
                    with open(process_file, 'r', encoding='utf-8') as f:
                        all_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: 无法读取现有评估文件 {process_file}，将创建新文件")
            
            # 2. 创建新的batch记录
            if batch_results:  # 只在有结果时创建新记录
                current_batch_id = len(all_data.get('batches', [])) + 1
                batch_process = {
                    'batch_id': current_batch_id,
                    'evaluations': [],
                    'batch_summary': {}
                }
                
                # 收集所有评估结果和evidence
                all_evidence = defaultdict(lambda: defaultdict(list))
                
                # 3. 处理每条评估结果
                for idx, result in enumerate(batch_results, 1):
                    evaluation = {
                        'record_id': result.get('metadata', {}).get('record_id', f"batch_{current_batch_id}_text_{idx}"),
                        'scores': {},
                        'optical_model_scores': result.get('optical_model_scores', {}),
                        'status': 'analyzed'
                    }
                    
                    # 为每个维度初始化所有特征词
                    for dim_name, traits in self.luxury_standards.items():
                        evaluation['scores'][dim_name] = {}
                        for trait in traits:
                            evaluation['scores'][dim_name][trait] = {
                                'score': None,
                                'evidence': '',
                                'explanation': ''
                            }
                    
                    # 处理实际得分
                    if result.get('dimension_scores'):
                        for dim_name, dim_data in result['dimension_scores'].items():
                            try:
                                score = float(dim_data.get('score'))
                            except (TypeError, ValueError):
                                score = None
                            evidence = dim_data.get('evidence', '')
                            explanation = dim_data.get('explanation', '')
                            evidence = str(evidence).strip() if evidence is not None else ''
                            explanation = str(explanation).strip() if explanation is not None else ''
                            
                            if score is not None and np.isfinite(score) and 0.0 <= score <= 1.0 and (evidence or explanation):
                                for trait in dim_data.get('matching_traits', []):
                                    # 确保特征词存在于标准中
                                    if trait in self.luxury_standards.get(dim_name, []):
                                        evaluation['scores'][dim_name][trait] = {
                                            'score': score,
                                            'evidence': evidence,
                                            'explanation': explanation
                                        }
                                        
                                        all_evidence[dim_name][trait].append({
                                            'record_id': evaluation['record_id'],
                                            'score': score,
                                            'evidence': evidence,
                                            # Explanations are valid support too.  Use them as a
                                            # fallback when evidence is empty to avoid zero division.
                                            'evidence_length': max(len(evidence), len(explanation), 0)
                                        })
                            else:
                                for trait in dim_data.get('matching_traits', []):
                                    if trait in self.luxury_standards.get(dim_name, []):
                                        evaluation['scores'][dim_name][trait].update({
                                            'score': None,
                                            'status': 'missing_evidence'
                                        })
                                        evaluation['status'] = 'partial'
                    else:
                        evaluation['status'] = 'failed'
                    
                    batch_process['evaluations'].append(evaluation)
                
                # 4. 初始化批次总结
                batch_process['batch_summary'] = {}
                for dim_name, traits in self.luxury_standards.items():
                    batch_process['batch_summary'][dim_name] = {}
                    for trait in traits:
                        batch_process['batch_summary'][dim_name][trait] = {
                            'score': None,
                            'evidence': '',
                            'contributing_records': [],
                            'evidence_weights': {},
                            'analyzed_count': 0,
                            'total_count': len(batch_results)
                        }
                
                # 5. 更新有得分的特征词
                for dim_name, traits in all_evidence.items():
                    for trait, evidences in traits.items():
                        if evidences:
                            try:
                                contribution_ids, weights = self._build_evidence_weights(evidences)
                                weighted_score = sum(
                                    evidence['score'] * weights[contribution_id]
                                    for evidence, contribution_id in zip(evidences, contribution_ids)
                                )
                                
                                combined_evidence = '; '.join(
                                    e['evidence']
                                    for e in evidences
                                )
                                
                                batch_process['batch_summary'][dim_name][trait].update({
                                    'score': round(weighted_score, 2),
                                    'evidence': combined_evidence,
                                    'contributing_records': [e['record_id'] for e in evidences],
                                    'evidence_weights': weights,
                                    'analyzed_count': len(evidences)
                                })
                            except Exception as e:
                                print(f"处理特征词 {trait} 时出错: {str(e)}")
                                continue
                
                # 6. 追加新的batch记录
                all_data['batches'].append(batch_process)
            
            # 7. 写入完整数据
            with open(process_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            return True

        except Exception as e:
            # A log that cannot be written is a reported failure, not a silent one:
            # the run status becomes "partial" and the report lists the file.
            print(f"Error saving evaluation process to {process_file}: {str(e)}")
            self.last_failures.append({"scope": "output", "file": process_file,
                                       "error": f"evaluation process log could not be written: {e}"})
            return False

    def _save_evaluation_result(self, final_result: Dict = None, output_folder: str = None):
        """保存最终评估结果到指定文件夹"""
        result_file = self.result_file
        if output_folder:
            result_file = os.path.join(output_folder, os.path.basename(self.result_file or "evaluation_result.json"))
        if not result_file:
            return None
            
        # 确保目录存在
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
            
        try:
            # Use an explicitly supplied result when available; otherwise calculate it.
            result = final_result if final_result is not None else self.memory_model.calculate_final_scores()
            if not isinstance(result, dict):
                raise ValueError("final_result must be a dictionary")
            
            # 写入结果文件
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            return result_file
                
        except Exception as e:
            print(f"Error saving evaluation result: {str(e)}")
            return None

    def _clean_text_content(self, content: str) -> str:
        """Normalize text without deleting potentially meaningful content.

        Bracketed phrases and ``#topic#`` spans are retained because they can
        carry brand or product semantics. Markup removal belongs to a separately
        documented preprocessing stage and must not happen silently here.
        """
        if content is None:
            return ""
        content = str(content)
        if not content:
            return ""

        # Replace control characters with a separator so adjacent words do not
        # become concatenated, then normalize common typographic variants.
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', ' ', content)
        punctuation = str.maketrans({
            '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u00ab': '"', '\u00bb': '"',
            '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u2032': "'",
            '\u300c': '"', '\u300d': '"', '\u300e': '"', '\u300f': '"',
            '\u2026': '...',
        })
        content = content.translate(punctuation)

        # Decode entities once (including &nbsp;, &quot; and &amp;) without
        # turning ordinary ampersands into missing text.
        content = html.unescape(content)
        return ' '.join(content.split())

    def process_single_file(self, file_path: str, output_dir: str = None) -> Dict:
        """处理单个JSONL文件并生成分析结果"""
        # 提取文件名并创建输出文件夹
        file_name = os.path.basename(file_path)
        brand_name = os.path.splitext(file_name)[0]
        
        # 如果文件名包含版本信息，去除版本号
        if "_v" in brand_name:
            brand_name = brand_name.split("_v")[0]
            
        # 创建输出文件夹
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(file_path), "analysis_results")
        
        brand_output_dir = os.path.join(output_dir, brand_name)
        os.makedirs(brand_output_dir, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"开始处理: {brand_name}")
        print(f"输入文件: {file_path}")
        print(f"输出目录: {brand_output_dir}")
        print(f"{'='*50}")
        
        # 重置模型状态，准备处理新文件
        self._reset_session()

        # 2. 加载数据
        try:
            raw_data = self.load_jsonl(file_path)
            if not raw_data:
                print(f"错误: 文件 {file_path} 无有效数据")
                return {
                    "brand": brand_name,
                    "status": "error",
                    "message": "无有效数据",
                    "total": 0,
                    "processed": 0,
                    "failed": 0,
                    "failed_count": 0,
                    "failure_details": []
                }
                
            raw_data = self._sort_records(raw_data)
            raw_data = self._apply_optional_preprocessing(raw_data)
            if not raw_data:
                return {
                    "brand": brand_name,
                    "status": "error",
                    "message": "预处理后无可用记录",
                    "total": 0,
                    "processed": 0,
                    "failed": 0,
                    "failed_count": 0,
                    "failure_details": [],
                    "preprocessing": self.preprocessing_summary,
                }
            print(f"加载了 {len(raw_data)} 条记录（已按可识别时间排序）")
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}")
            return {
                "brand": brand_name,
                "status": "error",
                "message": f"加载文件失败: {str(e)}",
                "total": 0,
                "processed": 0,
                "failed": 0,
                "failed_count": 0,
                "failure_details": [{"scope": "load", "error": str(e)}]
            }
        
        # 3. 批量处理
        all_results = []
        
        try:
            with tqdm(total=len(raw_data), desc=f"分析 {brand_name}") as pbar:
                for i in range(0, len(raw_data), self.batch_size):
                    batch = raw_data[i:i + self.batch_size]
                    batch_num = i//self.batch_size + 1
                    # analyze_batch 已经对每个项目执行带退避的 API 重试；
                    # 外层不再重复调用同一批次，避免成功记忆被重复写入。
                    # process_single_file already created the isolated session;
                    # do not reset here or accumulated failure details would
                    # be discarded at the first batch.
                    batch_results = self.analyze_batch(batch, brand_output_dir, reset=False)
                    if batch_results:
                        all_results.extend(batch_results)
                        pbar.update(len(batch_results))
                    else:
                        print(f"批次 {batch_num} 未返回结果；失败项已记录在 last_failures")
                    if self.retry_delay > 0:
                        time.sleep(min(self.retry_delay, 2.0))
            
            # 4. 生成并保存最终评分.  A non-empty result is not sufficient
            # for success: every input record must either be processed or be
            # listed explicitly in the failure report.
            failure_details = list(self.last_failures)
            processed_count = len(all_results)
            failed_count = len(failure_details)
            unaccounted = len(raw_data) - processed_count - failed_count
            if unaccounted > 0:
                failure_details.append({
                    "scope": "batch",
                    "error": "records were neither processed nor reported as failed",
                    "count": unaccounted,
                })
                failed_count += unaccounted

            if all_results:
                try:
                    # 执行反思
                    self.memory_model.reflection()
                    
                    # 保存最终评估结果
                    result_file = self._save_evaluation_result(None, brand_output_dir)
                    if result_file is None:
                        failure_details.append({"scope": "output", "error": "final evaluation result could not be written"})
                        failed_count += 1
                    else:
                        print(f"最终评估已保存至: {result_file}")
                    
                    # 复制模型文件和标准文件到输出目录，方便后续查看
                    try:
                        if self.model_file and os.path.exists(self.model_file):
                            shutil.copy(self.model_file, brand_output_dir)
                        if self.standard_file and os.path.exists(self.standard_file):
                            shutil.copy(self.standard_file, brand_output_dir)
                    except Exception as e:
                        print(f"复制辅助文件时出错: {str(e)}")
                        
                    if result_file is None:
                        status = "failed"
                    else:
                        status = (
                            "success" if processed_count == len(raw_data) and failed_count == 0
                            else "partial"
                        )
                    return {
                        "brand": brand_name,
                        "status": status,
                        "total": len(raw_data),
                        "processed": processed_count,
                        "failed": failed_count,
                        "failed_count": failed_count,
                        "failure_details": failure_details,
                        "preprocessing": self.preprocessing_summary,
                        "output_dir": brand_output_dir
                    }
                    
                except Exception as e:
                    print(f"保存最终评估时出错: {str(e)}")
                    return {
                        "brand": brand_name,
                        "status": "partial",
                        "message": f"保存最终评估失败: {str(e)}",
                        "total": len(raw_data),
                        "processed": processed_count,
                        "failed": failed_count,
                        "failed_count": failed_count,
                        "failure_details": failure_details,
                        "preprocessing": self.preprocessing_summary,
                        "output_dir": brand_output_dir
                    }
            else:
                print("无可生成最终评估的结果")
                return {
                    "brand": brand_name,
                    "status": "error",
                    "message": "处理后无结果",
                    "total": len(raw_data),
                    "processed": 0,
                    "failed": failed_count,
                    "failed_count": failed_count,
                    "failure_details": failure_details,
                    "preprocessing": self.preprocessing_summary,
                    "output_dir": brand_output_dir
                }
                
        except Exception as e:
            print(f"处理文件时发生错误: {str(e)}")
            return {
                "brand": brand_name,
                "status": "error",
                "message": f"处理失败: {str(e)}",
                "total": len(raw_data),
                "processed": len(all_results) if all_results else 0,
                "failed": len(self.last_failures),
                "failed_count": len(self.last_failures),
                "failure_details": list(self.last_failures),
                "preprocessing": self.preprocessing_summary,
                "output_dir": brand_output_dir
            }
    
    def batch_process_files(self, input_dir: str, output_dir: str = None, 
                           file_pattern: str = "*.jsonl", max_parallel: int = 1):
        """批量处理目录中的所有JSONL文件"""
        # 1. 查找所有匹配的文件
        input_pattern = os.path.join(input_dir, file_pattern)
        jsonl_files = sorted(glob.glob(input_pattern))
        
        if not jsonl_files:
            print(f"错误: 在 {input_dir} 中未找到匹配 {file_pattern} 的文件")
            return
            
        print(f"找到 {len(jsonl_files)} 个待处理文件:")
        for f in jsonl_files:
            print(f"  - {os.path.basename(f)}")
            
        # 2. 创建输出目录
        if not output_dir:
            output_dir = os.path.join(input_dir, "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 3. 处理每个文件
        results = []
        
        if max_parallel == 1:
            # 串行处理
            for file_path in jsonl_files:
                result = self.process_single_file(file_path, output_dir)
                results.append(result)
        else:
            # 共享记忆模型不是线程安全的；公开版保持品牌间状态隔离。
            print("max_parallel>1 is reduced to serial processing to avoid shared-state races")
            for file_path in jsonl_files:
                result = self.process_single_file(file_path, output_dir)
                results.append(result)
        # 4. 生成批处理报告
        report = {
            "total_files": len(jsonl_files),
            "success_count": sum(1 for r in results if r["status"] == "success"),
            "partial_count": sum(1 for r in results if r["status"] == "partial"),
            "error_count": sum(1 for r in results if r["status"] == "error"),
            "failed_count": sum(1 for r in results if r["status"] == "failed"),
            "total_processed": sum(r.get("processed", 0) for r in results),
            "total_failed": sum(r.get("failed", r.get("failed_count", 0)) for r in results),
            "failure_details": [
                {"file": r.get("brand"), "failures": r.get("failure_details", [])}
                for r in results if r.get("failure_details")
            ],
            "file_results": results
        }
        
        # 保存报告
        report_file = os.path.join(output_dir, "batch_process_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        # 5. 打印摘要
        print("\n" + "="*60)
        print("批处理完成摘要")
        print("="*60)
        print(f"总文件数: {report['total_files']}")
        print(f"成功处理: {report['success_count']}")
        print(f"部分成功: {report['partial_count']}")
        print(f"处理失败: {report['error_count']}")
        print(f"结果写入失败: {report['failed_count']}")
        print(f"总处理记录: {report['total_processed']}")
        print(f"详细报告: {report_file}")
        print("="*60)
        
        return report
    
    @staticmethod
    def load_jsonl(file_path):
        """加载JSONL文件的辅助方法"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
            return TextAnalysisAgent._sort_records(data)
        except Exception as e:
            print(f"Error loading JSONL file: {str(e)}")
            return []

    def _save_luxury_standards(self):
        """将奢侈品标准保存到JSON文件"""
        if not self.standard_file:
            return
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.standard_file), exist_ok=True)
            
            # 保存到JSON文件
            with open(self.standard_file, 'w', encoding='utf-8') as f:
                json.dump(self.luxury_standards, f, indent=2, ensure_ascii=False)
                
            print(f"奢侈品标准已保存到: {self.standard_file}")
        except Exception as e:
            print(f"保存奢侈品标准时出错: {str(e)}")

    def _load_luxury_standards(self):
        """从JSON文件加载奢侈品标准（如果存在）"""
        if not self.standard_file:
            return
        try:
            if self.standard_file and os.path.exists(self.standard_file):
                # 首先检查文件是否为空
                if os.path.getsize(self.standard_file) == 0:
                    print(f"标准文件为空，将使用默认标准并重新创建文件")
                    return
                    
                # 尝试不同的编码方式
                encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
                loaded_standards = None
                
                for encoding in encodings:
                    try:
                        with open(self.standard_file, 'r', encoding=encoding) as f:
                            loaded_standards = json.load(f)
                            break
                    except UnicodeDecodeError:
                        # 尝试下一种编码
                        continue
                    except json.JSONDecodeError:
                        # JSON解析错误，文件可能已损坏
                        print(f"标准文件格式不正确，将使用默认标准并重新创建文件")
                        # 备份损坏的文件
                        backup_file = f"{self.standard_file}.bak"
                        try:
                            shutil.copy2(self.standard_file, backup_file)
                            print(f"已将损坏的标准文件备份为: {backup_file}")
                        except Exception as e:
                            print(f"备份损坏的标准文件时出错: {str(e)}")
                        return
                
                if loaded_standards is None:
                    print(f"无法解析标准文件，尝试了多种编码但都失败了，将使用默认标准并重新创建文件")
                    return
                
                # 验证加载的标准是否符合预期格式
                if isinstance(loaded_standards, dict) and all(isinstance(traits, list) for traits in loaded_standards.values()):
                    self.luxury_standards = loaded_standards
                    print(f"已从文件加载奢侈品标准: {self.standard_file}")
                else:
                    print(f"加载的标准格式无效，将使用默认标准")
            else:
                print(f"标准文件不存在，将使用默认标准并创建文件")
        except Exception as e:
            print(f"加载奢侈品标准时出错: {str(e)}，将使用默认标准")

    def update_luxury_standards(self, new_standards: Dict[str, List[str]]):
        """更新奢侈品标准并保存到文件
        
        Args:
            new_standards: 新的奢侈品标准字典，格式为{维度: [特质列表]}
        
        Returns:
            bool: 更新是否成功
        """
        try:
            # 验证新标准的格式
            if not isinstance(new_standards, dict):
                print("新标准必须是字典格式")
                return False
                
            for dimension, traits in new_standards.items():
                if not isinstance(traits, list) or not all(isinstance(trait, str) for trait in traits):
                    print(f"维度 '{dimension}' 的特质必须是字符串列表")
                    return False
            
            # 更新标准
            old_standards = self.luxury_standards.copy()
            self.luxury_standards.update(new_standards)
            
            # 更新内存模型中的标准
            self.memory_model = MemoryReflectionModel(self.luxury_standards, shared_memory=self.shared_memory, participant_name=self.name + ".memory")
            
            # 保存到文件
            self._save_luxury_standards()
            
            print(f"奢侈品标准已更新，添加/修改了 {len(new_standards)} 个维度")
            return True
            
        except Exception as e:
            print(f"更新奢侈品标准时出错: {str(e)}")
            self.luxury_standards = old_standards  # 恢复原始标准
            return False

def main():
    """主函数：执行批量文本分析"""
    try:
        # 解析命令行参数（如果需要）
        import argparse
        parser = argparse.ArgumentParser(description='批量分析品牌相关文本')
        parser.add_argument('--input_dir', type=str, default=os.getenv("OPIM_INPUT_DIR", "."),
                            help='输入目录，包含要分析的JSONL文件')
        parser.add_argument('--output_dir', type=str,
                            default=os.getenv("OPIM_OUTPUT_DIR"),
                            help=('输出目录；未提供时优先使用 OPIM_OUTPUT_DIR，'
                                  '否则默认为 input_dir 下的 analysis_results 文件夹'))
        parser.add_argument('--pattern', type=str, default="*.jsonl",
                            help='文件匹配模式，默认为*.jsonl')
        parser.add_argument('--parallel', type=int, default=1,
                            help='并行处理的文件数，默认为1（串行处理）')
        args = parser.parse_args()
        
        # 初始化分析器。运行时文件必须写入用户指定的输出目录，
        # 不能把模块源码目录当作 project_root。
        runtime_root = args.output_dir or os.path.join(args.input_dir, "analysis_results")
        try:
            analyzer = TextAnalysisAgent(project_root=runtime_root)
        except Exception as e:
            print(f"Error initializing analyzer: {str(e)}")
            return
        
        # 执行批量处理
        analyzer.batch_process_files(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            file_pattern=args.pattern,
            max_parallel=args.parallel
        )
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 
