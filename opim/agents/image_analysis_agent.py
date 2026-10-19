"""
图像分析代理 (Image Analysis Agent)
用于分析图像文件夹，评估图像内容的奢侈品特性

此代理支持：
1. 处理本地图像文件夹
2. 批量分析图像内容
3. 使用记忆流-反思-反应模型评估结果
4. 生成品牌特性分析报告
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Set, Optional, Tuple
import glob
import shutil
import uuid
import base64
import mimetypes
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
from PIL import Image
import io

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from opim.api_config import api_endpoint, api_key, api_model
except ImportError:
    from api_config import api_endpoint, api_key, api_model

try:
    from opim.models.memory_reflection_model_pic import MemoryReflectionModel
    from opim.models.shared_memory import SharedMemoryHub
except ImportError:
    from models.memory_reflection_model_pic import MemoryReflectionModel
    from models.shared_memory import SharedMemoryHub

class ImageAnalysisAgent:
    def __init__(self, project_root: Optional[str] = None, name: str = "ImageAnalysisAgent", shared_memory: Optional[SharedMemoryHub] = None):
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
        self.max_retries = max(1, int(os.getenv("OPIM_MAX_RETRIES", "6")))
        self.retry_delay = float(os.getenv("OPIM_RETRY_DELAY", "5"))
        
        # 添加文件路径配置
        self.process_file = (os.path.join(project_root, 'tests', 'results', 'evaluation_results', 'evaluation_process.json')
                             if project_root else None)
        self.result_file = (os.path.join(project_root, 'tests', 'results', 'evaluation_results', 'evaluation_result.json')
                            if project_root else None)
        
        # 添加模型文件路径 (解决引用错误)
        self.model_file = (os.path.join(project_root, 'models', 'memory_reflection_model_pic.py')
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
        self._manifest_cache = {}
        
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
    def _image_datetime(record: Dict) -> Optional[datetime]:
        metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
        values = [record.get(key) for key in (
            "publish_date_date", "timestamp", "created_at", "publish_time", "publish_date", "date"
        )]
        if isinstance(metadata, dict):
            values.extend(metadata.get(key) for key in (
                "publish_date_date", "timestamp", "created_at", "publish_time", "publish_date", "date"
            ))
        for value in values:
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

    def _load_image_manifest(self, folder_path: str) -> Dict[str, Dict]:
        """Load the release image manifest once and index it by normalized path."""
        # Walk up from this file until a directory containing data/xiaohongshu is found.
        package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        probe = package_root
        for _ in range(4):
            if os.path.isdir(os.path.join(probe, "data", "xiaohongshu")):
                package_root = probe
                break
            probe = os.path.dirname(probe)
        folder_abs = os.path.abspath(folder_path)
        xhs_image_root = os.path.abspath(os.path.join(package_root, "data", "xiaohongshu", "images"))
        candidates = []
        # The package-level manifest belongs only to the Xiaohongshu image
        # release.  Do not apply it to a Weibo folder merely because a file
        # happens to share the same basename.
        try:
            inside_xhs_release = os.path.commonpath([folder_abs, xhs_image_root]) == xhs_image_root
        except ValueError:
            inside_xhs_release = False
        if inside_xhs_release:
            candidates.append(os.path.join(package_root, "data", "xiaohongshu", "pairs.jsonl"))
        candidates.extend([
            os.path.join(os.path.dirname(folder_abs), "pairs.jsonl"),
            os.path.join(os.path.dirname(os.path.dirname(folder_abs)), "pairs.jsonl"),
        ])
        manifest_path = next((path for path in candidates if os.path.isfile(path)), None)
        if not manifest_path:
            return {}
        if manifest_path in self._manifest_cache:
            return self._manifest_cache[manifest_path]
        exact = {}
        by_brand_name = {}
        by_basename = {}
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if not isinstance(item, dict) or not item.get("image_path"):
                        continue
                    source_path = str(item["image_path"]).replace("\\", "/")
                    normalized = os.path.normpath(source_path).replace(os.sep, "/")
                    exact[normalized] = item
                    basename = os.path.basename(normalized)
                    by_basename.setdefault(basename, []).append(item)
                    brand = str(item.get("brand", ""))
                    if brand:
                        by_brand_name[(brand, os.path.basename(normalized))] = item
        except (OSError, UnicodeError):
            return {}
        # A basename is safe only when unique; duplicate names are resolved by brand.
        for basename, records in by_basename.items():
            if len(records) == 1:
                exact[basename] = records[0]
        indexed = {"exact": exact, "by_brand_name": by_brand_name, "package_root": package_root}
        self._manifest_cache[manifest_path] = indexed
        return indexed

    def _manifest_record_for_image(self, image_path: str, brand_name: str, manifest: Dict) -> Optional[Dict]:
        if not manifest:
            return None
        package_root = manifest.get("package_root", "")
        exact = manifest.get("exact", {})
        normalized = os.path.normpath(os.path.abspath(image_path)).replace(os.sep, "/")
        relative = os.path.relpath(os.path.abspath(image_path), package_root).replace(os.sep, "/") if package_root else ""
        for key in (normalized, relative, "data/" + relative if relative else "", os.path.basename(image_path)):
            if key and key in exact:
                return exact[key]
        return manifest.get("by_brand_name", {}).get((brand_name, os.path.basename(image_path)))

    @classmethod
    def _sort_images(cls, images: List[Dict]) -> List[Dict]:
        indexed = list(enumerate(images or []))
        def source_line(record: Dict) -> int:
            metadata = record.get('metadata', {}) if isinstance(record, dict) else {}
            value = record.get('source_line_no') if isinstance(record, dict) else None
            if value is None and isinstance(metadata, dict):
                value = metadata.get('source_line_no')
            try:
                return int(value)
            except (TypeError, ValueError):
                return 2**63 - 1
        return [record for _, record in sorted(
            indexed,
            key=lambda item: (
                cls._image_datetime(item[1]) is None,
                cls._image_datetime(item[1]) or datetime.max,
                source_line(item[1]),
                str(item[1].get("image_path", "")),
                item[0],
            ),
        )]

    def analyze_batch(self, images: List[Dict], output_folder: str = None, reset: bool = True) -> List[Dict]:
        """分析一个batch的图像，并将结果保存到指定文件夹"""
        try:
            if reset or not self._active_session:
                self._reset_session()
            images = self._sort_images(list(images or []))
            batch_results = []
            failed_images = []
            
            # 1. 获取每张图像的API分析结果
            for image_data in images:
                success = False
                last_error = ""
                for attempt in range(self.max_retries):
                    try:
                        # 检查图像路径或图像数据是否存在
                        image_path = image_data.get('image_path', '')
                        image_url = image_data.get('image_url', '')
                        image_base64 = image_data.get('image_base64', '')
                        
                        # 确保至少有一种图像数据可用
                        if not (image_path or image_url or image_base64):
                            print(f"错误: 图像数据不完整，缺少路径或URL或Base64数据")
                            break
                        
                        # 处理图像数据并创建提示词
                        if image_path and os.path.exists(image_path):
                            # 如果提供了本地图像路径，转换为Base64
                            try:
                                image_base64 = self._image_to_base64(image_path)
                            except Exception as e:
                                print(f"图像转换失败: {str(e)}")
                                break
                        
                        # 创建分析提示
                        prompt = self._create_batch_prompt(image_url=image_url, image_base64=image_base64)
                        response = self._call_api(prompt, image_url=image_url, image_base64=image_base64, mime_type=mimetypes.guess_type(image_path)[0] if image_path else "image/jpeg")
                        
                        result = None
                        if response:
                            result = self._process_api_response(response)
                        
                        # API/解析失败时重试；失败结果不进入记忆流。
                        if not result or 'results' not in result:
                            raise ValueError("API response did not contain a valid results list")
                        
                        # 处理API响应并提取结果
                        processed_result = self._process_batch_response(result, [image_data])
                        
                        if processed_result:
                            batch_results.extend(processed_result)
                            success = True
                            break
                        raise ValueError("image response produced no valid result")
                            
                    except Exception as e:
                        last_error = str(e)
                        print(f"处理图像失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                
                if not success:
                    failed_images.append(image_data)
                    self.last_failures.append({
                        "image_path": image_data.get("image_path", ""),
                        "error": last_error or "API response or image result parsing failed",
                    })
            
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
            # 返回空列表，但在调用方会再次尝试
            return []

    def analyze(self, images: List[Dict], output_folder: Optional[str] = None) -> Dict:
        """Run the complete three-stage evaluation for in-memory image records.

        Records are processed in batches of ``self.batch_size``; each batch is
        one time step t of the interference stage, so the trend-based
        quantities need at least two batches.
        """
        self._reset_session()
        images = self._sort_images(list(images or []))   # global chronological order before batching
        batch_size = max(1, int(getattr(self, "batch_size", 10) or 10))
        results: List[Dict] = []
        for start in range(0, len(images), batch_size):
            results.extend(self.analyze_batch(images[start:start + batch_size], output_folder=output_folder, reset=False))
        if results:
            self.memory_model.reflection()
        final = self.memory_model.calculate_final_scores()
        dimension_scores = {}
        evidence = {"image_evidence": []}
        for dimension, traits in final.get("dimensions", {}).items():
            values = [item.get("final_score") for item in traits.values() if item.get("final_score") is not None]
            if values:
                dimension_scores[dimension] = float(np.mean(values))
            evidence["image_evidence"].extend(
                item.get("evidence_summary", "") for item in traits.values() if item.get("evidence_summary")
            )
        return {
            "dimension_scores": dimension_scores,
            "evidence": evidence,
            "results": results,
            "failed": list(self.last_failures),
            "final_scores": final,
        }

    def _call_api(self, prompt: str, image_url: str = "", image_base64: str = "", mime_type: str = "image/jpeg") -> Optional[Dict]:
        """Call an OpenAI-compatible multimodal endpoint with the complete image."""
        if not self.api_key:
            raise RuntimeError("OPIM_API_KEY is required for ImageAnalysisAgent")
        if image_base64:
            image_content = {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime_type, image_base64)}}
        elif image_url:
            image_content = {"type": "image_url", "image_url": {"url": image_url}}
        else:
            raise ValueError("image_url or image_base64 is required")
        payload = {
            "model": self.api_model,
            "messages": [
                {"role": "system", "content": "你是一个专业的图像分析助手，请严格按照JSON格式返回分析结果。"},
                {"role": "user", "content": [{"type": "text", "text": prompt}, image_content]},
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
            "top_p": 0.95,
        }
        last_error = None
        # The batch loop owns the complete per-record retry budget.  Keep one
        # HTTP request here so transport retries cannot multiply that budget.
        api_attempts = 1
        for attempt in range(api_attempts):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("API response must be an object")
                return data
            except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < api_attempts and self.retry_delay > 0:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError("image API failed after %s attempt(s): %s" % (api_attempts, last_error))

    def _process_api_response(self, response: Dict) -> Optional[Dict]:
        """Extract the JSON object returned by the model; malformed output fails the item."""
        if not isinstance(response, dict):
            return None
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        if not isinstance(content, str) or not content.strip():
            return None
        text = content.strip()
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            text = text[start:end + 1]
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _standardize_results(self, result: Dict) -> Optional[Dict]:
        """Validate the source response schema without filling missing scores."""
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

    def _create_batch_prompt(self, image_url: str = None, image_base64: str = None) -> str:
        """创建批量分析提示词，处理图像分析"""
        # 准备图像描述部分
        image_description = ""
        if image_url:
            image_description = f"请分析以下图像URL所示内容: {image_url}\n\n"
        elif image_base64:
            # 图像像素通过 multimodal image_url content block 发送；提示词不截断或伪装Base64。
            image_description = "请分析随请求附带的本地图像内容。\n\n"
        else:
            image_description = "请分析提供的图像内容\n\n"
            
        prompt = f"""{image_description}根据图像内容，返回指定格式的JSON。

评估维度及特征词:
{json.dumps(self.luxury_standards, indent=2, ensure_ascii=False)}

请按以下JSON格式返回结果，必须包含所有维度和所有匹配的特征词的光学参数:
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
- wave_amplitude and wavelength_purity: required API parameters in [0,1]
- photon_density, coherence, fringe_clarity, and diffraction_resolution are computed by the optical model.

注意事项:
1. 仅返回JSON格式，不要添加任何解释文本
2. 只有在图像中有明确视觉元素支持的特征词才评分
3. 每个维度的分数范围是0-1之间
4. 匹配特征词为空时使用空数组[]
5. 只为匹配的特征词提供 wave_amplitude 与 wavelength_purity，不要为未匹配特征词编造评分
"""
        return prompt

    def _process_batch_response(self, response_data: Dict, original_images: List[Dict]) -> List[Dict]:
        """Validate one model result and preserve only evidence-backed scores."""
        if not isinstance(response_data, dict):
            return []
        results = response_data.get("results")
        if not isinstance(results, list) or not results:
            return []
        if len(results) != len(original_images):
            return []
        processed_results = []
        for index, result in enumerate(results):
            if not isinstance(result, dict) or not isinstance(result.get("dimension_scores"), dict):
                continue
            dimensions = {}
            optical_scores = result.get("optical_scores", result.get("optical_model_scores", {}))
            if not isinstance(optical_scores, dict):
                continue
            try:
                for dimension, scores in result["dimension_scores"].items():
                    if dimension not in self.luxury_standards or not isinstance(scores, dict):
                        raise ValueError("unknown or malformed dimension: %s" % dimension)
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
                    for trait in traits:
                        key = "%s_%s" % (dimension, trait)
                        values = optical_scores.get(key)
                        if not isinstance(values, dict):
                            raise ValueError("missing optical scores for " + key)
                        memory_values = values.get("memory_stream", values)
                        reflection_values = values.get("reflection", values)
                        required_memory = ("wave_amplitude", "wavelength_purity")
                        if any(name not in memory_values for name in required_memory):
                            raise ValueError("incomplete optical scores for " + key)
                        memory = {name: float(memory_values[name]) for name in required_memory}
                        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in memory.values()):
                            raise ValueError("optical scores must be finite and in [0,1]")
                        reflection = {}
                        for name in ("photon_density", "coherence", "fringe_clarity", "diffraction_resolution"):
                            if name in memory_values:
                                value = float(memory_values[name])
                                if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                                    raise ValueError("optical scores must be finite and in [0,1]")
                                memory[name] = value
                            if name in reflection_values:
                                value = float(reflection_values[name])
                                if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                                    raise ValueError("optical scores must be finite and in [0,1]")
                                reflection[name] = value
                        optical_scores[key] = {"memory_stream": memory, "reflection": reflection}
                if not dimensions:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            image = original_images[index]
            processed_results.append({
                "image_path": image.get("image_path", ""),
                "image_url": image.get("image_url", ""),
                "dimension_scores": dimensions,
                "optical_model_scores": {key: value for key, value in optical_scores.items() if isinstance(value, dict)},
                "metadata": {
                    **(image.get("metadata", {}) if isinstance(image.get("metadata", {}), dict) else {}),
                    **({"record_id": image.get("record_id")} if image.get("record_id") else {}),
                },
            })
        return processed_results

    @staticmethod
    def _build_evidence_weights(evidences: List[Dict]) -> Tuple[List[str], Dict[str, float]]:
        """Build unique contribution keys and weights that sum to one.

        A source text can have multiple image contributions with the same
        ``record_id``.  Duplicate ids receive deterministic suffixes instead
        of overwriting one another in the JSON weight map.
        """
        if not evidences:
            return [], {}
        lengths = [max(0, int(item.get('evidence_length') or 0)) for item in evidences]
        total_length = sum(lengths)
        if total_length <= 0:
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
                        'record_id': result.get('metadata', {}).get('record_id', f"batch_{current_batch_id}_image_{idx}"),
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

    def _validate_image(self, image_path: str) -> bool:
        """验证图像是否可以正常打开"""
        if not image_path or not os.path.exists(image_path):
            return False
            
        try:
            with Image.open(image_path) as img:
                # 尝试读取图像基本属性，验证图像是否正常
                _ = img.size
                return True
        except Exception as e:
            print(f"图像验证失败: {str(e)}")
            return False

    def process_folder(self, folder_path: str, output_dir: str = None) -> Dict:
        """处理单个包含图像的文件夹并生成分析结果"""
        # 提取文件夹名称并创建输出文件夹
        folder_name = os.path.basename(folder_path)
        brand_name = folder_name
        
        # 创建输出文件夹
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(folder_path), "analysis_results")
        
        brand_output_dir = os.path.join(output_dir, brand_name)
        os.makedirs(brand_output_dir, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"开始处理: {brand_name}")
        print(f"输入文件夹: {folder_path}")
        print(f"输出目录: {brand_output_dir}")
        print(f"{'='*50}")
        
        # 重置模型状态，准备处理新文件夹
        self._reset_session()

        # 2. 加载图像文件
        try:
            # Load the release manifest when available so image records retain
            # record_id/publish_date_date instead of relying on filename order.
            manifest = self._load_image_manifest(folder_path)
            # 扫描文件夹中的所有图像文件
            image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
            image_files = []
            
            for file in sorted(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and file_path.lower().endswith(image_extensions):
                    image_files.append(file_path)
                    
            if not image_files:
                print(f"错误: 文件夹 {folder_path} 中没有图像文件")
                return {
                    "brand": brand_name,
                    "status": "error",
                    "message": "文件夹中没有图像文件",
                    "total": 0,
                    "processed": 0,
                    "failed": 0,
                    "failed_count": 0,
                    "failure_details": []
                }
                
            # 验证图像文件
            valid_images = []
            invalid_images = []
            for image_path in image_files:
                if self._validate_image(image_path):
                    # 创建图像数据结构
                    manifest_record = self._manifest_record_for_image(image_path, brand_name, manifest)
                    metadata = {
                        'filename': os.path.basename(image_path),
                        'folder': folder_name,
                        'tags': [folder_name],  # 使用文件夹名称作为标签
                    }
                    image_data = {
                        'image_path': image_path,
                        'image_url': '',  # 本地图像没有URL
                        'metadata': metadata,
                    }
                    if manifest_record:
                        for field in ("publish_date_date", "source_id", "source_line_no", "status"):
                            if manifest_record.get(field) is not None:
                                image_data[field] = manifest_record[field]
                                metadata[field] = manifest_record[field]
                        if manifest_record.get("record_id"):
                            image_data["record_id"] = manifest_record["record_id"]
                            metadata["record_id"] = manifest_record["record_id"]
                    valid_images.append(image_data)
                else:
                    print(f"警告: 图像文件无效或无法读取: {image_path}")
                    invalid_images.append(image_path)

            # Invalid files are input failures too; retain them in the report
            # so a run cannot be labelled complete merely because they were
            # omitted before the API stage.
            if invalid_images:
                self.last_failures.extend({
                    "image_path": path,
                    "stage": "image_validation",
                    "error": "image file is missing or cannot be decoded",
                } for path in invalid_images)
                    
            if not valid_images:
                print(f"错误: 没有有效的图像文件")
                return {
                    "brand": brand_name,
                    "status": "error",
                    "message": "没有有效的图像文件",
                    "total": len(image_files),
                    "processed": 0,
                    "failed": len(self.last_failures),
                    "failed_count": len(self.last_failures),
                    "failure_details": list(self.last_failures)
                }
                
            valid_images = self._sort_images(valid_images)
            print(f"加载了 {len(valid_images)} 个有效图像文件（已按清单时间排序；无时间时按稳定路径排序）")
        except Exception as e:
            print(f"加载文件夹 {folder_path} 时出错: {str(e)}")
            return {
                "brand": brand_name,
                "status": "error",
                "message": f"加载文件夹失败: {str(e)}",
                "total": 0,
                "processed": 0,
                "failed": 0,
                "failed_count": 0,
                "failure_details": [{"scope": "load", "error": str(e)}]
            }
        
        # 3. 批量处理
        all_results = []
        
        try:
            with tqdm(total=len(valid_images), desc=f"分析 {brand_name}") as pbar:
                for i in range(0, len(valid_images), self.batch_size):
                    batch = valid_images[i:i + self.batch_size]
                    batch_num = i//self.batch_size + 1
                    # analyze_batch 已经对每个项目执行带退避的 API 重试；
                    # 外层不再重复调用同一批次，避免成功记忆被重复写入。
                    # process_folder already created the isolated session;
                    # do not reset here or validation failures collected while
                    # loading the folder would be discarded at the first batch.
                    batch_results = self.analyze_batch(batch, brand_output_dir, reset=False)
                    if batch_results:
                        all_results.extend(batch_results)
                        pbar.update(len(batch_results))
                    else:
                        print(f"批次 {batch_num} 未返回结果；失败项已记录在 last_failures")
                    if self.retry_delay > 0:
                        time.sleep(min(self.retry_delay, 2.0))
            
            # 4. 生成并保存最终评分.  A non-empty result is not sufficient
            # for success: every input image must either be processed or be
            # listed explicitly in the failure report.
            failure_details = list(self.last_failures)
            processed_count = len(all_results)
            failed_count = len(failure_details)
            total_images = len(image_files)
            unaccounted = total_images - processed_count - failed_count
            if unaccounted > 0:
                failure_details.append({
                    "scope": "batch",
                    "error": "images were neither processed nor reported as failed",
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
                            "success" if processed_count == total_images and failed_count == 0
                            else "partial"
                        )
                    return {
                        "brand": brand_name,
                        "status": status,
                        "total": total_images,
                        "processed": processed_count,
                        "failed": failed_count,
                        "failed_count": failed_count,
                        "failure_details": failure_details,
                        "output_dir": brand_output_dir
                    }
                    
                except Exception as e:
                    print(f"保存最终评估时出错: {str(e)}")
                    return {
                        "brand": brand_name,
                        "status": "partial",
                        "message": f"保存最终评估失败: {str(e)}",
                        "total": total_images,
                        "processed": processed_count,
                        "failed": failed_count,
                        "failed_count": failed_count,
                        "failure_details": failure_details,
                        "output_dir": brand_output_dir
                    }
            else:
                print("无可生成最终评估的结果")
                return {
                    "brand": brand_name,
                    "status": "error",
                    "message": "处理后无结果",
                    "total": total_images,
                    "processed": 0,
                    "failed": failed_count,
                    "failed_count": failed_count,
                    "failure_details": failure_details,
                    "output_dir": brand_output_dir
                }
                
        except Exception as e:
            print(f"处理文件夹时发生错误: {str(e)}")
            return {
                "brand": brand_name,
                "status": "error",
                "message": f"处理失败: {str(e)}",
                "total": len(valid_images),
                "processed": len(all_results) if all_results else 0,
                "failed": len(self.last_failures),
                "failed_count": len(self.last_failures),
                "failure_details": list(self.last_failures),
                "output_dir": brand_output_dir
            }

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

    def _image_to_base64(self, image_path: str) -> str:
        """将图像转换为Base64编码字符串"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"转换图像到Base64失败: {str(e)}")
            raise

    def batch_process_folders(self, input_dir: str, output_dir: str = None, max_parallel: int = 1):
        """批量处理目录中的所有图像文件夹"""
        # 1. 查找所有子文件夹
        try:
            subfolders = sorted(f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f)))
            
            if not subfolders:
                print(f"错误: 在 {input_dir} 中未找到子文件夹")
                return
                
            print(f"找到 {len(subfolders)} 个待处理文件夹:")
            for f in subfolders:
                print(f"  - {f}")
                
            # 2. 创建输出目录
            if not output_dir:
                output_dir = os.path.join(input_dir, "analysis_results")
            os.makedirs(output_dir, exist_ok=True)
            
            # 3. 处理每个文件夹
            results = []
            
            if max_parallel == 1:
                # 串行处理
                for subfolder in subfolders:
                    folder_path = os.path.join(input_dir, subfolder)
                    result = self.process_folder(folder_path, output_dir)
                    results.append(result)
            else:
                # 共享记忆模型不是线程安全的；品牌之间保持串行状态隔离。
                print("max_parallel>1 is reduced to serial processing to avoid shared-state races")
                for subfolder in subfolders:
                    folder_path = os.path.join(input_dir, subfolder)
                    result = self.process_folder(folder_path, output_dir)
                    results.append(result)
            # 4. 生成批处理报告
            report = {
                "total_folders": len(subfolders),
                "success_count": sum(1 for r in results if r["status"] == "success"),
                "partial_count": sum(1 for r in results if r["status"] == "partial"),
                "error_count": sum(1 for r in results if r["status"] == "error"),
                "failed_count": sum(1 for r in results if r["status"] == "failed"),
                "total_processed": sum(r.get("processed", 0) for r in results),
                "total_failed": sum(r.get("failed", r.get("failed_count", 0)) for r in results),
                "failure_details": [
                    {"folder": r.get("brand"), "failures": r.get("failure_details", [])}
                    for r in results if r.get("failure_details")
                ],
                "folder_results": results
            }
            
            # 保存报告
            report_file = os.path.join(output_dir, "batch_process_report.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                
            # 5. 打印摘要
            print("\n" + "="*60)
            print("批处理完成摘要")
            print("="*60)
            print(f"总文件夹数: {report['total_folders']}")
            print(f"成功处理: {report['success_count']}")
            print(f"部分成功: {report['partial_count']}")
            print(f"处理失败: {report['error_count']}")
            print(f"结果写入失败: {report['failed_count']}")
            print(f"总处理图像: {report['total_processed']}")
            print(f"详细报告: {report_file}")
            print("="*60)
            
            return report
            
        except Exception as e:
            print(f"批量处理文件夹时出错: {str(e)}")
            return None

def main():
    """主函数：执行批量图像分析"""
    try:
        # 解析命令行参数
        import argparse
        parser = argparse.ArgumentParser(description='批量分析品牌相关图像文件夹')
        parser.add_argument('--input_dir', type=str, default=os.getenv("OPIM_INPUT_DIR", "."),
                            help='输入目录，包含要分析的品牌图像子文件夹')
        parser.add_argument('--output_dir', type=str, default=os.getenv("OPIM_OUTPUT_DIR", None),
                            help='输出目录，默认为input_dir下的analysis_results文件夹')
        parser.add_argument('--parallel', type=int, default=1,
                            help='并行处理的文件夹数，默认为1（串行处理）')
        parser.add_argument('--single_folder', type=str, default=None,
                            help='指定单个文件夹进行处理，而不是处理所有子文件夹')
        args = parser.parse_args()
        
        # 初始化分析器。运行时文件必须写入用户指定的输出目录，
        # 不能把模块源码目录当作 project_root。
        runtime_root = args.output_dir or os.path.join(args.input_dir, "analysis_results")
        try:
            analyzer = ImageAnalysisAgent(project_root=runtime_root)
        except Exception as e:
            print(f"初始化分析器时出错: {str(e)}")
            return
        
        # 执行处理
        if args.single_folder:
            # 处理单个指定文件夹
            single_folder_path = os.path.join(args.input_dir, args.single_folder) if not os.path.isabs(args.single_folder) else args.single_folder
            output_dir = args.output_dir or os.path.join(os.path.dirname(single_folder_path), "analysis_results")
            
            print(f"处理单个文件夹: {single_folder_path}")
            result = analyzer.process_folder(
                folder_path=single_folder_path,
                output_dir=output_dir
            )
            print(f"处理结果: {result}")
        else:
            # 批量处理所有子文件夹
            analyzer.batch_process_folders(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                max_parallel=args.parallel
            )
        
    except Exception as e:
        print(f"主函数执行错误: {str(e)}")

if __name__ == "__main__":
    main() 
