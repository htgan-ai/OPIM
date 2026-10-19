from typing import List, Dict, Set, Any
from datetime import datetime
from collections import defaultdict
import contextlib

class SharedMemoryHub:
    """消息共享中心，负责管理多个智能体之间的消息交互和数据存储"""
    
    def __init__(self):
        # 参与者管理
        self._participants: Set[str] = set()
        
        # 消息管理
        self._memories: Dict[str, List[Dict]] = defaultdict(list)
        self._broadcast_history: List[Dict] = []
        
        # 数据存储管理
        self._data_pool: Dict = defaultdict(dict)
        self._operation_history: List[Dict] = []
        
    # === 参与者管理 ===
    def add_participant(self, agent_name: str) -> None:
        """添加参与者"""
        self._participants.add(agent_name)
        
    def remove_participant(self, agent_name: str) -> None:
        """移除参与者"""
        self._participants.remove(agent_name)
    
    # === 消息广播功能 ===
    def broadcast(self, message: Dict) -> None:
        """广播消息给所有参与者"""
        timestamp = datetime.now().isoformat()
        # Keep caller-owned dictionaries immutable and give each participant
        # its own event object so later mutation cannot mix agent state.
        event = dict(message)
        event['timestamp'] = timestamp
        
        # 记录广播历史
        self._broadcast_history.append(dict(event))
        
        # 给每个参与者的记忆中添加消息
        for participant in self._participants:
            self._memories[participant].append(dict(event))
    
    def get_memory(self, agent_name: str, recent_n: int = None) -> List[Dict]:
        """获取指定智能体的记忆"""
        if agent_name not in self._participants:
            return []
            
        memories = self._memories[agent_name]
        if recent_n is not None:
            return memories[-recent_n:]
        return memories
    
    # === 数据存储功能 ===
    def store(self, key: str, value: Any, metadata: Dict = None) -> None:
        """存储数据到内存池
        
        Args:
            key: 数据的键
            value: 要存储的值
            metadata: 相关的元数据
        """
        timestamp = datetime.now().isoformat()
        
        memory_item = {
            'value': value,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        self._data_pool[key] = memory_item
        self._operation_history.append({
            'action': 'store',
            'key': key,
            'timestamp': timestamp,
            'metadata': metadata or {}
        })
        
        # 广播存储事件
        self.broadcast({
            'type': 'data_store',
            'key': key,
            'metadata': metadata or {}
        })
    
    def retrieve(self, key: str) -> Any:
        """从内存池中检索数据"""
        memory_item = self._data_pool.get(key)
        return memory_item['value'] if memory_item else None
    
    def get_all_data(self) -> Dict:
        """获取所有存储的数据"""
        return {
            k: v['value'] 
            for k, v in self._data_pool.items()
        }
    
    # === 清理功能 ===
    def clear(self, target: str = None, key: str = None) -> None:
        """清除数据
        
        Args:
            target: 'memory' 或 'data' 或 None (清除所有)
            key: 如果指定，只清除该键的数据
        """
        if target == 'memory' or target is None:
            if key:
                self._memories[key].clear()
            else:
                self._memories.clear()
                self._broadcast_history.clear()
                
        if target == 'data' or target is None:
            if key:
                self._data_pool.pop(key, None)
            else:
                self._data_pool.clear()
                self._operation_history.clear()
    
    # === 属性访问 ===
    @property
    def participants(self) -> Set[str]:
        """当前参与者集合"""
        return self._participants.copy()
    
    @property
    def broadcast_history(self) -> List[Dict]:
        """广播历史"""
        return self._broadcast_history.copy()
    
    # Controller compatibility helpers.
    def add_brand_analysis(self, key: str, value: Any) -> None:
        self.store(f"brand_analysis:{key}", value, {"scope": "brand"})

    def add_user_analysis(self, key: str, value: Any) -> None:
        self.store(f"user_analysis:{key}", value, {"scope": "user"})

    def add_correction(self, target: str, agent: str, correction: Any) -> None:
        self.store(f"correction:{target}:{agent}", correction, {"scope": "correction", "target": target, "agent": agent})

    def add_analysis(self, key: str, value: Any) -> None:
        self.store(f"analysis:{key}", value, {"scope": "analysis"})

    @property
    def operation_history(self) -> List[Dict]:
        """数据操作历史"""
        return self._operation_history.copy()


# Backward-compatible alias.
SharedMemoryPool = SharedMemoryHub

@contextlib.contextmanager
def memory_hub(participants: List[str] = None, announcement: Dict = None):
    """消息共享中心的上下文管理器
    
    Args:
        participants: 初始参与者列表
        announcement: 初始广播消息
    """
    hub = SharedMemoryHub()
    
    try:
        # 添加初始参与者
        if participants:
            for participant in participants:
                hub.add_participant(participant)
        
        # 广播初始消息
        if announcement:
            hub.broadcast(announcement)
            
        yield hub
        
    finally:
        # 清理资源
        hub.clear() 
