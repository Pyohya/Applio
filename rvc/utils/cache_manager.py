import os
import time
import pickle
import hashlib
import threading
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Callable
import json

class CacheManager:
    """성능 향상을 위한 통합 캐시 관리자"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, Any] = {}
        self.cache_lock = threading.Lock()
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """캐시 디렉토리 생성"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_key(self, key: str, args: tuple = (), kwargs: Optional[dict] = None) -> str:
        """캐시 키 생성"""
        if kwargs is None:
            kwargs = {}
        key_data = f"{key}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_file_mtime(self, filepath: str) -> float:
        """파일 수정 시간 반환"""
        try:
            return os.path.getmtime(filepath)
        except:
            return 0
    
    def cache_file_list(self, directory: str, extensions: tuple, 
                       cache_duration: int = 300) -> List[str]:
        """파일 리스트 캐싱 (5분 기본)"""
        cache_key = self._get_cache_key("file_list", (directory, extensions))
        
        with self.cache_lock:
            # 메모리 캐시 확인
            if cache_key in self.memory_cache:
                cached_data, timestamp = self.memory_cache[cache_key]
                if time.time() - timestamp < cache_duration:
                    return cached_data
            
            # 실제 파일 리스트 생성
            files = []
            if os.path.exists(directory):
                for root, _, filenames in os.walk(directory):
                    for filename in filenames:
                        if filename.endswith(extensions):
                            files.append(os.path.join(root, filename))
            
            # 캐시 저장
            self.memory_cache[cache_key] = (sorted(files), time.time())
            return sorted(files)
    
    def cache_model_info(self, model_path: str) -> Optional[Dict]:
        """모델 정보 캐싱"""
        if not os.path.exists(model_path):
            return None
            
        cache_key = self._get_cache_key("model_info", (model_path,))
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # 캐시 파일이 모델보다 최신인지 확인
        if os.path.exists(cache_file):
            if self._get_file_mtime(cache_file) > self._get_file_mtime(model_path):
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass
        
        return None
    
    def save_model_info(self, model_path: str, info: Dict):
        """모델 정보 저장"""
        cache_key = self._get_cache_key("model_info", (model_path,))
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(info, f)
        except:
            pass
    
    def invalidate_file_cache(self):
        """파일 캐시 무효화"""
        with self.cache_lock:
            keys_to_remove = [k for k in self.memory_cache.keys() 
                            if k.startswith(self._get_cache_key("file_list", ())[:8])]
            for key in keys_to_remove:
                del self.memory_cache[key]

def cached_property(func: Callable) -> property:
    """프로퍼티 캐싱 데코레이터"""
    attr_name = f'_cached_{func.__name__}'
    
    @wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return property(wrapper)

# 전역 캐시 매니저 인스턴스
cache_manager = CacheManager()