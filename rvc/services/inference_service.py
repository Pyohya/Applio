import asyncio
import concurrent.futures
import queue
import threading
import torch
import gc
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
import time
import traceback

@dataclass
class InferenceRequest:
    """추론 요청 데이터 클래스"""
    request_id: str
    audio_path: str
    model_name: str
    params: Dict[str, Any]
    callback: Optional[Callable] = None
    priority: int = 0
    
class GPUMemoryManager:
    """GPU 메모리 관리자"""
    
    def __init__(self, max_memory_usage: float = 0.9):
        self.max_memory_usage = max_memory_usage
        self.loaded_models: Dict[str, Any] = {}
        self.model_usage: Dict[str, float] = {}
        self.lock = threading.Lock()
        
    def get_gpu_memory_usage(self) -> float:
        """현재 GPU 메모리 사용률 반환"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        return 0.0
    
    def can_load_model(self, model_size_estimate: float) -> bool:
        """모델을 로드할 수 있는지 확인"""
        current_usage = self.get_gpu_memory_usage()
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory
            estimated_usage = (torch.cuda.memory_allocated() + model_size_estimate) / available_memory
            return estimated_usage < self.max_memory_usage
        return True
    
    def cleanup_memory(self):
        """GPU 메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def unload_least_used_model(self):
        """가장 적게 사용된 모델 언로드"""
        with self.lock:
            if not self.loaded_models:
                return
                
            # LRU 방식으로 모델 선택
            least_used = min(self.model_usage.items(), key=lambda x: x[1])[0]
            if least_used in self.loaded_models:
                del self.loaded_models[least_used]
                del self.model_usage[least_used]
                self.cleanup_memory()

class InferenceService:
    """비동기 추론 서비스"""
    
    def __init__(self, max_workers: int = 2, max_queue_size: int = 100):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.request_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.memory_manager = GPUMemoryManager()
        self.is_running = False
        self.worker_threads = []
        self.results_cache = {}
        self.active_requests = {}
        
    def start(self):
        """서비스 시작"""
        self.is_running = True
        for i in range(self.executor._max_workers):
            thread = threading.Thread(target=self._worker, name=f"InferenceWorker-{i}")
            thread.start()
            self.worker_threads.append(thread)
    
    def stop(self):
        """서비스 중지"""
        self.is_running = False
        # 큐에 종료 신호 전송
        for _ in range(len(self.worker_threads)):
            self.request_queue.put((float('inf'), None))
        # 워커 스레드 종료 대기
        for thread in self.worker_threads:
            thread.join()
        self.executor.shutdown(wait=True)
    
    def _worker(self):
        """워커 스레드 메인 루프"""
        while self.is_running:
            try:
                priority, request = self.request_queue.get(timeout=1)
                if request is None:  # 종료 신호
                    break
                    
                # 추론 실행
                result = self._process_request(request)
                
                # 콜백 실행
                if request.callback:
                    request.callback(request.request_id, result)
                    
                # 결과 저장
                self.results_cache[request.request_id] = result
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                traceback.print_exc()
    
    def _process_request(self, request: InferenceRequest) -> Dict[str, Any]:
        """추론 요청 처리"""
        start_time = time.time()
        
        try:
            # GPU 메모리 체크 및 정리
            if self.memory_manager.get_gpu_memory_usage() > 0.8:
                self.memory_manager.cleanup_memory()
            
            # 모델 로드 또는 캐시에서 가져오기
            model = self._get_or_load_model(request.model_name)
            
            # 실제 추론 실행 (이 부분은 실제 RVC 추론 코드로 대체)
            result = self._run_inference(model, request.audio_path, request.params)
            
            # 모델 사용 시간 업데이트
            self.memory_manager.model_usage[request.model_name] = time.time()
            
            return {
                'success': True,
                'result': result,
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _get_or_load_model(self, model_name: str) -> Any:
        """모델 로드 또는 캐시에서 반환"""
        if model_name in self.memory_manager.loaded_models:
            return self.memory_manager.loaded_models[model_name]
        
        # 메모리 부족시 다른 모델 언로드
        estimated_size = 500 * 1024 * 1024  # 500MB 추정
        while not self.memory_manager.can_load_model(estimated_size):
            self.memory_manager.unload_least_used_model()
        
        # 모델 로드 (실제 구현에서는 RVC 모델 로드)
        model = self._load_model_impl(model_name)
        self.memory_manager.loaded_models[model_name] = model
        self.memory_manager.model_usage[model_name] = time.time()
        
        return model
    
    def _load_model_impl(self, model_name: str) -> Any:
        """실제 모델 로드 구현 (placeholder)"""
        # 여기에 실제 RVC 모델 로드 코드 구현
        return {"name": model_name, "loaded": True}
    
    def _run_inference(self, model: Any, audio_path: str, params: Dict) -> Any:
        """실제 추론 실행 (placeholder)"""
        # 여기에 실제 RVC 추론 코드 구현
        return {"audio": audio_path, "converted": True}
    
    async def submit_request_async(self, request: InferenceRequest) -> str:
        """비동기 요청 제출"""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        def callback(request_id: str, result: Dict):
            loop.call_soon_threadsafe(future.set_result, result)
        
        request.callback = callback
        self.request_queue.put((-request.priority, request))
        
        return await future
    
    def submit_request(self, request: InferenceRequest) -> str:
        """동기 요청 제출"""
        self.request_queue.put((-request.priority, request))
        return request.request_id
    
    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[Dict]:
        """결과 조회"""
        start_time = time.time()
        while True:
            if request_id in self.results_cache:
                return self.results_cache.pop(request_id)
            
            if timeout and (time.time() - start_time) > timeout:
                return None
                
            time.sleep(0.1)

# 배치 처리를 위한 헬퍼 클래스
class BatchProcessor:
    """배치 처리 최적화"""
    
    def __init__(self, batch_size: int = 4, timeout: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_requests: List[InferenceRequest] = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
    def add_request(self, request: InferenceRequest):
        """배치에 요청 추가"""
        with self.lock:
            self.pending_requests.append(request)
            if len(self.pending_requests) >= self.batch_size:
                self.condition.notify()
    
    def get_batch(self) -> List[InferenceRequest]:
        """배치 가져오기"""
        with self.lock:
            # 배치가 차거나 타임아웃까지 대기
            self.condition.wait_for(
                lambda: len(self.pending_requests) >= self.batch_size,
                timeout=self.timeout
            )
            
            batch = self.pending_requests[:self.batch_size]
            self.pending_requests = self.pending_requests[self.batch_size:]
            return batch