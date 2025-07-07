import time
import threading
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProgressInfo:
    """진행 상황 정보"""
    task_id: str
    task_name: str
    current: int = 0
    total: int = 100
    status: str = "pending"
    message: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def progress(self) -> float:
        """진행률 (0.0 ~ 1.0)"""
        if self.total == 0:
            return 0.0
        return min(self.current / self.total, 1.0)
    
    @property
    def percentage(self) -> int:
        """진행률 (0 ~ 100)"""
        return int(self.progress * 100)
    
    @property
    def elapsed_time(self) -> float:
        """경과 시간"""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def eta(self) -> Optional[float]:
        """예상 남은 시간"""
        if self.current == 0 or self.progress >= 1.0:
            return None
        
        elapsed = self.elapsed_time
        rate = self.current / elapsed
        remaining = self.total - self.current
        
        return remaining / rate if rate > 0 else None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "current": self.current,
            "total": self.total,
            "progress": self.progress,
            "percentage": self.percentage,
            "status": self.status,
            "message": self.message,
            "elapsed_time": self.elapsed_time,
            "eta": self.eta,
            "metadata": self.metadata
        }

class ProgressTracker:
    """진행 상황 추적기"""
    
    def __init__(self):
        self.tasks: Dict[str, ProgressInfo] = {}
        self.callbacks: Dict[str, Callable[[ProgressInfo], None]] = {}
        self.lock = threading.Lock()
        self.global_callback: Optional[Callable[[str, ProgressInfo], None]] = None
    
    def create_task(
        self,
        task_id: str,
        task_name: str,
        total: int = 100,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProgressInfo:
        """새 작업 생성"""
        with self.lock:
            progress = ProgressInfo(
                task_id=task_id,
                task_name=task_name,
                total=total,
                status="running",
                metadata=metadata or {}
            )
            self.tasks[task_id] = progress
            
            self._notify_callbacks(task_id, progress)
            return progress
    
    def update_task(
        self,
        task_id: str,
        current: Optional[int] = None,
        delta: Optional[int] = None,
        message: Optional[str] = None,
        status: Optional[str] = None,
        metadata_update: Optional[Dict[str, Any]] = None
    ) -> Optional[ProgressInfo]:
        """작업 진행 상황 업데이트"""
        with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"작업 {task_id}를 찾을 수 없습니다")
                return None
            
            progress = self.tasks[task_id]
            
            # 진행 상황 업데이트
            if current is not None:
                progress.current = min(current, progress.total)
            elif delta is not None:
                progress.current = min(progress.current + delta, progress.total)
            
            # 메시지 업데이트
            if message is not None:
                progress.message = message
            
            # 상태 업데이트
            if status is not None:
                progress.status = status
                if status in ["completed", "failed", "cancelled"]:
                    progress.end_time = time.time()
            
            # 메타데이터 업데이트
            if metadata_update:
                progress.metadata.update(metadata_update)
            
            # 자동 완료 처리
            if progress.current >= progress.total and progress.status == "running":
                progress.status = "completed"
                progress.end_time = time.time()
            
            self._notify_callbacks(task_id, progress)
            return progress
    
    def complete_task(self, task_id: str, message: str = "완료") -> Optional[ProgressInfo]:
        """작업 완료"""
        return self.update_task(
            task_id,
            current=self.tasks.get(task_id, ProgressInfo("", "", 0, 1)).total,
            status="completed",
            message=message
        )
    
    def fail_task(self, task_id: str, error_message: str) -> Optional[ProgressInfo]:
        """작업 실패"""
        return self.update_task(task_id, status="failed", message=error_message)
    
    def cancel_task(self, task_id: str) -> Optional[ProgressInfo]:
        """작업 취소"""
        return self.update_task(task_id, status="cancelled", message="취소됨")
    
    def get_task(self, task_id: str) -> Optional[ProgressInfo]:
        """작업 정보 조회"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, ProgressInfo]:
        """모든 작업 조회"""
        with self.lock:
            return self.tasks.copy()
    
    def get_active_tasks(self) -> Dict[str, ProgressInfo]:
        """활성 작업만 조회"""
        with self.lock:
            return {
                task_id: progress
                for task_id, progress in self.tasks.items()
                if progress.status == "running"
            }
    
    def clear_completed_tasks(self):
        """완료된 작업 제거"""
        with self.lock:
            completed_ids = [
                task_id
                for task_id, progress in self.tasks.items()
                if progress.status in ["completed", "failed", "cancelled"]
            ]
            
            for task_id in completed_ids:
                del self.tasks[task_id]
                if task_id in self.callbacks:
                    del self.callbacks[task_id]
    
    def register_callback(
        self,
        task_id: str,
        callback: Callable[[ProgressInfo], None]
    ):
        """작업별 콜백 등록"""
        with self.lock:
            self.callbacks[task_id] = callback
    
    def set_global_callback(
        self,
        callback: Callable[[str, ProgressInfo], None]
    ):
        """전역 콜백 설정"""
        self.global_callback = callback
    
    def _notify_callbacks(self, task_id: str, progress: ProgressInfo):
        """콜백 호출"""
        # 작업별 콜백
        if task_id in self.callbacks:
            try:
                self.callbacks[task_id](progress)
            except Exception as e:
                logger.error(f"콜백 실행 오류: {e}")
        
        # 전역 콜백
        if self.global_callback:
            try:
                self.global_callback(task_id, progress)
            except Exception as e:
                logger.error(f"전역 콜백 실행 오류: {e}")

class ProgressContext:
    """진행 상황 컨텍스트 매니저"""
    
    def __init__(
        self,
        tracker: ProgressTracker,
        task_id: str,
        task_name: str,
        total: int = 100,
        auto_update: bool = True
    ):
        self.tracker = tracker
        self.task_id = task_id
        self.task_name = task_name
        self.total = total
        self.auto_update = auto_update
        self.progress: Optional[ProgressInfo] = None
    
    def __enter__(self) -> 'ProgressContext':
        self.progress = self.tracker.create_task(
            self.task_id,
            self.task_name,
            self.total
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.tracker.complete_task(self.task_id)
        else:
            self.tracker.fail_task(
                self.task_id,
                f"오류 발생: {exc_val}"
            )
    
    def update(self, current: Optional[int] = None, delta: int = 1, message: str = ""):
        """진행 상황 업데이트"""
        if self.auto_update and current is None:
            self.tracker.update_task(self.task_id, delta=delta, message=message)
        else:
            self.tracker.update_task(self.task_id, current=current, message=message)
    
    def set_message(self, message: str):
        """메시지만 업데이트"""
        self.tracker.update_task(self.task_id, message=message)

# 전역 진행 상황 추적기
progress_tracker = ProgressTracker()

def track_progress(task_name: str, total: int = 100):
    """데코레이터: 함수 실행 진행 상황 추적"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import uuid
            task_id = str(uuid.uuid4())
            
            with ProgressContext(progress_tracker, task_id, task_name, total) as ctx:
                # 함수에 progress 인자 전달
                if 'progress' in func.__code__.co_varnames:
                    kwargs['progress'] = ctx
                
                return func(*args, **kwargs)
        
        return wrapper
    return decorator