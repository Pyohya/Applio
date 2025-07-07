import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml

@dataclass
class InferenceConfig:
    """추론 설정"""
    pitch: int = 0
    filter_radius: int = 3
    index_rate: float = 0.75
    volume_envelope: float = 1.0
    protect: float = 0.5
    hop_length: int = 128
    f0_method: str = "rmvpe"
    split_audio: bool = False
    f0_autotune: bool = False
    f0_autotune_strength: float = 1.0
    clean_audio: bool = False
    clean_strength: float = 0.5
    export_format: str = "WAV"
    embedder_model: str = "contentvec"
    embedder_model_custom: Optional[str] = None
    sid: int = 0
    
    # 후처리 효과
    post_process: bool = False
    formant_shifting: bool = False
    formant_qfrency: float = 1.0
    formant_timbre: float = 1.0
    
    # 오디오 효과 설정
    reverb: bool = False
    reverb_room_size: float = 0.5
    reverb_damping: float = 0.5
    reverb_wet_gain: float = 0.5
    reverb_dry_gain: float = 0.5
    reverb_width: float = 0.5
    reverb_freeze_mode: float = 0.0
    
    pitch_shift: bool = False
    pitch_shift_semitones: float = 0.0
    
    limiter: bool = False
    limiter_threshold: float = -6.0
    limiter_release_time: float = 0.01
    
    gain: bool = False
    gain_db: float = 0.0
    
    distortion: bool = False
    distortion_gain: float = 25.0
    
    chorus: bool = False
    chorus_rate: float = 1.0
    chorus_depth: float = 0.25
    chorus_center_delay: float = 7.0
    chorus_feedback: float = 0.0
    chorus_mix: float = 0.5
    
    bitcrush: bool = False
    bitcrush_bit_depth: int = 8
    
    clipping: bool = False
    clipping_threshold: float = -6.0
    
    compressor: bool = False
    compressor_threshold: float = 0.0
    compressor_ratio: float = 1.0
    compressor_attack: float = 1.0
    compressor_release: float = 100.0
    
    delay: bool = False
    delay_seconds: float = 0.5
    delay_feedback: float = 0.0
    delay_mix: float = 0.5

@dataclass
class TrainingConfig:
    """학습 설정"""
    batch_size: int = 8
    total_epoch: int = 100
    save_every_epoch: int = 10
    save_only_latest: bool = False
    save_every_weights: bool = False
    sample_rate: int = 40000
    pretrained: bool = True
    cleanup: bool = False
    cache_data_in_gpu: bool = False
    custom_pretrained: bool = False
    g_pretrained_path: Optional[str] = None
    d_pretrained_path: Optional[str] = None
    overtraining_detector: bool = False
    overtraining_threshold: int = 50
    
@dataclass 
class AppConfig:
    """애플리케이션 전체 설정"""
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # UI 설정
    theme: str = "ParityError/Interstellar"
    language: str = "en_US"
    port: int = 6969
    share: bool = False
    
    # 성능 설정
    max_workers: int = 2
    cache_enabled: bool = True
    cache_duration: int = 300
    gpu_memory_limit: float = 0.9
    
    # 경로 설정
    models_dir: str = "logs"
    audio_dir: str = "assets/audios"
    presets_dir: str = "assets/presets"
    cache_dir: str = ".cache"
    
class ConfigManager:
    """구성 관리자"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self._ensure_directories()
        
    def _load_config(self) -> AppConfig:
        """설정 파일 로드"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.endswith('.yaml'):
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)
                
                # 중첩된 설정 처리
                if 'inference' in data:
                    data['inference'] = InferenceConfig(**data['inference'])
                if 'training' in data:
                    data['training'] = TrainingConfig(**data['training'])
                    
                return AppConfig(**data)
            except Exception as e:
                print(f"설정 파일 로드 실패: {e}")
                
        return AppConfig()
    
    def save_config(self):
        """설정 파일 저장"""
        try:
            config_dict = asdict(self.config)
            
            # InferenceConfig와 TrainingConfig도 dict로 변환
            if isinstance(self.config.inference, InferenceConfig):
                config_dict['inference'] = asdict(self.config.inference)
            if isinstance(self.config.training, TrainingConfig):
                config_dict['training'] = asdict(self.config.training)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml'):
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                    
            print(f"설정 저장 완료: {self.config_path}")
        except Exception as e:
            print(f"설정 저장 실패: {e}")
    
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        dirs = [
            self.config.models_dir,
            self.config.audio_dir,
            self.config.presets_dir,
            self.config.cache_dir
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_inference_config(self) -> InferenceConfig:
        """추론 설정 반환"""
        return self.config.inference
    
    def get_training_config(self) -> TrainingConfig:
        """학습 설정 반환"""
        return self.config.training
    
    def update_inference_config(self, **kwargs):
        """추론 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config.inference, key):
                setattr(self.config.inference, key, value)
        self.save_config()
    
    def update_training_config(self, **kwargs):
        """학습 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config.training, key):
                setattr(self.config.training, key, value)
        self.save_config()
    
    def load_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """프리셋 로드"""
        preset_path = os.path.join(self.config.presets_dir, f"{preset_name}.json")
        if os.path.exists(preset_path):
            try:
                with open(preset_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def save_preset(self, preset_name: str, preset_data: Dict[str, Any]):
        """프리셋 저장"""
        preset_path = os.path.join(self.config.presets_dir, f"{preset_name}.json")
        try:
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
            return True
        except:
            return False
    
    def list_presets(self) -> List[str]:
        """사용 가능한 프리셋 목록"""
        presets = []
        if os.path.exists(self.config.presets_dir):
            for file in os.listdir(self.config.presets_dir):
                if file.endswith('.json'):
                    presets.append(file[:-5])
        return sorted(presets)
    
    def export_config(self, export_path: str):
        """설정 내보내기"""
        config_dict = asdict(self.config)
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def import_config(self, import_path: str):
        """설정 가져오기"""
        if os.path.exists(import_path):
            try:
                with open(import_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 새 설정으로 덮어쓰기
                self.config = AppConfig(**data)
                self.save_config()
                self._ensure_directories()
                return True
            except:
                pass
        return False

# 전역 설정 매니저 인스턴스
config_manager = ConfigManager()