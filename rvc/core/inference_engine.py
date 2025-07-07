import os
import sys
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import traceback

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.infer.infer import VoiceConverter
from rvc.core.config_manager import InferenceConfig, config_manager
from rvc.utils.cache_manager import cache_manager
from rvc.services.inference_service import InferenceService, InferenceRequest

logger = logging.getLogger(__name__)

class InferenceEngine:
    """개선된 RVC 추론 엔진"""
    
    def __init__(self):
        self.voice_converter: Optional[VoiceConverter] = None
        self.inference_service: Optional[InferenceService] = None
        self.config = config_manager.get_inference_config()
        self._initialize()
        
    def _initialize(self):
        """엔진 초기화"""
        try:
            # Voice Converter 초기화
            self.voice_converter = VoiceConverter()
            
            # 비동기 추론 서비스 초기화
            if config_manager.config.max_workers > 0:
                self.inference_service = InferenceService(
                    max_workers=config_manager.config.max_workers
                )
                self.inference_service.start()
                
            logger.info("추론 엔진 초기화 완료")
        except Exception as e:
            logger.error(f"추론 엔진 초기화 실패: {e}")
            raise
    
    def convert_audio(
        self,
        input_path: str,
        output_path: str,
        model_path: str,
        index_path: str = "",
        config: Optional[InferenceConfig] = None,
        **kwargs
    ) -> Tuple[str, str]:
        """단일 오디오 변환"""
        
        # 설정 병합
        if config is None:
            config = self.config
        
        # kwargs로 설정 오버라이드
        config_dict = config.__dict__.copy()
        config_dict.update(kwargs)
        
        try:
            # 모델 정보 캐싱 확인
            model_info = cache_manager.cache_model_info(model_path)
            if model_info is None:
                # 캐시 미스 - 실제 변환 수행
                start_time = time.time()
                
                if self.voice_converter:
                    self.voice_converter.convert_audio(
                        audio_input_path=input_path,
                        audio_output_path=output_path,
                        model_path=model_path,
                        index_path=index_path,
                        **config_dict
                    )
                
                # 성능 정보 캐싱
                model_info = {
                    'model_path': model_path,
                    'processing_time': time.time() - start_time
                }
                cache_manager.save_model_info(model_path, model_info)
            else:
                # 캐시 히트 - 빠른 처리
                self.voice_converter.convert_audio(
                    audio_input_path=input_path,
                    audio_output_path=output_path,
                    model_path=model_path,
                    index_path=index_path,
                    **config_dict
                )
            
            # 출력 형식에 따른 경로 조정
            if config_dict.get('export_format', 'WAV').upper() != 'WAV':
                output_path = output_path.replace('.wav', 
                    f".{config_dict['export_format'].lower()}")
            
            return (f"파일 {input_path} 변환 성공", output_path)
            
        except Exception as e:
            logger.error(f"오디오 변환 실패: {e}")
            return (f"변환 실패: {str(e)}", "")
    
    def convert_audio_batch(
        self,
        input_folder: str,
        output_folder: str,
        model_path: str,
        index_path: str = "",
        config: Optional[InferenceConfig] = None,
        parallel: bool = True,
        **kwargs
    ) -> str:
        """배치 오디오 변환"""
        
        if config is None:
            config = self.config
            
        config_dict = config.__dict__.copy()
        config_dict.update(kwargs)
        
        try:
            # 오디오 파일 리스트 캐싱 사용
            audio_extensions = (
                '.wav', '.mp3', '.flac', '.ogg', '.opus', 
                '.m4a', '.mp4', '.aac', '.alac', '.wma', 
                '.aiff', '.webm', '.ac3'
            )
            
            audio_files = cache_manager.cache_file_list(
                input_folder, audio_extensions
            )
            
            if not audio_files:
                return f"폴더 {input_folder}에 오디오 파일이 없습니다."
            
            # 출력 디렉토리 생성
            os.makedirs(output_folder, exist_ok=True)
            
            if parallel and self.inference_service:
                # 병렬 처리
                return self._parallel_batch_convert(
                    audio_files, output_folder, model_path, 
                    index_path, config_dict
                )
            else:
                # 순차 처리
                return self._sequential_batch_convert(
                    audio_files, output_folder, model_path,
                    index_path, config_dict
                )
                
        except Exception as e:
            logger.error(f"배치 변환 실패: {e}")
            return f"배치 변환 실패: {str(e)}"
    
    def _parallel_batch_convert(
        self,
        audio_files: List[str],
        output_folder: str,
        model_path: str,
        index_path: str,
        config_dict: Dict
    ) -> str:
        """병렬 배치 변환"""
        import uuid
        
        request_ids = []
        
        for audio_file in audio_files:
            output_name = os.path.splitext(os.path.basename(audio_file))[0] + "_output.wav"
            output_path = os.path.join(output_folder, output_name)
            
            if os.path.exists(output_path):
                continue
                
            request = InferenceRequest(
                request_id=str(uuid.uuid4()),
                audio_path=audio_file,
                model_name=model_path,
                params={
                    'output_path': output_path,
                    'index_path': index_path,
                    **config_dict
                }
            )
            
            request_id = self.inference_service.submit_request(request)
            request_ids.append(request_id)
        
        # 결과 대기
        success_count = 0
        for request_id in request_ids:
            result = self.inference_service.get_result(request_id, timeout=300)
            if result and result.get('success'):
                success_count += 1
        
        return f"배치 변환 완료: {success_count}/{len(audio_files)} 파일 성공"
    
    def _sequential_batch_convert(
        self,
        audio_files: List[str],
        output_folder: str,
        model_path: str,
        index_path: str,
        config_dict: Dict
    ) -> str:
        """순차 배치 변환"""
        success_count = 0
        
        for i, audio_file in enumerate(audio_files):
            output_name = os.path.splitext(os.path.basename(audio_file))[0] + "_output.wav"
            output_path = os.path.join(output_folder, output_name)
            
            if os.path.exists(output_path):
                continue
            
            try:
                self.voice_converter.convert_audio(
                    audio_input_path=audio_file,
                    audio_output_path=output_path,
                    model_path=model_path,
                    index_path=index_path,
                    **config_dict
                )
                success_count += 1
                logger.info(f"변환 진행: {i+1}/{len(audio_files)}")
            except Exception as e:
                logger.error(f"파일 {audio_file} 변환 실패: {e}")
        
        return f"배치 변환 완료: {success_count}/{len(audio_files)} 파일 성공"
    
    def convert_with_tts(
        self,
        text: str,
        tts_voice: str,
        tts_rate: int,
        output_tts_path: str,
        output_rvc_path: str,
        model_path: str,
        index_path: str = "",
        config: Optional[InferenceConfig] = None,
        **kwargs
    ) -> Tuple[str, str]:
        """TTS + RVC 변환"""
        
        if config is None:
            config = self.config
            
        config_dict = config.__dict__.copy()
        config_dict.update(kwargs)
        
        try:
            # TTS 생성 (기존 코드 활용)
            import subprocess
            tts_script_path = os.path.join("rvc", "lib", "tools", "tts.py")
            
            command_tts = [
                sys.executable,
                tts_script_path,
                "",  # tts_file
                text,
                tts_voice,
                str(tts_rate),
                output_tts_path,
            ]
            
            subprocess.run(command_tts, check=True)
            
            # RVC 변환
            return self.convert_audio(
                input_path=output_tts_path,
                output_path=output_rvc_path,
                model_path=model_path,
                index_path=index_path,
                config=config,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"TTS + RVC 변환 실패: {e}")
            return (f"변환 실패: {str(e)}", "")
    
    def cleanup(self):
        """리소스 정리"""
        if self.inference_service:
            self.inference_service.stop()
        
        if self.voice_converter:
            self.voice_converter.cleanup_model()
        
        # 캐시 정리
        cache_manager.invalidate_file_cache()
        
        logger.info("추론 엔진 정리 완료")

# 싱글톤 인스턴스
_inference_engine = None

def get_inference_engine() -> InferenceEngine:
    """추론 엔진 싱글톤 인스턴스 반환"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
    return _inference_engine