import gradio as gr
import os
import json
from typing import Dict, List, Optional, Tuple
import logging

from rvc.core.inference_engine import get_inference_engine
from rvc.core.config_manager import config_manager, InferenceConfig
from rvc.utils.cache_manager import cache_manager
from assets.i18n.i18n import I18nAuto

logger = logging.getLogger(__name__)
i18n = I18nAuto()

class InferenceTabV2:
    """개선된 추론 탭 UI"""
    
    def __init__(self):
        self.inference_engine = get_inference_engine()
        self.current_preset = None
        
    def create_ui(self) -> gr.Column:
        """UI 생성"""
        with gr.Column() as tab:
            # 헤더
            gr.Markdown("## 🎤 음성 변환 (개선된 버전)")
            gr.Markdown("더 빠르고 직관적인 음성 변환을 경험하세요!")
            
            with gr.Row():
                # 왼쪽 패널 - 모델 선택
                with gr.Column(scale=1):
                    self._create_model_panel()
                
                # 오른쪽 패널 - 설정
                with gr.Column(scale=2):
                    self._create_settings_panel()
            
            # 중앙 - 오디오 입력/출력
            self._create_audio_panel()
            
            # 하단 - 실행 버튼
            self._create_action_panel()
            
        return tab
    
    def _create_model_panel(self):
        """모델 선택 패널"""
        gr.Markdown("### 📦 모델 선택")
        
        # 모델 파일 선택
        self.model_dropdown = gr.Dropdown(
            label=i18n("Voice Model"),
            info=i18n("음성 모델을 선택하세요"),
            choices=self._get_cached_models(),
            interactive=True,
            elem_id="model_dropdown"
        )
        
        # 인덱스 파일 선택
        self.index_dropdown = gr.Dropdown(
            label=i18n("Index File"),
            info=i18n("인덱스 파일 (선택사항)"),
            choices=self._get_cached_indexes(),
            interactive=True,
            elem_id="index_dropdown"
        )
        
        # 새로고침 버튼
        refresh_btn = gr.Button(
            "🔄 새로고침",
            size="sm",
            elem_id="refresh_btn"
        )
        
        # 모델 정보 표시
        self.model_info = gr.Markdown(
            value="모델을 선택하면 정보가 표시됩니다.",
            elem_id="model_info"
        )
        
        # 이벤트 연결
        refresh_btn.click(
            fn=self._refresh_models,
            outputs=[self.model_dropdown, self.index_dropdown]
        )
        
        self.model_dropdown.change(
            fn=self._on_model_change,
            inputs=[self.model_dropdown],
            outputs=[self.index_dropdown, self.model_info]
        )
    
    def _create_settings_panel(self):
        """설정 패널"""
        gr.Markdown("### ⚙️ 변환 설정")
        
        # 프리셋 선택
        with gr.Row():
            self.preset_dropdown = gr.Dropdown(
                label="프리셋",
                choices=config_manager.list_presets(),
                value="기본",
                interactive=True,
                elem_id="preset_dropdown"
            )
            
            save_preset_btn = gr.Button("💾", size="sm")
            delete_preset_btn = gr.Button("🗑️", size="sm")
        
        # 기본 설정
        with gr.Accordion("기본 설정", open=True):
            with gr.Row():
                self.pitch_slider = gr.Slider(
                    label=i18n("Pitch"),
                    info=i18n("음높이 조정 (-12 ~ +12)"),
                    minimum=-12,
                    maximum=12,
                    step=1,
                    value=0,
                    interactive=True
                )
                
                self.index_rate_slider = gr.Slider(
                    label=i18n("Index Rate"),
                    info=i18n("인덱스 영향도"),
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.75,
                    interactive=True
                )
            
            with gr.Row():
                self.filter_radius_slider = gr.Slider(
                    label=i18n("Filter Radius"),
                    minimum=0,
                    maximum=7,
                    step=1,
                    value=3,
                    interactive=True
                )
                
                self.protect_slider = gr.Slider(
                    label=i18n("Protect"),
                    info=i18n("자음 보호"),
                    minimum=0,
                    maximum=0.5,
                    step=0.01,
                    value=0.33,
                    interactive=True
                )
        
        # 고급 설정
        with gr.Accordion("고급 설정", open=False):
            self.f0_method = gr.Radio(
                label=i18n("F0 Method"),
                choices=["rmvpe", "pm", "harvest", "crepe"],
                value="rmvpe",
                interactive=True
            )
            
            with gr.Row():
                self.split_audio = gr.Checkbox(
                    label=i18n("Split Audio"),
                    value=False,
                    interactive=True
                )
                
                self.autotune = gr.Checkbox(
                    label=i18n("Autotune"),
                    value=False,
                    interactive=True
                )
                
                self.clean_audio = gr.Checkbox(
                    label=i18n("Clean Audio"),
                    value=False,
                    interactive=True
                )
        
        # 후처리 효과
        with gr.Accordion("후처리 효과", open=False):
            self.post_process = gr.Checkbox(
                label="후처리 활성화",
                value=False,
                interactive=True
            )
            
            with gr.Column(visible=False) as self.effects_panel:
                with gr.Row():
                    self.reverb = gr.Checkbox(label="리버브", value=False)
                    self.chorus = gr.Checkbox(label="코러스", value=False)
                    self.delay = gr.Checkbox(label="딜레이", value=False)
                
                self.reverb_settings = gr.Column(visible=False)
                with self.reverb_settings:
                    self.reverb_room_size = gr.Slider(
                        label="룸 크기",
                        minimum=0,
                        maximum=1,
                        value=0.5
                    )
                    self.reverb_damping = gr.Slider(
                        label="댐핑",
                        minimum=0,
                        maximum=1,
                        value=0.5
                    )
        
        # 이벤트 연결
        self.preset_dropdown.change(
            fn=self._load_preset,
            inputs=[self.preset_dropdown],
            outputs=self._get_all_setting_components()
        )
        
        self.post_process.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[self.post_process],
            outputs=[self.effects_panel]
        )
        
        self.reverb.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[self.reverb],
            outputs=[self.reverb_settings]
        )
    
    def _create_audio_panel(self):
        """오디오 입력/출력 패널"""
        gr.Markdown("### 🎵 오디오")
        
        with gr.Row():
            # 입력
            with gr.Column():
                gr.Markdown("**입력 오디오**")
                
                self.audio_input = gr.Audio(
                    label="오디오 업로드",
                    type="filepath",
                    elem_id="audio_input"
                )
                
                # 빠른 선택
                self.quick_select = gr.Dropdown(
                    label="빠른 선택",
                    choices=self._get_sample_audios(),
                    interactive=True
                )
                
                self.record_audio = gr.Audio(
                    label="녹음하기",
                    type="filepath",
                    source="microphone",
                    visible=False
                )
                
                self.toggle_record = gr.Button(
                    "🎙️ 녹음 모드",
                    size="sm"
                )
            
            # 출력
            with gr.Column():
                gr.Markdown("**출력 오디오**")
                
                self.audio_output = gr.Audio(
                    label="변환 결과",
                    type="filepath",
                    interactive=False,
                    elem_id="audio_output"
                )
                
                self.output_format = gr.Radio(
                    label="출력 형식",
                    choices=["WAV", "MP3", "FLAC"],
                    value="WAV",
                    interactive=True
                )
                
                # 진행 상태
                self.progress_bar = gr.Progress(visible=False)
                self.status_text = gr.Markdown(
                    value="대기 중...",
                    elem_id="status_text"
                )
        
        # 이벤트 연결
        self.quick_select.change(
            fn=lambda x: x,
            inputs=[self.quick_select],
            outputs=[self.audio_input]
        )
        
        self.toggle_record.click(
            fn=self._toggle_record_mode,
            outputs=[self.record_audio, self.audio_input]
        )
    
    def _create_action_panel(self):
        """실행 버튼 패널"""
        with gr.Row():
            self.convert_btn = gr.Button(
                "🚀 변환 시작",
                variant="primary",
                size="lg",
                elem_id="convert_btn"
            )
            
            self.stop_btn = gr.Button(
                "⏹️ 중지",
                variant="stop",
                size="lg",
                visible=False,
                elem_id="stop_btn"
            )
            
            self.batch_btn = gr.Button(
                "📁 배치 변환",
                variant="secondary",
                size="lg",
                elem_id="batch_btn"
            )
        
        # 배치 변환 패널
        with gr.Row(visible=False) as self.batch_panel:
            self.input_folder = gr.Textbox(
                label="입력 폴더",
                placeholder="오디오 파일이 있는 폴더 경로"
            )
            self.output_folder = gr.Textbox(
                label="출력 폴더",
                placeholder="변환된 파일을 저장할 폴더"
            )
            self.batch_start_btn = gr.Button("배치 시작")
        
        # 이벤트 연결
        self.convert_btn.click(
            fn=self._convert_audio,
            inputs=self._get_conversion_inputs(),
            outputs=[self.audio_output, self.status_text]
        )
        
        self.batch_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[self.batch_panel]
        )
        
        self.batch_start_btn.click(
            fn=self._batch_convert,
            inputs=[
                self.input_folder,
                self.output_folder,
                self.model_dropdown,
                self.index_dropdown
            ],
            outputs=[self.status_text]
        )
    
    # 헬퍼 메서드들
    def _get_cached_models(self) -> List[str]:
        """캐시된 모델 목록 반환"""
        model_extensions = ('.pth', '.onnx')
        models = cache_manager.cache_file_list(
            config_manager.config.models_dir,
            model_extensions
        )
        # G_, D_ 제외
        return [m for m in models if not any(
            os.path.basename(m).startswith(prefix) 
            for prefix in ['G_', 'D_']
        )]
    
    def _get_cached_indexes(self) -> List[str]:
        """캐시된 인덱스 목록 반환"""
        return cache_manager.cache_file_list(
            config_manager.config.models_dir,
            ('.index',)
        )
    
    def _get_sample_audios(self) -> List[str]:
        """샘플 오디오 목록 반환"""
        audio_extensions = (
            '.wav', '.mp3', '.flac', '.ogg', '.opus',
            '.m4a', '.mp4', '.aac', '.alac', '.wma'
        )
        return cache_manager.cache_file_list(
            config_manager.config.audio_dir,
            audio_extensions
        )
    
    def _refresh_models(self) -> Tuple[gr.Dropdown, gr.Dropdown]:
        """모델 목록 새로고침"""
        cache_manager.invalidate_file_cache()
        return (
            gr.update(choices=self._get_cached_models()),
            gr.update(choices=self._get_cached_indexes())
        )
    
    def _on_model_change(self, model_path: str) -> Tuple[gr.Dropdown, str]:
        """모델 변경 이벤트"""
        if not model_path:
            return gr.update(), "모델을 선택하세요."
        
        # 매칭되는 인덱스 찾기
        model_name = os.path.basename(model_path).split('.')[0]
        indexes = self._get_cached_indexes()
        matched_index = None
        
        for idx in indexes:
            if model_name in os.path.basename(idx):
                matched_index = idx
                break
        
        # 모델 정보 표시
        info = f"**선택된 모델:** {os.path.basename(model_path)}\n"
        model_info = cache_manager.cache_model_info(model_path)
        if model_info:
            info += f"**처리 시간:** {model_info.get('processing_time', 0):.2f}초"
        
        return (
            gr.update(value=matched_index),
            info
        )
    
    def _toggle_record_mode(self) -> Tuple[gr.Audio, gr.Audio]:
        """녹음 모드 토글"""
        # 간단한 토글 구현
        return (
            gr.update(visible=True),
            gr.update(visible=False)
        )
    
    def _get_all_setting_components(self) -> List:
        """모든 설정 컴포넌트 반환"""
        return [
            self.pitch_slider,
            self.index_rate_slider,
            self.filter_radius_slider,
            self.protect_slider,
            self.f0_method,
            self.split_audio,
            self.autotune,
            self.clean_audio
        ]
    
    def _get_conversion_inputs(self) -> List:
        """변환에 필요한 입력 컴포넌트 반환"""
        return [
            self.audio_input,
            self.model_dropdown,
            self.index_dropdown,
            self.pitch_slider,
            self.index_rate_slider,
            self.filter_radius_slider,
            self.protect_slider,
            self.f0_method,
            self.split_audio,
            self.autotune,
            self.clean_audio,
            self.output_format
        ]
    
    def _load_preset(self, preset_name: str) -> List:
        """프리셋 로드"""
        if not preset_name:
            return [gr.update()] * len(self._get_all_setting_components())
        
        preset_data = config_manager.load_preset(preset_name)
        if not preset_data:
            return [gr.update()] * len(self._get_all_setting_components())
        
        return [
            preset_data.get('pitch', 0),
            preset_data.get('index_rate', 0.75),
            preset_data.get('filter_radius', 3),
            preset_data.get('protect', 0.33),
            preset_data.get('f0_method', 'rmvpe'),
            preset_data.get('split_audio', False),
            preset_data.get('autotune', False),
            preset_data.get('clean_audio', False)
        ]
    
    def _convert_audio(
        self,
        audio_input: str,
        model_path: str,
        index_path: str,
        pitch: int,
        index_rate: float,
        filter_radius: int,
        protect: float,
        f0_method: str,
        split_audio: bool,
        autotune: bool,
        clean_audio: bool,
        output_format: str
    ) -> Tuple[Optional[str], str]:
        """오디오 변환 실행"""
        
        if not audio_input or not model_path:
            return None, "⚠️ 오디오와 모델을 선택해주세요."
        
        try:
            # 출력 경로 생성
            output_name = os.path.splitext(os.path.basename(audio_input))[0]
            output_path = os.path.join(
                config_manager.config.audio_dir,
                f"{output_name}_converted.wav"
            )
            
            # 설정 생성
            config = InferenceConfig(
                pitch=pitch,
                index_rate=index_rate,
                filter_radius=filter_radius,
                protect=protect,
                f0_method=f0_method,
                split_audio=split_audio,
                f0_autotune=autotune,
                clean_audio=clean_audio,
                export_format=output_format
            )
            
            # 변환 실행
            message, output_file = self.inference_engine.convert_audio(
                input_path=audio_input,
                output_path=output_path,
                model_path=model_path,
                index_path=index_path or "",
                config=config
            )
            
            return output_file, f"✅ {message}"
            
        except Exception as e:
            logger.error(f"변환 실패: {e}")
            return None, f"❌ 변환 실패: {str(e)}"
    
    def _batch_convert(
        self,
        input_folder: str,
        output_folder: str,
        model_path: str,
        index_path: str
    ) -> str:
        """배치 변환 실행"""
        
        if not all([input_folder, output_folder, model_path]):
            return "⚠️ 모든 필드를 입력해주세요."
        
        try:
            result = self.inference_engine.convert_audio_batch(
                input_folder=input_folder,
                output_folder=output_folder,
                model_path=model_path,
                index_path=index_path or "",
                parallel=True
            )
            
            return f"✅ {result}"
            
        except Exception as e:
            logger.error(f"배치 변환 실패: {e}")
            return f"❌ 배치 변환 실패: {str(e)}"

def create_inference_tab_v2() -> gr.Column:
    """개선된 추론 탭 생성"""
    tab = InferenceTabV2()
    return tab.create_ui()