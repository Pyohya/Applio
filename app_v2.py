import gradio as gr
import sys
import os
import logging
import argparse
import signal
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 현재 디렉토리를 경로에 추가
now_dir = os.getcwd()
sys.path.append(now_dir)

# Zluda hijack
import rvc.lib.zluda

# 기본 탭들 임포트
from tabs.train.train import train_tab
from tabs.extra.extra import extra_tab
from tabs.report.report import report_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.plugins.plugins import plugins_tab
from tabs.settings.settings import settings_tab

# 개선된 추론 탭 임포트
from rvc.ui.inference_tab_v2 import create_inference_tab_v2

# 개선된 core 모듈 사용
from core_v2 import run_prerequisites_script
from rvc.core.config_manager import config_manager

# i18n 초기화
from assets.i18n.i18n import I18nAuto
i18n = I18nAuto()

# Discord 프레즌스
from tabs.settings.sections.presence import load_config_presence
if load_config_presence():
    try:
        from assets.discord_presence import RPCManager
        RPCManager.start_presence()
    except:
        logging.warning("Discord presence 시작 실패")

# 설치 확인
try:
    import assets.installation_checker as installation_checker
    installation_checker.check_installation()
except Exception as e:
    logging.error(f"설치 확인 실패: {e}")

# Flask 서버 (선택사항)
from assets.flask.server import start_flask, load_config_flask
if load_config_flask():
    logging.info("Flask 서버 시작")
    start_flask()

# 테마 로드
import assets.themes.loadThemes as loadThemes
my_theme = loadThemes.load_theme() or "ParityError/Interstellar"

# Gradio 인터페이스 정의
class ApplioApp:
    def __init__(self):
        self.app = None
        self.config = config_manager.config
        
    def create_interface(self):
        """Gradio 인터페이스 생성"""
        with gr.Blocks(
            theme=my_theme,
            title="Applio - 개선된 버전",
            css=self._get_custom_css()
        ) as self.app:
            # 헤더
            gr.Markdown(
                """
                # 🎤 Applio - 고품질 음성 변환 도구
                ### 개선된 성능과 사용자 경험을 제공합니다
                """
            )
            
            # 빠른 시작 가이드
            if config_manager.config.language == "ko_KR":
                with gr.Accordion("🚀 빠른 시작 가이드", open=False):
                    gr.Markdown(
                        """
                        1. **추론** 탭에서 음성 변환을 시작하세요
                        2. 모델과 오디오를 선택하고 변환 버튼을 클릭합니다
                        3. 고급 설정은 필요에 따라 조정하세요
                        
                        💡 **팁**: 프리셋을 사용하면 빠르게 설정할 수 있습니다!
                        """
                    )
            
            # 상태 표시
            with gr.Row():
                self.status_text = gr.Markdown(
                    value="✅ 시스템 준비 완료",
                    elem_id="system_status"
                )
                self.gpu_info = gr.Markdown(
                    value=self._get_gpu_info(),
                    elem_id="gpu_info"
                )
            
            # 메인 탭들
            with gr.Tabs() as tabs:
                # 개선된 추론 탭 사용
                with gr.Tab(i18n("Inference")):
                    create_inference_tab_v2()
                
                with gr.Tab(i18n("Training")):
                    train_tab()
                
                with gr.Tab(i18n("TTS")):
                    tts_tab()
                
                with gr.Tab(i18n("Voice Blender")):
                    voice_blender_tab()
                
                with gr.Tab(i18n("Plugins")):
                    plugins_tab()
                
                with gr.Tab(i18n("Download")):
                    download_tab()
                
                with gr.Tab(i18n("Report a Bug")):
                    report_tab()
                
                with gr.Tab(i18n("Extra")):
                    extra_tab()
                
                with gr.Tab(i18n("Settings")):
                    settings_tab()
            
            # 푸터
            gr.Markdown(
                """
                ---
                <div style='text-align: center'>
                    <p>🌟 Applio v2 - 성능과 편의성이 개선된 버전</p>
                    <p>
                        <a href='https://discord.gg/urxFjYmYYh'>Discord</a> | 
                        <a href='https://github.com/IAHispano/Applio'>GitHub</a>
                    </p>
                </div>
                """,
                elem_id="footer"
            )
        
        return self.app
    
    def _get_custom_css(self):
        """커스텀 CSS"""
        return """
        /* 개선된 UI 스타일 */
        .gradio-container {
            max-width: 1400px !important;
        }
        
        /* 상태 표시 개선 */
        #system_status {
            background-color: #f0f9ff;
            padding: 10px;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
        }
        
        #gpu_info {
            background-color: #fef3c7;
            padding: 10px;
            border-radius: 8px;
            border-left: 4px solid #f59e0b;
        }
        
        /* 버튼 스타일 개선 */
        .primary {
            background-color: #3b82f6 !important;
            transition: all 0.3s ease;
        }
        
        .primary:hover {
            background-color: #2563eb !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
        
        /* 애니메이션 추가 */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .gradio-container > * {
            animation: fadeIn 0.5s ease-out;
        }
        
        /* 탭 스타일 개선 */
        .tabs {
            margin-top: 20px;
        }
        
        .tab-nav button {
            transition: all 0.3s ease;
        }
        
        .tab-nav button:hover {
            background-color: #e5e7eb;
        }
        
        .tab-nav button.selected {
            background-color: #3b82f6 !important;
            color: white !important;
        }
        
        /* 반응형 개선 */
        @media (max-width: 768px) {
            .gradio-container {
                padding: 10px !important;
            }
        }
        
        /* 다크 모드 지원 */
        @media (prefers-color-scheme: dark) {
            #system_status {
                background-color: #1e3a5f;
                color: #e0e7ff;
            }
            
            #gpu_info {
                background-color: #451a03;
                color: #fef3c7;
            }
        }
        
        /* 툴팁 스타일 */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #1f2937;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            z-index: 1000;
        }
        
        /* 로딩 스피너 개선 */
        .pending {
            background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
        }
        
        @keyframes loading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        
        /* 스크롤바 스타일 */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        """
    
    def _get_gpu_info(self):
        """GPU 정보 가져오기"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return f"🖥️ GPU: {gpu_name} ({gpu_memory:.1f}GB)"
            else:
                return "💻 CPU 모드로 실행 중"
        except:
            return "❓ GPU 정보를 가져올 수 없습니다"
    
    def launch(self):
        """앱 실행"""
        if self.app is None:
            self.app = self.create_interface()
        
        # 포트 설정
        port = self._get_port_from_args()
        
        # 실행 옵션
        launch_kwargs = {
            "favicon_path": "assets/ICON.ico",
            "share": "--share" in sys.argv,
            "inbrowser": "--open" in sys.argv,
            "server_port": port,
            "server_name": "0.0.0.0" if "--public" in sys.argv else "127.0.0.1",
        }
        
        # 최대 시도 횟수
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                logging.info(f"포트 {port}에서 Applio 시작 중...")
                self.app.launch(**launch_kwargs)
                break
            except OSError:
                logging.warning(f"포트 {port} 사용 불가, {port - 1} 시도 중...")
                port -= 1
                launch_kwargs["server_port"] = port
            except Exception as error:
                logging.error(f"Gradio 실행 오류: {error}")
                break
    
    def _get_port_from_args(self):
        """명령줄 인자에서 포트 가져오기"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=config_manager.config.port)
        parser.add_argument("--share", action="store_true")
        parser.add_argument("--open", action="store_true")
        parser.add_argument("--public", action="store_true")
        
        args, _ = parser.parse_known_args()
        return args.port

def signal_handler(sig, frame):
    """종료 시그널 처리"""
    logging.info("종료 신호 받음, 정리 중...")
    
    # 추론 엔진 정리
    try:
        from rvc.core.inference_engine import get_inference_engine
        engine = get_inference_engine()
        engine.cleanup()
    except:
        pass
    
    # 설정 저장
    try:
        config_manager.save_config()
    except:
        pass
    
    logging.info("종료 완료")
    sys.exit(0)

if __name__ == "__main__":
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Prerequisites 다운로드
    logging.info("필수 구성 요소 확인 중...")
    run_prerequisites_script(
        pretraineds_v1_f0=False,
        pretraineds_v1_nof0=False,
        pretraineds_v2_f0=True,
        pretraineds_v2_nof0=False,
        models=True,
        exe=True,
    )
    
    # 앱 시작
    app = ApplioApp()
    app.launch()