"""
개선된 core 모듈 - 기존 API와의 호환성 유지
"""

import os
import sys
import json
from functools import lru_cache
from typing import Optional

now_dir = os.getcwd()
sys.path.append(now_dir)

# 새로운 모듈들 임포트
from rvc.core.inference_engine import get_inference_engine
from rvc.core.config_manager import config_manager, InferenceConfig
from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline
from rvc.train.process.model_blender import model_blender
from rvc.train.process.model_information import model_information
from rvc.train.process.extract_small_model import extract_small_model
from rvc.lib.tools.analyzer import analyze_audio
from rvc.lib.tools.launch_tensorboard import launch_tensorboard_pipeline
from rvc.lib.tools.model_download import model_download_pipeline

python = sys.executable

# 기존 API와의 호환성을 위한 전역 변수
@lru_cache(maxsize=1)
def load_voices_data():
    with open(
        os.path.join("rvc", "lib", "tools", "tts_voices.json"), "r", encoding="utf-8"
    ) as file:
        return json.load(file)

voices_data = load_voices_data()
locales = list({voice["ShortName"] for voice in voices_data})

# 추론 엔진 싱글톤
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        _engine = get_inference_engine()
    return _engine

# 기존 API 호환 함수들
def run_infer_script(
    pitch: int,
    filter_radius: int,
    index_rate: float,
    volume_envelope: int,
    protect: float,
    hop_length: int,
    f0_method: str,
    input_path: str,
    output_path: str,
    pth_path: str,
    index_path: str,
    split_audio: bool,
    f0_autotune: bool,
    f0_autotune_strength: float,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    f0_file: str,
    embedder_model: str,
    embedder_model_custom: str = None,
    formant_shifting: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
    post_process: bool = False,
    reverb: bool = False,
    pitch_shift: bool = False,
    limiter: bool = False,
    gain: bool = False,
    distortion: bool = False,
    chorus: bool = False,
    bitcrush: bool = False,
    clipping: bool = False,
    compressor: bool = False,
    delay: bool = False,
    reverb_room_size: float = 0.5,
    reverb_damping: float = 0.5,
    reverb_wet_gain: float = 0.5,
    reverb_dry_gain: float = 0.5,
    reverb_width: float = 0.5,
    reverb_freeze_mode: float = 0.5,
    pitch_shift_semitones: float = 0.0,
    limiter_threshold: float = -6,
    limiter_release_time: float = 0.01,
    gain_db: float = 0.0,
    distortion_gain: float = 25,
    chorus_rate: float = 1.0,
    chorus_depth: float = 0.25,
    chorus_center_delay: float = 7,
    chorus_feedback: float = 0.0,
    chorus_mix: float = 0.5,
    bitcrush_bit_depth: int = 8,
    clipping_threshold: float = -6,
    compressor_threshold: float = 0,
    compressor_ratio: float = 1,
    compressor_attack: float = 1.0,
    compressor_release: float = 100,
    delay_seconds: float = 0.5,
    delay_feedback: float = 0.0,
    delay_mix: float = 0.5,
    sid: int = 0,
):
    """기존 API와 호환되는 추론 함수"""
    
    # InferenceConfig 생성
    config = InferenceConfig(
        pitch=pitch,
        filter_radius=filter_radius,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect,
        hop_length=hop_length,
        f0_method=f0_method,
        split_audio=split_audio,
        f0_autotune=f0_autotune,
        f0_autotune_strength=f0_autotune_strength,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        export_format=export_format,
        embedder_model=embedder_model,
        embedder_model_custom=embedder_model_custom,
        sid=sid,
        post_process=post_process,
        formant_shifting=formant_shifting,
        formant_qfrency=formant_qfrency,
        formant_timbre=formant_timbre,
        reverb=reverb,
        reverb_room_size=reverb_room_size,
        reverb_damping=reverb_damping,
        reverb_wet_gain=reverb_wet_gain,
        reverb_dry_gain=reverb_dry_gain,
        reverb_width=reverb_width,
        reverb_freeze_mode=reverb_freeze_mode,
        pitch_shift=pitch_shift,
        pitch_shift_semitones=pitch_shift_semitones,
        limiter=limiter,
        limiter_threshold=limiter_threshold,
        limiter_release_time=limiter_release_time,
        gain=gain,
        gain_db=gain_db,
        distortion=distortion,
        distortion_gain=distortion_gain,
        chorus=chorus,
        chorus_rate=chorus_rate,
        chorus_depth=chorus_depth,
        chorus_center_delay=chorus_center_delay,
        chorus_feedback=chorus_feedback,
        chorus_mix=chorus_mix,
        bitcrush=bitcrush,
        bitcrush_bit_depth=bitcrush_bit_depth,
        clipping=clipping,
        clipping_threshold=clipping_threshold,
        compressor=compressor,
        compressor_threshold=compressor_threshold,
        compressor_ratio=compressor_ratio,
        compressor_attack=compressor_attack,
        compressor_release=compressor_release,
        delay=delay,
        delay_seconds=delay_seconds,
        delay_feedback=delay_feedback,
        delay_mix=delay_mix,
    )
    
    # 새로운 엔진으로 실행
    engine = _get_engine()
    return engine.convert_audio(
        input_path=input_path,
        output_path=output_path,
        model_path=pth_path,
        index_path=index_path,
        config=config,
        f0_file=f0_file
    )

def run_batch_infer_script(
    pitch: int,
    filter_radius: int,
    index_rate: float,
    volume_envelope: int,
    protect: float,
    hop_length: int,
    f0_method: str,
    input_folder: str,
    output_folder: str,
    pth_path: str,
    index_path: str,
    split_audio: bool,
    f0_autotune: bool,
    f0_autotune_strength: float,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    f0_file: str,
    embedder_model: str,
    embedder_model_custom: str = None,
    **kwargs  # 나머지 매개변수들
):
    """기존 API와 호환되는 배치 추론 함수"""
    
    config = InferenceConfig(
        pitch=pitch,
        filter_radius=filter_radius,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect,
        hop_length=hop_length,
        f0_method=f0_method,
        split_audio=split_audio,
        f0_autotune=f0_autotune,
        f0_autotune_strength=f0_autotune_strength,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        export_format=export_format,
        embedder_model=embedder_model,
        embedder_model_custom=embedder_model_custom,
        **kwargs
    )
    
    engine = _get_engine()
    result = engine.convert_audio_batch(
        input_folder=input_folder,
        output_folder=output_folder,
        model_path=pth_path,
        index_path=index_path,
        config=config,
        parallel=True
    )
    
    return result

def run_tts_script(
    tts_file: str,
    tts_text: str,
    tts_voice: str,
    tts_rate: int,
    pitch: int,
    filter_radius: int,
    index_rate: float,
    volume_envelope: int,
    protect: float,
    hop_length: int,
    f0_method: str,
    output_tts_path: str,
    output_rvc_path: str,
    pth_path: str,
    index_path: str,
    split_audio: bool,
    f0_autotune: bool,
    f0_autotune_strength: float,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    f0_file: str,
    embedder_model: str,
    embedder_model_custom: str = None,
    sid: int = 0,
):
    """기존 API와 호환되는 TTS + RVC 함수"""
    
    config = InferenceConfig(
        pitch=pitch,
        filter_radius=filter_radius,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect,
        hop_length=hop_length,
        f0_method=f0_method,
        split_audio=split_audio,
        f0_autotune=f0_autotune,
        f0_autotune_strength=f0_autotune_strength,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        export_format=export_format,
        embedder_model=embedder_model,
        embedder_model_custom=embedder_model_custom,
        sid=sid,
    )
    
    engine = _get_engine()
    return engine.convert_with_tts(
        text=tts_text or tts_file,  # tts_file은 레거시 지원
        tts_voice=tts_voice,
        tts_rate=tts_rate,
        output_tts_path=output_tts_path,
        output_rvc_path=output_rvc_path,
        model_path=pth_path,
        index_path=index_path,
        config=config,
        f0_file=f0_file
    )

# 나머지 기존 함수들은 원본 core.py의 구현을 그대로 사용
def run_preprocess_script(
    model_name: str,
    dataset_path: str,
    sample_rate: int,
    cpu_cores: int,
    cut_preprocess: bool,
    process_effects: bool,
    noise_reduction: bool,
    clean_strength: float,
):
    import subprocess
    from rvc.configs.config import Config
    
    config = Config()
    per = 3.0 if config.is_half else 3.7
    preprocess_script_path = os.path.join("rvc", "train", "preprocess", "preprocess.py")
    command = [
        python,
        preprocess_script_path,
        *map(
            str,
            [
                os.path.join("logs", model_name),
                dataset_path,
                sample_rate,
                per,
                cpu_cores,
                cut_preprocess,
                process_effects,
                noise_reduction,
                clean_strength,
            ],
        ),
    ]
    subprocess.run(command)
    return f"Model {model_name} preprocessed successfully."

def run_extract_script(
    model_name: str,
    rvc_version: str,
    f0_method: str,
    hop_length: int,
    cpu_cores: int,
    gpu: int,
    sample_rate: int,
    embedder_model: str,
    embedder_model_custom: str = None,
):
    import subprocess
    
    model_path = os.path.join("logs", model_name)
    extract = os.path.join("rvc", "train", "extract", "extract.py")

    command_1 = [
        python,
        extract,
        *map(
            str,
            [
                model_path,
                f0_method,
                hop_length,
                cpu_cores,
                gpu,
                rvc_version,
                sample_rate,
                embedder_model,
                embedder_model_custom,
            ],
        ),
    ]

    subprocess.run(command_1)
    return f"Model {model_name} extracted successfully."

def run_train_script(
    model_name: str,
    rvc_version: str,
    save_every_epoch: int,
    save_only_latest: bool,
    save_every_weights: bool,
    total_epoch: int,
    sample_rate: int,
    batch_size: int,
    gpu: int,
    pitch_guidance: bool,
    overtraining_detector: bool,
    overtraining_threshold: int,
    pretrained: bool,
    cleanup: bool,
    index_algorithm: str = "Auto",
    cache_data_in_gpu: bool = False,
    custom_pretrained: bool = False,
    g_pretrained_path: str = None,
    d_pretrained_path: str = None,
):
    import subprocess
    
    # 학습 설정 업데이트
    config_manager.update_training_config(
        batch_size=batch_size,
        total_epoch=total_epoch,
        save_every_epoch=save_every_epoch,
        save_only_latest=save_only_latest,
        save_every_weights=save_every_weights,
        sample_rate=sample_rate,
        pretrained=pretrained,
        cleanup=cleanup,
        cache_data_in_gpu=cache_data_in_gpu,
        custom_pretrained=custom_pretrained,
        g_pretrained_path=g_pretrained_path,
        d_pretrained_path=d_pretrained_path,
        overtraining_detector=overtraining_detector,
        overtraining_threshold=overtraining_threshold,
    )
    
    # 기존 스크립트 실행
    train_script_path = os.path.join("rvc", "train", "train.py")
    command = [
        python,
        train_script_path,
        *map(
            str,
            [
                os.path.join("logs", model_name),
                save_every_epoch,
                total_epoch,
                batch_size,
                gpu,
                pitch_guidance,
                pretrained,
                save_only_latest,
                save_every_weights,
                cache_data_in_gpu,
                overtraining_detector,
                overtraining_threshold,
                cleanup,
                rvc_version,
                sample_rate,
                custom_pretrained,
                g_pretrained_path,
                d_pretrained_path,
            ],
        ),
    ]
    
    subprocess.run(command)
    return f"Model {model_name} trained successfully."

# 나머지 함수들도 동일한 방식으로 래핑
def run_index_script(model_name: str, rvc_version: str, index_algorithm: str):
    import subprocess
    index_script_path = os.path.join("rvc", "train", "process", "extract_index.py")
    command = [
        python,
        index_script_path,
        os.path.join("logs", model_name),
        rvc_version,
        index_algorithm,
    ]
    subprocess.run(command)
    return f"Index file for {model_name} generated successfully."

def run_model_extract_script(
    pth_path: str,
    model_name: str,
    sample_rate: int,
    pitch_guidance: bool,
    rvc_version: str,
    epoch: int,
    step: int,
):
    return extract_small_model(
        pth_path, model_name, sample_rate, pitch_guidance, rvc_version, epoch, step
    )

def run_model_information_script(pth_path: str):
    return model_information(pth_path)

def run_model_blender_script(
    model_name: str, pth_path_1: str, pth_path_2: str, ratio: float
):
    return model_blender(model_name, pth_path_1, pth_path_2, ratio)

def run_tensorboard_script():
    launch_tensorboard_pipeline()

def run_download_script(model_link: str):
    return model_download_pipeline(model_link)

def run_prerequisites_script(
    pretraineds_v1_f0: bool,
    pretraineds_v1_nof0: bool,
    pretraineds_v2_f0: bool,
    pretraineds_v2_nof0: bool,
    models: bool,
    exe: bool,
):
    prequisites_download_pipeline(
        pretraineds_v1_f0,
        pretraineds_v1_nof0,
        pretraineds_v2_f0,
        pretraineds_v2_nof0,
        models,
        exe,
    )

def run_audio_analyzer_script(
    input_path: str, save_plot_path: str = "logs/audio_analysis.png"
):
    audio_info, plot_path = analyze_audio(input_path, save_plot_path)
    print(
        f"Audio info of {input_path}: {audio_info}\n"
        f"Audio analysis plot saved at: {plot_path}"
    )
    return audio_info, plot_path

# 나머지 함수들은 원본 core.py에서 가져옴
import_voice_converter = lambda: _get_engine().voice_converter
get_config = lambda: config_manager.config