#!/bin/bash

# Applio v2 - 개선된 버전 실행 스크립트

echo "🚀 Applio v2 시작 중..."
echo "개선된 성능과 UI를 경험해보세요!"

# 현재 디렉토리 저장
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Python 확인
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3가 설치되어 있지 않습니다."
    exit 1
fi

# 개선된 버전 실행
echo "✨ 개선된 Applio를 시작합니다..."
python3 app_v2.py "$@"