# Jetson MLOps Lab

Jetson Orin Nano (JetPack 7.2) 한 대로 LLM MLOps 파이프라인의 핵심 구성요소를 직접 구현하며 학습하는 7일 스프린트 기록입니다.

## 왜 이 프로젝트인가

클라우드 멀티노드 클러스터 대신, 리소스가 제한된 엣지 디바이스 위에서 추론 서빙 · 모델 최적화 · 파인튜닝 · 모니터링의 원리를 바닥부터 구현했습니다. 제약이 있는 환경일수록 "왜 그렇게 동작하는지"를 더 깊이 이해해야 하기 때문입니다.

## 진행 현황

| Day | 주제 | 상태 | 노트북 | 산출물 |
|---|---|---|---|---|
| 0 | 환경 구축 | ✅ | - | venv, Jupyter, torch/transformers 설치 및 검증 |
| 1 | PyTorch → HuggingFace | ✅ | [day1_practice](notebooks/day1_practice.ipynb) | - |
| 2 | 추론의 원리 (KV Cache, TTFT/TPOT) | 📝 준비됨 | [day2_practice](notebooks/day2_practice.ipynb) | - |
| 3 | 추론 서버 (FastAPI + 배칭) | 📝 준비됨 | [day3_practice](notebooks/day3_practice.ipynb) | [serving/server.py](serving/server.py) |
| 4 | 모델 양자화 (INT8/INT4) | 📝 준비됨 | [day4_practice](notebooks/day4_practice.ipynb) | - |
| 5 | LoRA 파인튜닝 (PEFT) | 📝 준비됨 | [day5_practice](notebooks/day5_practice.ipynb) | - |
| 6 | 평가 · 모니터링 대시보드 | 📝 준비됨 | [day6_practice](notebooks/day6_practice.ipynb) | - |

> ✅ = 직접 실습 완료 · 📝 준비됨 = 노트북 작성 및 Jetson 실기기에서 코드 검증까지는 끝났지만 아직 직접 실습 전

## 구조

```
notebooks/   각 Day별 실습용 / 문제풀이 노트북 쌍
serving/     FastAPI 기반 추론 서버 코드
scripts/     환경 구축 및 포트폴리오 자동 게시 스크립트
docs/        벤치마크 결과, 회고
```

## 환경

- Jetson Orin Nano, JetPack 7.2 (L4T R39)
- 노트북은 VSCode에서 편집, 커널은 Jetson의 Jupyter 서버에 연결하여 실행

## 참고

- [docs/시행착오_기록.md](docs/시행착오_기록.md) — 환경 구성, Day별로 겪은 문제와 원인/해결 과정 정리
