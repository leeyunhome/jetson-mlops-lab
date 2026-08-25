# Jetson MLOps Lab

Jetson Orin Nano (JetPack 7.2) 한 대로 LLM MLOps 파이프라인의 핵심 구성요소를 직접 구현하며 학습하는 7일 스프린트 기록입니다.

실행 결과가 포함된 노트북은 웹에서 바로 볼 수 있습니다 → **https://leeyunhome.github.io/jetson-mlops-lab/**

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

## 추가 트랙 (LLM 스프린트와 별개)

| 주제 | 상태 | 노트북 | 핵심 |
|---|---|---|---|
| 비전 모델 경량화 (torchvision 양자화 + TensorRT) | ✅ | [vision_compression_practice](notebooks/vision_compression_practice.ipynb) | ResNet-18을 INT8 양자화(CPU) vs TensorRT(GPU, FP32/FP16/INT8)로 비교. TensorRT INT8이 PyTorch eager 대비 10.4배 빠름(실측). CPU INT8은 4.2배 작아지지만 오히려 16% 느림 |

## 구조

```
notebooks/   각 Day별 실습용 / 문제풀이 노트북 쌍
serving/     FastAPI 기반 추론 서버 코드
scripts/     환경 구축 및 포트폴리오 자동 게시 스크립트
benchmarks/  측정 원본(trtexec 로그)과 이를 정리한 데이터셋
site/        GitHub Pages 랜딩 페이지 (차트)
docs/        벤치마크 결과, 회고
```

## 측정 결과 시각화

랜딩 페이지의 차트는 `benchmarks/vision_compression.json` 하나만 바라봅니다. 이 파일은
`trtexec` 원본 로그와 노트북 셀 출력에서 기계적으로 생성되므로, 그래프의 모든 숫자는
저장소 안의 원본까지 되짚어 확인할 수 있습니다.

```bash
python scripts/collect_benchmarks.py   # 로그 + 노트북 출력 → 데이터셋 갱신
```

노트북 셀을 다시 실행해 값이 바뀌면 위 명령만 다시 돌리면 됩니다. 단, TensorRT 엔진을
새로 빌드한 경우(`trtexec`를 돌리는 셀)에는 Jetson의 `~/vision_lab/trt_*.log`를
`benchmarks/logs/`로 다시 복사해야 새 측정이 반영됩니다.

차트는 외부 라이브러리 없이 인라인 SVG로 그리며, 라이트/다크 두 모드 모두에서
색각 이상 대비 검사를 통과한 팔레트를 씁니다. 값은 색에만 기대지 않도록 마크 옆에
직접 표기하고, 같은 수치를 표로도 제공합니다.

## 환경

- Jetson Orin Nano, JetPack 7.2 (L4T R39)
- 노트북은 VSCode에서 편집, 커널은 Jetson의 Jupyter 서버에 연결하여 실행

## 실행 결과 자동 기록

셀을 실행할 때마다 손으로 커밋하지 않도록, 노트북 저장을 감시해 새로 실행된 셀을 찾아
그 섹션·코드·출력을 담은 커밋 메시지로 자동 커밋·푸시하는 스크립트를 두었습니다.
표준 라이브러리만 사용하므로 별도 설치가 필요 없습니다.

```bash
python scripts/nbautocommit.py notebooks/vision_compression_practice.ipynb
```

| 옵션 | 설명 |
|---|---|
| `--dry-run` | 커밋하지 않고 생성될 메시지만 출력 |
| `--no-push` | 커밋만 하고 푸시하지 않음 |
| `--debounce N` | 마지막 저장 후 N초간 조용해야 커밋 (기본 8초, 연속 실행을 한 커밋으로 묶음) |
| `--once` | 감시 없이 한 번만 검사 |

노트북이 디스크에 저장되어야 감지되므로 VS Code 자동 저장을 켜두는 편이 편합니다.
감시 대상 노트북만 스테이징하므로 작업 트리의 다른 변경사항은 건드리지 않습니다.

## 참고

- [docs/시행착오_기록.md](docs/시행착오_기록.md) — 환경 구성, Day별로 겪은 문제와 원인/해결 과정 정리
