# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

Jetson Orin Nano(JetPack 7.2, L4T R39) 한 대로 LLM MLOps 파이프라인 핵심 구성요소(추론 서빙, 모델 최적화, 파인튜닝, 모니터링)를 직접 구현하며 학습하는 7일 스프린트 기록. 클라우드 멀티노드 대신 리소스가 제한된 엣지 디바이스에서 원리를 바닥부터 익히는 것이 목적.

## 핵심 아키텍처: 코드는 이 저장소에, 실행은 원격 Jetson에서

이 저장소의 노트북/스크립트는 로컬에서 실행되지 않는다. 실제 실행 환경은 물리적으로 분리된 Jetson Orin Nano이며, 접근 경로는 다음과 같은 이중 SSH 홉이다:

```
작업 PC → codeql-host (10.10.237.5, 게이트웨이 리눅스 머신) → Jetson (192.168.237.8, 이더넷 고정 IP)
```

- **주 연결 경로는 이더넷 고정 IP `192.168.237.8`이다** (codeql-host의 `enp4s0`와 같은 `192.168.237.0/16` 대역). USB C타입 케이블로 `192.168.55.1`(l4tbr0 브릿지, USB 전용 연결) 경로도 여전히 존재하지만, 이건 다른 용도로 케이블을 뺄 수도 있는 **보조/폴백 경로**다 — USB가 물리적으로 빠져 있어도 이더넷이 연결되어 있으면 Jetson 접속에는 문제 없다.
- codeql-host 로그인 계정: `codeql-host`. Jetson 로그인 계정: `manager`. 비밀번호는 코드/문서에 남기지 않으므로 별도 확인 필요.
- Jetson에는 `sshpass`가 설치되어 있어 codeql-host에서 Jetson으로 비밀번호 기반 비대화형 접속이 가능하다: `sshpass -p <pw> ssh manager@192.168.237.8 '<command>'`.
- Jetson의 Python 가상환경은 `~/mlops-lab-env` (Python 3.12.3). torch/transformers/accelerate 등은 전부 이 venv 안에 설치되어 있고, 시스템 파이썬에는 없다.
- 작업 PC의 `~/.ssh/config`에 `Host jetson`(HostName `192.168.237.8`, ProxyJump `codeql-host`) alias가 등록되어 있어 `ssh jetson` 한 줄로 접속 가능.

### 작업 PC가 여러 대일 수 있음 — 경로가 다를 수 있다

- **작업 PC #1 (Windows, devuser)**: 사내망 밖 세그먼트에 있어 Jetson(`192.168.237.8`)에 직접 못 붙고 **codeql-host를 반드시 거쳐야 함** (이중 홉). `~/.ssh/config`의 `jetson` 항목이 `ProxyJump codeql-host`로 되어 있는 이유.
- **작업 PC #2 (Linux, deeplearning@deeplearning-H110-D3, 10.10.237.222)**: codeql-host와 같은 사내망 세그먼트에 있어 **Jetson에 직접(단일 홉) SSH 가능** — `ping`/`ssh` 둘 다 codeql-host 경유 없이 바로 됨. 이 PC의 `~/.ssh/config`는 `jetson` 항목에 `ProxyJump` 없이 `HostName 192.168.237.8`만 지정되어 있음. Jupyter 터널도 이 PC→Jetson 1단계 포트포워딩(`ssh -N -L 8888:127.0.0.1:8888 jetson`)만으로 충분, codeql-host를 안 거쳐도 됨.
- **결론**: "codeql-host를 반드시 거쳐야 한다"는 규칙이 아니라 **"그 PC가 Jetson과 같은 네트워크 세그먼트에 있는지"가 핵심**이다. 새 작업 PC에서는 먼저 `ping 192.168.237.8`로 직접 도달 여부를 확인하고, 되면 codeql-host 홉을 생략할 것.

## Jupyter 커널 연결 방법

Jetson의 Jupyter Lab 서버는 **의도적으로 `127.0.0.1`(로컬호스트 전용)에만 바인딩**되어 있다 — 네트워크에 코드 실행 서비스를 노출하지 않기 위함. 접근하려면 SSH 포트포워딩만 사용해야 한다 (`--ip=0.0.0.0` 등으로 네트워크에 노출하는 방식은 지양):

1. codeql-host에서 Jetson으로: `ssh -N -L 8888:127.0.0.1:8888 manager@192.168.237.8` (백그라운드로 유지)
2. 작업 PC에서 codeql-host로: `ssh -N -L 8888:127.0.0.1:8888 codeql-host@10.10.237.5` (백그라운드로 유지)
3. VS Code에서 노트북 열고 커널 선택 → **Existing Jupyter Server** → `http://localhost:8888/?token=<Jupyter 시작 로그에 출력된 토큰>`
4. 커널 목록에 뜨는 **"MLOps Lab (Jetson)"**(`kernelspec: mlops-lab`) 선택 — 기본 "Python 3 (ipykernel)"과 동일한 venv를 가리키므로 아무거나 골라도 되지만 이름이 명확한 쪽을 권장.

Jupyter 서버 실행 (Jetson 쪽, `~`에서):
```
nohup ~/mlops-lab-env/bin/jupyter lab --no-browser --ip=127.0.0.1 --port=8888 > ~/jupyter.log 2>&1 &
```
토큰은 `~/jupyter.log`에 출력된다.

## 알려진 하드웨어/소프트웨어 제약

- **GPU compute capability 8.7 (Orin) 관련 경고**: PyPI의 일반 torch 휠은 sm_87 전용 사전 컴파일 커널을 포함하지 않아 `torch.cuda.is_available()` 등에서 "No published PyTorch CUDA builds... support this GPU" 경고가 뜬다. 이는 아키텍처가 잘못 설치된 게 아니라(aarch64 휠 맞음), 인접 CC(8.0)용 PTX를 런타임에 JIT 컴파일해서 대체 실행하기 때문 — 기본 연산(matmul, 소형 LLM 추론)은 정상 동작하지만 첫 호출이 느리다. 단, **flash-attention/xformers/bitsandbytes처럼 아키텍처별 사전 컴파일 커널에 의존하는 라이브러리는 PTX 폴백이 없어 설치·동작 자체가 안 될 수 있음** — Day 5(LoRA) 진행 시 NVIDIA jetson-containers/dusty-nv 배포판 검토 필요.
- **메모리는 8GB 통합 메모리(LPDDR5)** — CPU/OS와 공유되므로 모델에 실제로 쓸 수 있는 여유는 5~6GB 수준. 7B급 fp16 모델은 애초에 안 올라가며, 4bit 양자화나 3B급 이하 모델 중심으로 설계할 것.
- **HuggingFace `AutoModelForCausalLM.from_pretrained(..., torch_dtype=...)`의 `torch_dtype` 인자는 deprecated** — 현재 설치된 transformers 버전에서는 `dtype=`을 사용해야 경고 없이 동작한다.
- **`tokenizer.apply_chat_template(..., return_tensors="pt")`를 바로 쓰지 말 것** — 이 환경에서는 `.shape` 접근 시 `AttributeError`가 난다. 항상 `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`로 문자열을 얻은 뒤 `tokenizer(text, return_tensors="pt")`로 별도 토큰화할 것.
- **`model.generate(..., use_cache=False)`가 이 transformers 버전에서 무시된다** — 실제로 측정해도 속도 차이가 거의 없다. KV cache 효과를 보여주려면 `model()`을 직접 반복 호출하는 수동 루프로 "캐시 재사용 vs 매 스텝 전체 재계산"을 구현해야 한다 (Day2 참고).
- **`bitsandbytes`는 import와 소규모 토이 텐서 연산은 성공하지만, 실제 모델 `generate()` 도중 `cuBLAS API failed with status 15`로 실패한다** — 토이 예제 성공을 신뢰하지 말 것. `load_in_8bit`/`load_in_4bit` 전반에 이 문제가 있다.
- **`optimum-quanto`의 int4(AWQ 커널)는 최초 실행 시 CUDA 확장 여러 개를 동시에 JIT 컴파일하며 8GB RAM + 2GB swap을 전부 소진해 기기 전체가 멈출 수 있다** (실제로 재현됨, load average 20 근접, SSH 응답 불가 수 분 지속). `ninja` 미설치 시엔 먼저 그 에러가 뜨고, 설치 후 재시도하면 이 메모리 고갈이 발생한다. int4가 필요하면 GGUF+llama.cpp 사전 양자화 체크포인트 경로를 쓸 것 — 즉석 컴파일이 없다.
- **venv 안의 콘솔 스크립트(예: `ninja`)가 PATH에 없다** — `~/mlops-lab-env/bin/python3`를 직접 호출하는 방식(activate 스크립트 미실행)에서는 `os.environ["PATH"]`에 `~/mlops-lab-env/bin`이 안 잡혀 있어, 이를 필요로 하는 JIT 컴파일 등이 실패할 수 있다. 필요시 `os.environ["PATH"] = os.path.expanduser("~/mlops-lab-env/bin") + os.pathsep + os.environ.get("PATH", "")`로 직접 보정.
- **matplotlib/pandas/fastapi/uvicorn/peft/datasets/ninja/optimum-quanto/bitsandbytes는 기본 venv에 없다** — 필요할 때 그때그때 `~/mlops-lab-env/bin/pip install`로 설치했음 (전부 aarch64 설치 자체는 문제없이 됨).
- **스트리밍 기반 측정(`TextIteratorStreamer`)에서 워밍업 없이 콜드 스타트로 측정하면 TTFT가 수십 초(관측치: 49초)까지 튈 수 있다** — Day1/Day2에서 본 CC8.7 PTX JIT 비용과 같은 원인이지만 스트리밍 경로에서는 유독 크게 나타났다. 성능 측정 코드에는 항상 워밍업 호출을 먼저 넣을 것.

### 비전 경량화 트랙 (vision_compression_practice.ipynb) 관련
- **`torchvision`은 반드시 `pip install --no-deps torchvision`으로 설치할 것** — 그냥 설치하면 의존성 해결 중 특수 빌드된 `torch 2.13.0+cu130`(Jetson CUDA 빌드)을 일반 빌드로 덮어써 CUDA가 깨질 수 있다. `--no-deps`로 0.28.0 설치 시 torch 안 건드리고 CUDA 정상 동작 확인함.
- **PyTorch 양자화 백엔드 기본값이 `x86`이라 aarch64에서 `RuntimeError: unknown architecture`가 난다** — `torch.backends.quantized.engine = "qnnpack"`(ARM 백엔드)를 명시해야 동작. torchvision `resnet18(quantize=True)`의 내부 백엔드 자동선택도 실패하므로, quantizable 아키텍처에 float 가중치를 얹고 수동 PTQ(fuse→qconfig(qnnpack)→prepare→calibrate→convert)로 진행함.
- **CPU INT8 양자화는 크기는 ~1/4로 줄지만 이 Jetson CPU에서 오히려 느림**(실측 FP32 734ms vs INT8 906ms/batch16). torchvision 양자화 모델은 CPU 전용이라 GPU도 못 씀 — 엣지 속도 향상은 TensorRT로 가야 한다는 결론.
- **TensorRT는 `trtexec`가 JetPack에 기본 포함(`/usr/bin/trtexec`)** — 별도 파이썬 바인딩 불필요. PyTorch→ONNX→`trtexec`로 엔진 빌드+벤치. 실측: ResNet-18 batch16 GPU Compute Time이 PyTorch eager 56ms → TRT FP32 27.8 → FP16 10.6 → INT8 5.4ms (INT8은 eager 대비 ~10배).
- **torch 2.13 ONNX export는 `onnxscript` 필요**하고, 큰 가중치를 **외부 데이터 파일(`*.onnx.data`)로 분리 저장**한다 — `.onnx`와 `.onnx.data`가 같은 폴더에 함께 있어야 trtexec가 읽는다.
- **PowerShell→plink 원격 명령에서 `$?`를 쓰지 말 것** — PowerShell이 자체 `$?`(True/False)로 확장해버려 원격 bash에 엉뚱하게 전달된다. 원격 종료코드가 필요하면 다른 방식(로그 파일 확인 등)을 쓸 것. (`\"a|b\"` 형태의 grep 패턴도 PowerShell 파서가 깨뜨리므로, 원격에서 grep 대신 파일로 저장 후 `tail`로 읽는 게 안전.)

### 객체 계수 파이프라인 트랙 (vision_counting_pipeline_practice.ipynb) 관련
- **`ultralytics`는 반드시 `--no-deps`로 설치할 것** — `ultralytics`와 `ultralytics-thop` 둘 다 torch를 의존성으로 걸고 있어, 그냥 설치하면 위 torchvision과 똑같이 `torch 2.13.0+cu130`을 일반 빌드로 덮어쓴다. torch와 무관한 순수 의존성(opencv-python-headless, psutil, py-cpuinfo, PyYAML, matplotlib, scipy, pandas)만 정상 설치한다. 추적 실행 시 ultralytics가 `lap`을 자체 AutoUpdate로 설치하는데, numpy만 의존하므로 무해하다.
- **설치 검증에 `importlib.reload(torch)`를 쓰지 말 것** — 이미 로드된 torch를 reload하면 `TORCH_LIBRARY('triton')` 중복 등록으로 `RuntimeError`가 난다. `importlib.metadata.version("torch")`로 **디스크 메타데이터**를 설치 전후 비교하는 방식이 안전하고, pip 교체 감지에도 이쪽이 정확하다.
- **matplotlib 그래프 텍스트에 한글을 쓰지 말 것** — 이 Jetson에 한글 폰트가 없어 `Glyph missing from font(s) DejaVu Sans` 경고와 함께 □로 렌더링된다. 그래프 제목·축은 영문, `print` 출력과 마크다운 설명은 한글로 분리한다.
- **노트북별로 새 커널을 띄울 것** — 이미 다른 노트북에 물린 커널을 고르면 `WORK`, `model` 같은 전역 변수가 충돌한다.
- **샘플 이미지는 실행 시점 다운로드만 하고 저장소에 포함하지 않는다** (COCO val2017 / Laboro Tomato CC BY-NC-SA).
- **`trtexec --dumpProfile` 표는 숫자 4열이 앞, 레이어 이름이 뒤**(TensorRT 10.x). 마지막 `Total` 행은 집계에서 제외해야 한다.

### YOLO-World 트랙 (yolo_world_greenhouse_practice.ipynb) 관련
- **`model.set_classes([...])`는 추론 직전에 매번 호출할 것** — 이전 셀에서 다른 프롬프트로 설정했다면 남아있는 상태로 추론된다. 프롬프트를 바꿔가며 비교하는 셀([코드 4])에서는 루프 안에서 매번 `set_classes`를 다시 부른다.
- `yolov8s-world.pt`는 `ultralytics`가 최초 1회 자동 다운로드한다(`YOLO11n`과 마찬가지로 이미 `--no-deps` 설치된 환경 재사용, 추가 pip 설치 없음).
- 오픈보캡 검출은 폐쇄형(YOLO11n) 대비 느리다 — 실시간이 필요하면 이 트랙은 프로토타이핑/희귀 클래스용으로만 쓰고 배포는 파인튜닝된 폐쇄형 모델로 가야 한다.

### YOLO-World 파인튜닝 (yolo_world_greenhouse_practice.ipynb 4~5단계) 관련
- **COCO bbox → YOLO 라벨 변환 시 클래스는 단일(`tomato`)로 합칠 것** — 3클래스(완숙/반숙/미숙)로 하면 제로샷 프롬프트(`"tomato"` 하나)와 비교 기준이 어긋난다.
- **`tomato_ripeness_training_practice`의 데이터를 재사용** — `~/tomato_lab/laboro_tomato_big`를 다시 받지 않고 그대로 쓴다. 그 노트북의 [코드 1]을 먼저 실행해둬야 한다.
- **파인튜닝 전/후 비교는 검출 개수가 아니라 신뢰도로 볼 것** — "tomato"는 흔한 단어라 제로샷도 이미 잘 잡는 경우가 많다. 개수 차이가 작다고 파인튜닝이 무의미했다고 결론 내리지 말 것.

## 진행 현황 및 구조

Day별 진행 상태는 `README.md`의 표로 관리한다 (Day 0~6, PyTorch→HuggingFace부터 평가/모니터링까지). **✅(직접 실습 완료)와 📝 준비됨(노트북 작성·Jetson 실기기 검증은 끝났지만 사용자가 아직 직접 실습 전)을 구분할 것** — Claude가 노트북을 대신 작성/검증했다고 곧바로 ✅로 표시하지 말고, 사용자가 실제로 그 Day를 진행했다고 확인해줄 때만 ✅로 올릴 것.

```
notebooks/   각 Day별 실습용 / 문제풀이 노트북 쌍 (예: day1_practice.ipynb)
serving/     FastAPI 기반 추론 서버 코드
scripts/     환경 구축 및 포트폴리오 자동 게시 스크립트
docs/        벤치마크 결과, 회고
```
