# AGENTS.md

이 파일은 AGENTS.md 크로스툴 표준을 읽는 에이전트(Google Antigravity/Gemini 등)를 위한 진입점입니다.
Claude Code는 같은 내용을 `CLAUDE.md`에서 읽습니다 — 둘은 같은 프로젝트의 같은 규칙을 가리키므로,
**둘 중 하나만 수정하지 말고 항상 함께 갱신**하세요 (또는 이 파일을 갱신하는 대신 CLAUDE.md를 갱신한 뒤
이 파일 상단의 "최신 내용은 CLAUDE.md 참고" 문구만 유지해도 됩니다).

## 작업 시작 전 반드시 읽을 것 (순서대로)

1. **`CLAUDE.md`** — 아키텍처, 접속 경로, 알려진 하드웨어/소프트웨어 제약이 전부 여기 있습니다.
2. **`docs/시행착오_기록.md`** — 지금까지 겪은 문제와 원인/해결 과정 (노트북 markdown보다 더 자세함).
3. **`README.md`** — Day별 진행 상태(✅ 직접 실습 완료 / 📝 준비됨), 저장소 구조.

## 이 프로젝트 한 줄 요약

Jetson Orin Nano(JetPack 7.2) 한 대로 LLM MLOps 파이프라인(추론 서빙·양자화·LoRA·모니터링)과
비전 모델 경량화(양자화+TensorRT)를 실습하는 개인 학습 스프린트. **코드는 이 저장소에 있지만
실행은 물리적으로 분리된 Jetson에서 이루어짐** — 이게 가장 중요한 구조적 특징입니다.

## 지금 당장 알아야 할 핵심 3가지

1. **커널 위치 ≠ 파일 위치**: 노트북 커널은 SSH로 연결된 Jetson에서 돌지만, `.ipynb` 파일 자체는
   이 저장소(작업 PC)에 있습니다. 상대경로(`../serving` 등)를 코드에서 쓰면 Jetson 쪽 엉뚱한 곳에
   파일이 생깁니다 — 항상 어느 머신 기준 경로인지 구분하세요.
2. **작업 PC마다 Jetson 접속 경로가 다를 수 있음**: 지금 PC가 Jetson(`192.168.237.8`)에 직접
   `ping`이 되면 1홉으로 바로 SSH 가능(`ssh jetson`), 안 되면 `codeql-host`(`10.10.237.5`)를 거쳐야
   합니다 — 자세한 판단법은 CLAUDE.md의 "작업 PC가 여러 대일 수 있음" 절 참고.
3. **비밀번호는 이 저장소 어디에도 없습니다** (의도적). codeql-host/Jetson 접속 정보는 사용자에게
   직접 확인하세요. 이미 SSH 키가 등록되어 있다면(`~/.ssh/config` 확인) 비밀번호 없이 바로 됩니다.

## 하지 말아야 할 것 (자주 겪은 실패)

- `bitsandbytes`의 `load_in_8bit`/`load_in_4bit` — 토이 예제는 성공해도 실제 모델 `generate()`에서
  `cuBLAS API failed with status 15`로 실패함.
- `optimum-quanto`의 int4(AWQ) 경로를 안전장치(`MAX_JOBS=1` 등) 없이 실행 — 최초 실행 시 CUDA 확장을
  동시 병렬 컴파일하며 8GB RAM+2GB swap을 전부 소진해 **기기 전체가 멈춘 적이 실제로 있음**.
- `tokenizer.apply_chat_template(..., return_tensors="pt")`를 바로 쓰기 — `.shape` 접근에서
  `AttributeError`. `tokenize=False`로 문자열만 받고 별도로 `tokenizer(text, return_tensors="pt")` 할 것.
- `model.generate(..., use_cache=False)`로 KV cache 효과를 측정하려 하기 — 이 버전에서는 무시됨.

나머지 세부 사항(정확한 명령어, 실측 수치, 전체 트러블슈팅 히스토리)은 위 1~3번 문서에 있습니다.
