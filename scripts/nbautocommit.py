#!/usr/bin/env python3
"""노트북 셀 실행을 감시해 자동으로 커밋·푸시한다.

셀이 실행되어 노트북이 저장되면, 새로 실행된 셀을 찾아내 그 셀이 속한 섹션·
코드 요약·실제 출력을 담은 커밋 메시지를 만들고 커밋한다. 표준 라이브러리만
사용하므로 별도 설치가 필요 없다.

    python scripts/nbautocommit.py notebooks/vision_compression_practice.ipynb

여러 노트북을 동시에 감시할 수 있고, --dry-run으로 실제 커밋 없이 생성될
메시지만 확인할 수 있다. 자세한 옵션은 --help 참고.

동작 전제
    - 노트북이 디스크에 저장되어야 감지된다. VS Code에서는 자동 저장을 켜거나
      셀 실행 후 Ctrl+S를 눌러야 한다.
    - 감시 대상 노트북만 스테이징하므로, 작업 트리의 다른 변경사항은 건드리지
      않는다.
    - 첫 실행 시에는 현재 상태를 기준선으로만 기록하고 커밋하지 않는다.
      (--commit-existing으로 변경 가능)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# 커밋 메시지에 담을 분량 상한. 메시지가 로그를 통째로 삼키지 않도록 제한한다.
MAX_CODE_LINES = 8
MAX_OUTPUT_LINES = 12
MAX_LINE_CHARS = 100
MAX_SUBJECT_CHARS = 72


def log(msg: str) -> None:
    print(f"[nbautocommit] {msg}", flush=True)


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


# ---------------------------------------------------------------- 노트북 읽기


@dataclass
class Cell:
    """커밋 메시지를 만드는 데 필요한 만큼만 추린 코드 셀."""

    key: str
    index: int
    execution_count: int | None
    source: str
    heading: str
    stdout: list[str]
    error: str | None

    def fingerprint(self) -> str:
        """실행 여부를 판별하기 위한 지문 (실행 횟수 + 출력 내용)."""
        payload = json.dumps(
            [self.execution_count, self.stdout, self.error], ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def extract_heading(cells: list[dict], index: int) -> str:
    """해당 셀 바로 앞에 오는 마크다운 heading을 찾아 섹션명으로 쓴다."""
    for prev in range(index - 1, -1, -1):
        cell = cells[prev]
        if cell.get("cell_type") != "markdown":
            continue
        for line in reversed("".join(cell.get("source", [])).splitlines()):
            if line.lstrip().startswith("#"):
                return line.lstrip("#").strip()
    return ""


def collect_outputs(cell: dict) -> tuple[list[str], str | None]:
    """셀 출력에서 stdout 텍스트와 예외 정보를 뽑아낸다."""
    stdout: list[str] = []
    error: str | None = None
    for out in cell.get("outputs", []):
        kind = out.get("output_type")
        if kind == "stream" and out.get("name") == "stdout":
            stdout.extend("".join(out.get("text", [])).splitlines())
        elif kind == "execute_result":
            text = "".join(out.get("data", {}).get("text/plain", []))
            stdout.extend(text.splitlines())
        elif kind == "error":
            error = f"{out.get('ename', 'Error')}: {out.get('evalue', '')}".strip()
    return stdout, error


def read_cells(path: Path) -> list[Cell]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    raw_cells = nb.get("cells", [])
    cells: list[Cell] = []
    for i, cell in enumerate(raw_cells):
        if cell.get("cell_type") != "code":
            continue
        stdout, error = collect_outputs(cell)
        cells.append(
            Cell(
                key=cell.get("id") or f"index:{i}",
                index=i,
                execution_count=cell.get("execution_count"),
                source="".join(cell.get("source", [])),
                heading=extract_heading(raw_cells, i),
                stdout=stdout,
                error=error,
            )
        )
    return cells


# ------------------------------------------------------------ 커밋 메시지 작성


def clip(line: str) -> str:
    line = line.rstrip()
    return line if len(line) <= MAX_LINE_CHARS else line[: MAX_LINE_CHARS - 1] + "…"


def summarize_code(source: str) -> list[str]:
    """주석·빈 줄을 걷어내고 핵심 코드 줄만 남긴다."""
    lines = [
        ln.rstrip()
        for ln in source.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if len(lines) <= MAX_CODE_LINES:
        return [clip(ln) for ln in lines]
    head = [clip(ln) for ln in lines[: MAX_CODE_LINES - 1]]
    return head + [f"... (총 {len(lines)}줄)"]


def summarize_output(cell: Cell) -> list[str]:
    """경고·트레이스백 잡음을 걷어낸 의미 있는 출력만 남긴다."""
    lines = [ln for ln in cell.stdout if ln.strip()]
    if len(lines) <= MAX_OUTPUT_LINES:
        return [clip(ln) for ln in lines]
    kept = [clip(ln) for ln in lines[: MAX_OUTPUT_LINES - 1]]
    return kept + [f"... ({len(lines) - MAX_OUTPUT_LINES + 1}줄 생략)"]


def build_subject(stem: str, cells: list[Cell]) -> str:
    if len(cells) == 1:
        cell = cells[0]
        subject = f"Run {stem} cell #{cell.index}"
        if cell.heading:
            subject += f" — {cell.heading}"
    else:
        span = f"#{cells[0].index}–#{cells[-1].index}"
        subject = f"Run {stem} cells {span} ({len(cells)} cells)"

    if len(subject) > MAX_SUBJECT_CHARS:
        subject = subject[: MAX_SUBJECT_CHARS - 1] + "…"
    return subject


def build_message(notebook: Path, cells: list[Cell]) -> str:
    stem = notebook.stem
    parts = [build_subject(stem, cells), ""]

    if len(cells) > 1:
        sections: list[str] = []
        for cell in cells:
            if cell.heading and cell.heading not in sections:
                sections.append(cell.heading)
        if sections:
            parts.append("다룬 섹션:")
            parts.extend(f"  - {clip(s)}" for s in sections)
            parts.append("")

    if any(c.error for c in cells):
        parts.append("일부 셀이 예외로 종료됨 — 시행착오 기록 목적으로 함께 남긴다.")
        parts.append("")

    for cell in cells:
        header = f"셀 #{cell.index}"
        if cell.execution_count is not None:
            header += f" (In[{cell.execution_count}])"
        if cell.heading:
            header += f" — {cell.heading}"
        parts.append(header)

        code = summarize_code(cell.source)
        if code:
            parts.append("  코드:")
            parts.extend(f"    {ln}" for ln in code)

        output = summarize_output(cell)
        if output:
            parts.append("  출력:")
            parts.extend(f"    {ln}" for ln in output)

        if cell.error:
            parts.append(f"  예외: {clip(cell.error)}")

        parts.append("")

    parts.append(f"{notebook.as_posix()} 실행 결과를 자동 기록 (nbautocommit).")
    return "\n".join(parts).rstrip() + "\n"


# ------------------------------------------------------------------ 상태 관리


def state_path(repo: Path, notebook: Path) -> Path:
    digest = hashlib.sha256(notebook.as_posix().encode("utf-8")).hexdigest()[:12]
    directory = repo / ".git" / "nbautocommit"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{digest}.json"


def load_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(path: Path, cells: list[Cell]) -> None:
    snapshot = {c.key: c.fingerprint() for c in cells}
    path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")


def newly_executed(cells: list[Cell], previous: dict[str, str]) -> list[Cell]:
    """이전 스냅샷 대비 새로 실행된(또는 재실행된) 셀만 골라낸다."""
    changed = []
    for cell in cells:
        if cell.execution_count is None and not cell.stdout and not cell.error:
            continue  # 아직 실행된 적 없는 셀
        if previous.get(cell.key) != cell.fingerprint():
            changed.append(cell)
    return changed


# ---------------------------------------------------------------- 감시 루프


def wait_until_stable(path: Path, debounce: float, interval: float) -> None:
    """파일 쓰기가 끝날 때까지, 그리고 연속 실행이 잦아들 때까지 기다린다."""
    last = None
    quiet_since = time.monotonic()
    while True:
        try:
            stat = path.stat()
            signature = (stat.st_mtime, stat.st_size)
        except OSError:
            signature = None
        if signature != last:
            last = signature
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= debounce:
            return
        time.sleep(interval)


def has_staged_changes(repo: Path) -> bool:
    result = run_git(repo, "diff", "--cached", "--quiet", check=False)
    return result.returncode != 0


def commit_and_push(
    repo: Path, notebook: Path, message: str, push: bool, remote: str, branch: str
) -> bool:
    relative = notebook.relative_to(repo).as_posix()
    run_git(repo, "add", "--", relative)

    if not has_staged_changes(repo):
        log("노트북 내용이 마지막 커밋과 동일하다 — 커밋 생략")
        run_git(repo, "reset", "--quiet", "HEAD", "--", relative, check=False)
        return False

    message_file = repo / ".git" / "nbautocommit" / "COMMIT_MSG"
    message_file.write_text(message, encoding="utf-8")
    run_git(repo, "-c", "i18n.commitEncoding=UTF-8", "commit", "-F", str(message_file))
    log(f"커밋 완료: {message.splitlines()[0]}")

    if push:
        result = run_git(repo, "push", remote, branch, check=False)
        if result.returncode == 0:
            log(f"푸시 완료 → {remote}/{branch}")
        else:
            log(f"푸시 실패 (커밋은 로컬에 남아 있음): {result.stderr.strip()}")
            return False
    return True


def process(notebook: Path, repo: Path, args: argparse.Namespace) -> None:
    cells = read_cells(notebook)
    store = state_path(repo, notebook)
    previous = load_state(store)

    if not previous and not args.commit_existing:
        save_state(store, cells)
        log(f"{notebook.name}: 현재 상태를 기준선으로 기록 (이번엔 커밋하지 않음)")
        return

    changed = newly_executed(cells, previous)
    if not changed:
        return

    log(f"{notebook.name}: 새로 실행된 셀 {len(changed)}개 감지")
    message = build_message(notebook.relative_to(repo), changed)

    if args.dry_run:
        print("-" * 60)
        print(message, end="")
        print("-" * 60)
        return

    commit_and_push(repo, notebook, message, args.push, args.remote, args.branch)
    save_state(store, cells)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="노트북 셀 실행을 감시해 자동 커밋·푸시한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("notebooks", nargs="+", type=Path, help="감시할 .ipynb 경로")
    parser.add_argument(
        "--interval", type=float, default=2.0, help="파일 검사 주기(초). 기본 2"
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=8.0,
        help="마지막 저장 후 이 시간(초)만큼 조용해야 커밋한다. "
        "값을 키우면 연속 실행이 한 커밋으로 묶인다. 기본 8",
    )
    parser.add_argument("--remote", default="origin", help="푸시할 remote. 기본 origin")
    parser.add_argument("--branch", default=None, help="푸시할 브랜치. 기본 현재 브랜치")
    parser.add_argument("--no-push", dest="push", action="store_false", help="커밋만 하고 푸시하지 않는다")
    parser.add_argument("--dry-run", action="store_true", help="커밋 없이 메시지만 출력")
    parser.add_argument("--once", action="store_true", help="한 번만 검사하고 종료")
    parser.add_argument(
        "--commit-existing",
        action="store_true",
        help="첫 실행에서 이미 실행된 셀들도 커밋 대상에 포함한다",
    )
    args = parser.parse_args()

    notebooks = [p.resolve() for p in args.notebooks]
    for path in notebooks:
        if not path.exists():
            log(f"파일을 찾을 수 없다: {path}")
            return 1

    result = run_git(notebooks[0].parent, "rev-parse", "--show-toplevel", check=False)
    if result.returncode != 0:
        log("git 저장소 안에서 실행해야 한다")
        return 1
    repo = Path(result.stdout.strip()).resolve()

    if args.branch is None:
        args.branch = run_git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    log(f"저장소: {repo}")
    log(f"감시 대상: {', '.join(p.name for p in notebooks)}")
    if not args.dry_run:
        log(f"커밋 후 푸시: {'예 → ' + args.remote + '/' + args.branch if args.push else '아니오'}")

    if args.once:
        for path in notebooks:
            process(path, repo, args)
        return 0

    log(f"감시 시작 (검사 {args.interval}초 / 디바운스 {args.debounce}초). 중지: Ctrl+C")
    seen = {p: p.stat().st_mtime for p in notebooks}
    try:
        while True:
            time.sleep(args.interval)
            for path in notebooks:
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if mtime == seen.get(path):
                    continue
                wait_until_stable(path, args.debounce, args.interval)
                seen[path] = path.stat().st_mtime
                try:
                    process(path, repo, args)
                except Exception as exc:  # 감시는 계속되어야 한다
                    log(f"{path.name} 처리 중 오류: {exc}")
    except KeyboardInterrupt:
        log("감시 종료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
