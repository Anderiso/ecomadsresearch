import os
import re
import json
import math
import asyncio
import logging
import tempfile
import subprocess
import imageio_ffmpeg
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("adremix")

app = FastAPI(title="AdRemix")

# ---------------------------------------------------------------------------
# Clients – initialised lazily so the app still boots even without keys
# (useful for UI-only dev). Endpoints that need them will fail with a clear
# error instead.
# ---------------------------------------------------------------------------

def _openai():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(500, "OPENAI_API_KEY is not set in .env")
    return OpenAI(api_key=key)

def _anthropic():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(500, "ANTHROPIC_API_KEY is not set in .env")
    return Anthropic(api_key=key)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
SEGMENT_MAX_TOKENS = int(os.getenv("SEGMENT_MAX_TOKENS", "8192"))
REWRITE_MAX_TOKENS = int(os.getenv("REWRITE_MAX_TOKENS", "8192"))
SCRIPT_MAX_ATTEMPTS = int(os.getenv("SCRIPT_MAX_ATTEMPTS", "3"))
PARTITION_MAX_ATTEMPTS = int(os.getenv("PARTITION_MAX_ATTEMPTS", "3"))
PARTITION_MAX_TOKENS = int(os.getenv("PARTITION_MAX_TOKENS", "4096"))
PACE_WORD_TOLERANCE = 5
PACE_SEVERE_UNDER_WORDS = 10
PACE_SEVERE_OVER_WORDS = 10
ORIGINAL_LENGTH_TOLERANCE = 20
REWRITE_LOG_OUTPUT_MAX_CHARS = 12000

WHISPER_MAX_BYTES = 25 * 1024 * 1024
WHISPER_TARGET_BYTES = int(24.5 * 1024 * 1024)  # safety margin under OpenAI's cap
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MIN_MP3_BITRATE_KBPS = 32
MAX_MP3_BITRATE_KBPS = 192

# ---------------------------------------------------------------------------
# Audio prep – extract full track & compress when upload exceeds Whisper limit
# ---------------------------------------------------------------------------

def _ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()

def _media_duration_seconds(path: str) -> float:
    proc = subprocess.run(
        [_ffmpeg_exe(), "-i", path],
        capture_output=True,
        text=True,
    )
    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)",
        proc.stderr or "",
    )
    if not match:
        return 0.0
    h, m, s, cs = (int(x) for x in match.groups())
    return h * 3600 + m * 60 + s + cs / 100

def _extract_full_audio_mp3(input_path: str, output_path: str, bitrate_kbps: int, mono: bool) -> None:
    """Extract the complete audio track (no trimming) to MP3."""
    cmd = [
        _ffmpeg_exe(), "-y",
        "-i", input_path,
        "-vn",
        "-map", "0:a:0?",
        "-acodec", "libmp3lame",
        "-b:a", f"{bitrate_kbps}k",
    ]
    if mono:
        cmd.extend(["-ac", "1"])
    cmd.append(output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "ffmpeg failed").strip()
        raise RuntimeError(err[-800:])

def compress_audio_for_whisper(input_path: str) -> tuple[str, int]:
    """Return (mp3 path, compressed size bytes). Full audio preserved; only re-encoded."""
    duration = _media_duration_seconds(input_path)
    output_path = tempfile.mktemp(suffix=".mp3")

    bitrates: list[int] = []
    if duration > 0:
        ideal = int((WHISPER_TARGET_BYTES * 8) / duration / 1000)
        ideal = max(MIN_MP3_BITRATE_KBPS, min(MAX_MP3_BITRATE_KBPS, ideal))
        bitrates.append(ideal)
    for rate in (128, 96, 64, 48, 32):
        if rate not in bitrates:
            bitrates.append(rate)

    last_error = "Could not extract audio from this file."
    for mono in (False, True):
        for bitrate in bitrates:
            try:
                _extract_full_audio_mp3(input_path, output_path, bitrate, mono)
            except RuntimeError as e:
                last_error = str(e)
                continue
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                last_error = "No audio track found in this file."
                continue
            size = os.path.getsize(output_path)
            if size <= WHISPER_MAX_BYTES:
                return output_path, size
            last_error = f"Audio still {size / (1024 * 1024):.1f} MB at {bitrate}kbps."

    if os.path.isfile(output_path):
        os.unlink(output_path)
    raise HTTPException(
        413,
        "Could not compress the full audio under 25 MB for Whisper. "
        "The recording may be too long — try a shorter clip. "
        f"({last_error})",
    )

async def _save_upload_to_temp(file: UploadFile, suffix: str) -> tuple[str, int]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        size = 0
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp_path = tmp.name
                tmp.close()
                os.unlink(tmp_path)
                raise HTTPException(413, "File must be under 500 MB.")
            tmp.write(chunk)
        return tmp.name, size

def _whisper_transcribe(client: OpenAI, audio_path: str):
    with open(audio_path, "rb") as f:
        return client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

def _format_whisper_result(result, *, compressed: bool = False, original_size: int = 0, audio_size: int = 0):
    segments = []
    for s in (result.segments or []):
        seg = s if isinstance(s, dict) else s.__dict__
        segments.append({
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": seg.get("text", "").strip(),
        })

    words = []
    for w in (result.words or []):
        wd = w if isinstance(w, dict) else w.__dict__
        words.append({
            "start": wd.get("start", 0),
            "end": wd.get("end", 0),
            "word": wd.get("word", ""),
        })

    duration = segments[-1]["end"] if segments else 0

    payload = {
        "text": result.text,
        "segments": segments,
        "words": words,
        "duration": round(duration, 2),
        "compressed": compressed,
    }
    if compressed:
        payload["original_size_mb"] = round(original_size / (1024 * 1024), 2)
        payload["audio_size_mb"] = round(audio_size / (1024 * 1024), 2)
    return JSONResponse(payload)

def _strip_json_fences(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.lstrip().startswith("json"):
            raw = raw.split("\n", 1)[1]
    if raw.rstrip().endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()

def _parse_json_object(text: str) -> dict:
    raw = _strip_json_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
        raise

def _count_words(text: str) -> int:
    return len((text or "").split())

def _word_list(text: str) -> list[str]:
    return (text or "").split()

def _format_segment_time(total_seconds: int) -> str:
    minutes, seconds = divmod(int(total_seconds), 60)
    return f"{minutes}:{seconds:02d}"

def calc_segment_plan(duration_sec: float, segment_length: int) -> tuple[int, float]:
    """Segment count = ceil(duration / segment_length), e.g. 118s → 8 × 15s chunks."""
    segment_length = max(1, int(segment_length))
    duration_sec = max(0.1, float(duration_sec))
    num_segments = max(1, math.ceil(duration_sec / segment_length))
    effective_duration = num_segments * segment_length
    return num_segments, float(effective_duration)

def _segment_word_targets(total_words: int, num_segments: int) -> list[int]:
    """Even split of total_words across segments (e.g. 411 over 8 → three 52s, five 51s)."""
    base = max(1, total_words // num_segments)
    extra = total_words % num_segments
    return [base + (1 if i < extra else 0) for i in range(num_segments)]

def _build_pace_plan(
    duration_sec: float,
    segment_length: int,
    target_wpm: int,
    original_word_count: int,
) -> dict:
    num_segments, effective_duration = calc_segment_plan(duration_sec, segment_length)
    segment_word_targets = _segment_word_targets(original_word_count, num_segments)
    words_per_segment = round(original_word_count / num_segments)
    wpm_words_per_segment = max(1, round((target_wpm / 60) * segment_length))
    total_word_target = original_word_count
    total_word_min = max(1, original_word_count - ORIGINAL_LENGTH_TOLERANCE)
    total_word_max = original_word_count + ORIGINAL_LENGTH_TOLERANCE
    return {
        "num_segments": num_segments,
        "effective_duration": effective_duration,
        "segment_length": segment_length,
        "target_wpm": target_wpm,
        "original_word_count": original_word_count,
        "words_per_segment": words_per_segment,
        "wpm_words_per_segment": wpm_words_per_segment,
        "segment_word_targets": segment_word_targets,
        "total_word_target": total_word_target,
        "total_word_min": total_word_min,
        "total_word_max": total_word_max,
    }

def _word_targets_for_script(full_script: str, num_segments: int) -> list[int]:
    return _segment_word_targets(_count_words(full_script), num_segments)

def _boundary_score_at_cut(words: list[str], end_exclusive: int) -> int:
    if end_exclusive <= 0 or end_exclusive > len(words):
        return -999
    prev = words[end_exclusive - 1]
    if prev.endswith((".", "!", "?")):
        return 100
    if prev.endswith((",", ";", ":")):
        return 50
    if prev.endswith(("…", '"', "'")):
        return 30
    return 0

def _partition_full_script_server(full_script: str, word_targets: list[int]) -> list[dict]:
    """Deterministic even split with phrase-boundary snapping (fallback if Claude partition fails)."""
    words = _word_list(full_script)
    if not words:
        return []

    segments: list[dict] = []
    start = 0
    n = len(word_targets)

    for i, target in enumerate(word_targets):
        if i == n - 1:
            chunk = words[start:]
        else:
            ideal_end = start + target
            min_end = start + max(1, target - PACE_WORD_TOLERANCE)
            max_end = start + target + PACE_WORD_TOLERANCE

            remaining_min = sum(
                max(1, word_targets[j] - PACE_WORD_TOLERANCE)
                for j in range(i + 1, n)
            )
            max_end = min(max_end, len(words) - remaining_min)
            min_end = min(min_end, max_end)

            best_end = ideal_end
            best_key = (-999, 9999)
            for end in range(min_end, max_end + 1):
                dist = abs(end - ideal_end)
                score = _boundary_score_at_cut(words, end)
                key = (score, -dist)
                if key > best_key:
                    best_key = key
                    best_end = end
            chunk = words[start:best_end]
            start = best_end

        segments.append({
            "segment_number": i + 1,
            "transcript": " ".join(chunk),
        })

    return segments

def _partition_full_script_exact(full_script: str, word_targets: list[int]) -> list[dict]:
    """Exact word-count split — last resort when phrase-aware splitting cannot balance."""
    words = _word_list(full_script)
    segments: list[dict] = []
    idx = 0
    for i, target in enumerate(word_targets):
        if i == len(word_targets) - 1:
            chunk = words[idx:]
        else:
            chunk = words[idx : idx + target]
            idx += target
        segments.append({
            "segment_number": i + 1,
            "transcript": " ".join(chunk),
        })
    return segments

def _rewrite_segment_word_counts(result: dict) -> list[int]:
    return [_count_words((s.get("transcript") or "").strip()) for s in (result.get("segments") or [])]

def _format_rewrite_debug(result: dict, plan: dict) -> str:
    counts = _rewrite_segment_word_counts(result)
    targets = plan.get("segment_word_targets") or []
    total = _count_words((result.get("full_script") or "").strip())
    lines = [
        f"  total words: {total} (target {plan['total_word_target']} ±{ORIGINAL_LENGTH_TOLERANCE})",
        f"  segment counts: {counts}",
        f"  segment targets: {targets}",
    ]
    for i, wc in enumerate(counts):
        tgt = targets[i] if i < len(targets) else plan["words_per_segment"]
        delta = wc - tgt
        if abs(delta) > PACE_WORD_TOLERANCE:
            lines.append(f"  segment {i + 1}: {wc} words (target {tgt}, off by {delta:+d})")
    return "\n".join(lines)

def _log_rewrite_validation_failure(
    stage: str,
    attempt: int,
    issues: list[str],
    result: dict,
    plan: dict,
    *,
    full_script: str = "",
) -> None:
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if len(payload) > REWRITE_LOG_OUTPUT_MAX_CHARS:
        payload = payload[:REWRITE_LOG_OUTPUT_MAX_CHARS] + "\n… (truncated)"
    debug = _format_rewrite_debug(
        {"full_script": full_script, "segments": result.get("segments") or []},
        plan,
    )
    log.warning(
        "Rewrite %s FAILED (attempt %d)\n"
        "Issues:\n  - %s\n"
        "%s\n"
        "Claude JSON output:\n%s",
        stage,
        attempt + 1,
        "\n  - ".join(issues),
        debug,
        payload,
    )

def _annotate_rewrite_segments(segments: list, segment_length: int) -> list:
    annotated = []
    for i, seg in enumerate(segments):
        transcript = (seg.get("transcript") or "").strip()
        start_s = i * segment_length
        end_s = (i + 1) * segment_length
        annotated.append({
            "segment_number": seg.get("segment_number", i + 1),
            "transcript": transcript,
            "start_time": _format_segment_time(start_s),
            "end_time": _format_segment_time(end_s),
            "word_count": _count_words(transcript),
        })
    return annotated

def _validate_full_script(full_script: str, plan: dict) -> list[str]:
    issues: list[str] = []
    text = (full_script or "").strip()
    if not text:
        issues.append("full_script is missing or empty")
        return issues

    total_words = _count_words(text)
    total_target = plan["total_word_target"]
    if total_words < plan["total_word_min"]:
        issues.append(
            f"full script has {total_words} words — must stay within "
            f"{total_target} ±{ORIGINAL_LENGTH_TOLERANCE} (min {plan['total_word_min']})"
        )
    if total_words > plan["total_word_max"]:
        issues.append(
            f"full script has {total_words} words — must stay within "
            f"{total_target} ±{ORIGINAL_LENGTH_TOLERANCE} (max {plan['total_word_max']})"
        )
    return issues

def _validate_partition(
    segments: list,
    full_script: str,
    plan: dict,
    word_targets: list[int] | None = None,
) -> list[str]:
    issues: list[str] = []
    full_script = (full_script or "").strip()
    num_expected = plan["num_segments"]
    targets = word_targets or _word_targets_for_script(full_script, num_expected)

    if len(segments) != num_expected:
        issues.append(f"expected {num_expected} segments but got {len(segments)}")
        return issues

    concat = " ".join((s.get("transcript") or "").strip() for s in segments)
    if _word_list(concat) != _word_list(full_script):
        issues.append("segment transcripts must use the exact same words as full_script in order")

    for i, seg in enumerate(segments):
        wc = _count_words(seg.get("transcript", ""))
        target = targets[i] if i < len(targets) else plan["words_per_segment"]
        seg_num = seg.get("segment_number", i + 1)
        if wc < target - PACE_SEVERE_UNDER_WORDS:
            issues.append(
                f"segment {seg_num} has {wc} words (target {target}) — too short"
            )
        elif wc > target + PACE_SEVERE_OVER_WORDS:
            issues.append(
                f"segment {seg_num} has {wc} words (target {target}) — too long; "
                "do not dump leftover copy into the final chunk"
            )
        elif abs(wc - target) > PACE_WORD_TOLERANCE:
            issues.append(
                f"segment {seg_num} has {wc} words (target {target}, ±{PACE_WORD_TOLERANCE})"
            )

    return issues

def _partition_move_hints(segments: list, word_targets: list[int]) -> list[str]:
    """Actionable instructions for moving words between neighboring segments."""
    hints: list[str] = []
    n = len(segments)

    for i, seg in enumerate(segments):
        wc = _count_words(seg.get("transcript", ""))
        target = word_targets[i] if i < len(word_targets) else (word_targets[-1] if word_targets else 0)
        delta = wc - target
        if abs(delta) <= PACE_WORD_TOLERANCE:
            continue

        seg_num = i + 1
        if delta > 0:
            excess = delta
            if i > 0:
                hints.append(
                    f"Segment {seg_num} is {excess} words too long ({wc} vs target {target}). "
                    f"Move ~{excess} word(s) from the START of segment {seg_num} to the END of "
                    f"segment {seg_num - 1} (shift the cut point earlier in the script)."
                )
            elif i < n - 1:
                hints.append(
                    f"Segment {seg_num} is {excess} words too long ({wc} vs target {target}). "
                    f"Move ~{excess} word(s) from the END of segment {seg_num} to the START of "
                    f"segment {seg_num + 1}."
                )
        else:
            need = -delta
            if i < n - 1:
                hints.append(
                    f"Segment {seg_num} is {need} words too short ({wc} vs target {target}). "
                    f"Move ~{need} word(s) from the START of segment {seg_num + 1} to the END of "
                    f"segment {seg_num} (shift the cut point later in the script)."
                )
            elif i > 0:
                hints.append(
                    f"Segment {seg_num} is {need} words too short ({wc} vs target {target}). "
                    f"Move ~{need} word(s) from the END of segment {seg_num - 1} to the START of "
                    f"segment {seg_num}."
                )

    return hints

def _format_failed_partition_json(segments: list) -> str:
    payload = json.dumps({"segments": segments}, ensure_ascii=False, indent=2)
    if len(payload) > 8000:
        return payload[:8000] + "\n… (truncated)"
    return payload

def _trim_script_to_max_words(full_script: str, max_words: int) -> str:
    """Trim from the end at sentence boundaries when Claude overshoots the word budget."""
    words = _word_list(full_script)
    if len(words) <= max_words:
        return full_script

    # Walk backward from max_words to find a sentence/clause boundary.
    for cut in range(max_words, max(max_words - 30, 1), -1):
        if cut <= 0:
            break
        if words[cut - 1].endswith((".", "!", "?", "…")) or (
            cut < len(words) and words[cut - 1].endswith(",")
        ):
            return " ".join(words[:cut])

    return " ".join(words[:max_words])

def _build_script_retry_feedback(issues: list[str], full_script: str, plan: dict) -> str:
    wc = _count_words(full_script)
    target = plan["total_word_target"]
    min_w = plan["total_word_min"]
    max_w = plan["total_word_max"]
    tol = ORIGINAL_LENGTH_TOLERANCE

    if wc > max_w:
        action = (
            f"TOO LONG by {wc - max_w} words. Hard ceiling is {max_w} words — you MUST delete "
            f"at least {wc - max_w} words. Tighten sentences, cut redundant lines, do not add anything new."
        )
    elif wc < min_w:
        action = (
            f"TOO SHORT by {min_w - wc} words. Add at least {min_w - wc} words without changing structure."
        )
    else:
        action = f"Aim for exactly {target} words (allowed range {min_w}–{max_w})."

    preview = full_script
    if len(preview) > 4000:
        preview = preview[:4000] + "\n… (truncated)"

    return (
        "Your previous full_script FAILED the word-count check:\n- "
        + "\n- ".join(issues)
        + f"\n\nYou returned {wc} words. Required range: {min_w}–{max_w} "
        f"(ideal {target}, ±{tol}).\n{action}\n\n"
        "Edit the script below — same message and tone, but hit the word budget. "
        "Replace existing benefits only — do not add new ones from the brand profile.\n\n"
        "YOUR PREVIOUS SCRIPT:\n"
        + preview
    )

def _build_partition_retry_feedback(
    issues: list[str],
    failed_segments: list[dict],
    word_targets: list[int],
) -> str:
    counts = [_count_words(s.get("transcript", "")) for s in failed_segments]
    target_line = ", ".join(f"seg{i + 1}~{t}" for i, t in enumerate(word_targets))
    move_hints = _partition_move_hints(failed_segments, word_targets)
    failed_json = _format_failed_partition_json(failed_segments)

    fixes_block = (
        "\n- ".join(move_hints)
        if move_hints
        else "Rebalance every segment toward its target by shifting cut points only."
    )

    return (
        "Your previous partition FAILED validation. Make MINOR fixes only — shift cut points "
        "by moving words between neighboring segments. Do NOT rewrite, add, or remove any words.\n\n"
        "WHY IT FAILED:\n- "
        + "\n- ".join(issues)
        + f"\n\nYOUR PREVIOUS OUTPUT (segment word counts: {counts}):\n"
        + failed_json
        + f"\n\nREQUIRED TARGETS: {target_line}\n\n"
        "HOW TO FIX (apply these moves):\n- "
        + fixes_block
        + "\n\nReturn corrected JSON with the exact same words in the same order — only the "
        "segment boundaries should change."
    )

def _script_rewrite_prompt(
    *,
    transcript: str,
    brand_name: str,
    product_description: str,
    target_audience: str,
    tone: str,
    plan: dict,
    original_word_count: int,
    retry_feedback: str = "",
) -> str:
    total_target = plan["total_word_target"]
    min_w = plan["total_word_min"]
    max_w = plan["total_word_max"]
    tol = ORIGINAL_LENGTH_TOLERANCE
    retry_block = f"\n== FIX REQUEST ==\n{retry_feedback}\n" if retry_feedback else ""

    return f"""You are an elite direct-response ad copywriter who specialises in short-form video ads.

GOAL — TIGHT BRAND SWAP:
Mirror the original ad's structure, hooks, and pacing. Do not rewrite from scratch or pad
with extra copy.

BENEFITS — replace in place, never pile on:
The original ad already mentions benefits. For each one, either (a) swap it for a relevant
benefit from the brand profile below, or (b) leave it as-is if it's close enough to your product.
Do NOT add benefits that weren't in the original. Do NOT try to fit every benefit listed in
the brand profile — you are replacing what the original already said, not expanding the list.

WORD BUDGET (hard limit — validated):
- Original: {original_word_count} words → yours MUST be {min_w}–{max_w} (ideal {total_target}, ±{tol})
- NEVER exceed {max_w}. When in doubt, write shorter.
- Add a phrase only if you remove one elsewhere. Net length stays flat.

== ORIGINAL TRANSCRIPT ({original_word_count} words) ==
{transcript}

== NEW BRAND ==
Name: {brand_name}
Product: {product_description}
Target audience: {target_audience}
Tone/voice: {tone}
{retry_block}
== OUTPUT (strict JSON, no markdown fences) ==
{{
  "full_script": "the complete rewritten script as one continuous piece of spoken copy",
  "word_count": {total_target}
}}

Set word_count to the exact number of words in full_script. Return ONLY the JSON object."""

def _partition_prompt(
    *,
    full_script: str,
    plan: dict,
    word_targets: list[int],
    retry_feedback: str = "",
) -> str:
    n = plan["num_segments"]
    seg_len = plan["segment_length"]
    wc = _count_words(full_script)
    target_breakdown = "\n".join(
        f"  Segment {i + 1}: ~{t} words (±{PACE_WORD_TOLERANCE})"
        for i, t in enumerate(word_targets)
    )
    retry_block = f"\n== FIX REQUEST ==\n{retry_feedback}\n" if retry_feedback else ""

    return f"""You are a video editor splitting a voiceover script into {n} timed chunks for {seg_len}-second clips.

YOUR ONLY JOB IS TO CUT — NOT REWRITE:
- Do NOT change, add, or remove ANY words
- Every word from the script must appear once, in order
- Concatenating all segment transcripts must reproduce the full script exactly

BALANCE (critical):
Split into exactly {n} consecutive chunks with these word-count targets:
{target_breakdown}
Total script: {wc} words. Distribute evenly — never dump leftover words into the final segment.
Cut at the nearest natural phrase boundary (sentence end, comma, clause break) within ±{PACE_WORD_TOLERANCE} of each target.

== FULL SCRIPT ({wc} words) ==
{full_script}
{retry_block}
== OUTPUT (strict JSON, no markdown fences) ==
{{
  "segments": [
    {{
      "segment_number": 1,
      "transcript": "exact words from full_script for this chunk only"
    }}
  ]
}}

Return exactly {n} segments. Return ONLY the JSON object."""

def _call_claude_json(client: Anthropic, prompt: str, max_tokens: int) -> dict:
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if msg.stop_reason == "max_tokens":
        raise HTTPException(500, "Claude response was cut off — please retry.")
    return _parse_json_object(msg.content[0].text)

# ---------------------------------------------------------------------------
# 1. Transcribe – upload video/audio → OpenAI Whisper
# ---------------------------------------------------------------------------

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or ".mp4")[1]
    tmp_path, upload_size = await _save_upload_to_temp(file, suffix)

    whisper_path = tmp_path
    compressed_path: str | None = None

    try:
        if upload_size > WHISPER_MAX_BYTES:
            compressed_path, audio_size = await asyncio.to_thread(
                compress_audio_for_whisper, tmp_path
            )
            whisper_path = compressed_path
            was_compressed = True
        else:
            audio_size = upload_size
            was_compressed = False

        client = _openai()
        result = await asyncio.to_thread(_whisper_transcribe, client, whisper_path)

        return _format_whisper_result(
            result,
            compressed=was_compressed,
            original_size=upload_size,
            audio_size=audio_size,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        if os.path.isfile(tmp_path):
            os.unlink(tmp_path)
        if compressed_path and os.path.isfile(compressed_path):
            os.unlink(compressed_path)

# ---------------------------------------------------------------------------
# 2. Rewrite – transcript + brand context → Claude rewrite
# ---------------------------------------------------------------------------

@app.post("/api/rewrite")
async def rewrite(request: Request):
    data = await request.json()
    transcript = data.get("transcript", "")
    brand_name = data.get("brand_name", "")
    product_description = data.get("product_description", "")
    target_audience = data.get("target_audience", "")
    tone = data.get("tone", "")
    duration = float(data.get("duration", 60))
    segment_length = int(data.get("segment_length", 15))
    target_wpm = int(data.get("target_wpm") or 150)

    if not transcript:
        raise HTTPException(400, "transcript is required")

    original_word_count = _count_words(transcript)
    plan = _build_pace_plan(duration, segment_length, target_wpm, original_word_count)
    plan["source_duration"] = round(duration, 1)
    client = _anthropic()

    log.info(
        "Rewrite start: %d-word transcript, %.1fs → %d × %ds segments, target %d ±%d words",
        original_word_count,
        duration,
        plan["num_segments"],
        segment_length,
        plan["total_word_target"],
        ORIGINAL_LENGTH_TOLERANCE,
    )

    try:
        script_attempts = 0
        full_script = ""

        # Phase 1: write full script (length + tone + brand swap)
        script_issues: list[str] = []
        for attempt in range(SCRIPT_MAX_ATTEMPTS):
            feedback = ""
            if attempt > 0 and script_issues:
                feedback = _build_script_retry_feedback(script_issues, full_script, plan)

            log.info("Rewrite phase 1 — full script (attempt %d)", attempt + 1)
            script_result = await asyncio.to_thread(
                _call_claude_json,
                client,
                _script_rewrite_prompt(
                    transcript=transcript,
                    brand_name=brand_name,
                    product_description=product_description,
                    target_audience=target_audience,
                    tone=tone,
                    plan=plan,
                    original_word_count=original_word_count,
                    retry_feedback=feedback,
                ),
                REWRITE_MAX_TOKENS,
            )
            script_attempts = attempt + 1
            full_script = (script_result.get("full_script") or "").strip()
            script_issues = _validate_full_script(full_script, plan)
            if not script_issues:
                break

            log.warning(
                "Script validation FAILED (attempt %d)\n  - %s\n  Words returned: %d",
                attempt + 1,
                "\n  - ".join(script_issues),
                _count_words(full_script),
            )

        script_trimmed = False
        if script_issues and _count_words(full_script) > plan["total_word_max"]:
            before = _count_words(full_script)
            full_script = _trim_script_to_max_words(full_script, plan["total_word_max"])
            script_issues = _validate_full_script(full_script, plan)
            if not script_issues:
                script_trimmed = True
                log.warning(
                    "Script trimmed server-side from %d to %d words (max %d)",
                    before,
                    _count_words(full_script),
                    plan["total_word_max"],
                )

        if script_issues:
            raise HTTPException(
                500,
                detail={
                    "message": "Script rewrite failed length validation after retry.",
                    "phase": "script",
                    "issues": script_issues,
                    "attempts": script_attempts,
                    "word_count": _count_words(full_script),
                },
            )

        actual_word_count = _count_words(full_script)
        word_targets = _segment_word_targets(actual_word_count, plan["num_segments"])
        plan["actual_word_count"] = actual_word_count
        plan["segment_word_targets"] = word_targets
        if script_trimmed:
            plan["script_trimmed"] = True

        log.info(
            "Rewrite phase 1 OK (%d words, %d attempt%s%s)",
            actual_word_count,
            script_attempts,
            "s" if script_attempts != 1 else "",
            ", trimmed" if script_trimmed else "",
        )

        # Phase 2: partition into balanced timed chunks
        partition_attempts = 0
        partition_source = "claude"
        segments_raw: list[dict] = []
        partition_issues: list[str] = []

        for attempt in range(PARTITION_MAX_ATTEMPTS):
            feedback = ""
            if attempt > 0 and partition_issues and segments_raw:
                feedback = _build_partition_retry_feedback(
                    partition_issues, segments_raw, word_targets
                )

            log.info("Rewrite phase 2 — partition (attempt %d)", attempt + 1)
            partition_result = await asyncio.to_thread(
                _call_claude_json,
                client,
                _partition_prompt(
                    full_script=full_script,
                    plan=plan,
                    word_targets=word_targets,
                    retry_feedback=feedback,
                ),
                PARTITION_MAX_TOKENS,
            )
            partition_attempts = attempt + 1
            segments_raw = partition_result.get("segments") or []
            partition_issues = _validate_partition(
                segments_raw, full_script, plan, word_targets
            )
            if not partition_issues:
                break

            _log_rewrite_validation_failure(
                "partition", attempt, partition_issues, partition_result, plan, full_script=full_script
            )

        if partition_issues:
            log.warning(
                "Partition failed after %d Claude attempt(s) — using server-side split",
                partition_attempts,
            )
            partition_source = "server"
            segments_raw = _partition_full_script_server(full_script, word_targets)
            partition_issues = _validate_partition(
                segments_raw, full_script, plan, word_targets
            )
            if partition_issues:
                log.warning(
                    "Phrase-aware server split still unbalanced — using exact word-count split"
                )
                partition_source = "server-exact"
                segments_raw = _partition_full_script_exact(full_script, word_targets)
                partition_issues = _validate_partition(
                    segments_raw, full_script, plan, word_targets
                )
            if partition_issues:
                merged = {"full_script": full_script, "segments": segments_raw}
                raise HTTPException(
                    500,
                    detail={
                        "message": "Could not partition script into balanced segments.",
                        "phase": "partition",
                        "issues": partition_issues,
                        "attempts": partition_attempts,
                        "debug": _format_rewrite_debug(merged, plan),
                    },
                )

        segments = _annotate_rewrite_segments(segments_raw, plan["segment_length"])

        log.info(
            "Rewrite OK — script %d attempt(s), partition via %s (%d Claude attempt(s)): %s",
            script_attempts,
            partition_source,
            partition_attempts if partition_source == "claude" else 0,
            [s["word_count"] for s in segments],
        )

        return JSONResponse({
            "full_script": full_script,
            "segments": segments,
            "meta": plan,
            "script_attempts": script_attempts,
            "partition_attempts": partition_attempts,
            "partition_source": partition_source,
        })
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        log.exception("Claude returned invalid JSON for rewrite")
        raise HTTPException(500, detail={
            "message": "Claude returned invalid JSON for rewrite — please retry.",
            "error": str(e),
        })
    except Exception as e:
        log.exception("Rewrite failed")
        raise HTTPException(500, detail={"message": f"Rewrite failed: {e}"})

# ---------------------------------------------------------------------------
# 3. Segment & Generate Prompts – rewritten script → 15-sec segments with
#    Seeddance prompts and a style-lock for visual consistency
# ---------------------------------------------------------------------------

def _build_hyper_animation_block(enabled: bool) -> str:
    if not enabled:
        return ""
    return """
== ANIMATION STYLE (user enabled: HYPER / BIZARRE) ==
The user wants scroll-stopping ANIMATED ad visuals — not polite, logical, or realistic.
Every beat should feel like a manic short-form ad cut: exaggerated motion, surreal physics,
impossible camera moves, smash cuts, morphs, zoom punches, object transformations, visual gags.
Less "character stands and talks" — more chaotic, bizarre, action-packed spectacle that hooks
attention continuously. Still map each beat to the script's meaning, but express it through WILD
visuals (swarming cartoon parasites vacuumed into a softgel, gut inflating like a balloon then
popping, skeleton surfing a wave of liquid). Push motion and surprise — go weird when the script
allows it. Default to energy and spectacle over calm explanation."""

NO_ONSCREEN_TEXT_BLOCK = """
== NO ON-SCREEN TEXT (mandatory) ==
AI video models render text poorly. NEVER ask the generator to create readable words in the frame.

In seeddance_prompt, camera, and transition_out — do NOT mention:
- On-screen text, captions, titles, subtitles, lower-thirds, or typography
- "Bold text", "put text here", "show the words", price tags with readable numbers, logos with legible lettering
- Pointing at a "link", "URL", "CTA text", or "buy now" banner

Instead describe ONLY what we SEE: characters, expressions, gestures, props, products, environments,
camera moves, lighting, and physical action. If the script mentions "link below", show the character
pointing downward or holding the product — never render the words on screen."""

def _build_segment_beat_instructions(
    enabled: bool,
    beat_min: int,
    beat_max: int,
    segment_length: int,
    *,
    hyper_animation: bool = False,
) -> tuple[str, str]:
    """Return (main_instruction_block, critical_rules_suffix) for segment prompt."""
    if not enabled:
        return (
            f"""
Each chunk is one {segment_length}-second Seeddance clip. Describe the full segment in
seeddance_prompt — action, setting, camera, movement, and mood. No fixed beat count required;
use as many or as few scene changes as the script naturally needs.""",
            "",
        )

    beat_min = max(0, min(10, beat_min))
    beat_max = max(0, min(10, beat_max))
    if beat_min > beat_max:
        beat_min, beat_max = beat_max, beat_min

    range_label = f"{beat_min}–{beat_max}" if beat_min != beat_max else str(beat_min)
    beat_rule = (
        f"- Every seeddance_prompt MUST contain {range_label} visual "
        f"{'beat' if beat_min == beat_max == 1 else 'beats'}"
        + (" and at least 1 within-segment transition." if beat_max > 1 else ".")
    )

    if beat_min == 0 and beat_max == 0:
        beat_body = f"""
VISUAL BEATS (user setting: {range_label} per segment):
A single continuous shot with no cutaways is acceptable. Describe one cohesive scene for the
full {segment_length} seconds."""
    elif beat_max <= 1:
        beat_body = f"""
VISUAL BEATS (user setting: {range_label} per segment):
Keep each clip to one main visual idea — a single scene or continuous action for the full
{segment_length} seconds. No multi-beat shot list required."""
    else:
        beat_body = f"""
IMPORTANT — ONE GENERATION PER SEGMENT, MULTIPLE SCENES INSIDE IT:
Each chunk is a single {segment_length}-second Seeddance clip. The user requires {range_label}
visual beats per segment (map beats to phrases in the script).

For every segment:
1. Break the script chunk into {range_label} visual beats (one per phrase or sentence).
2. Write a mini shot list INSIDE seeddance_prompt: Beat 1 → transition → Beat 2 → …
3. Use hard cuts, camera shifts, insert shots, gesture changes, and prop reveals BETWEEN beats.
4. Pace beats to fill the full {segment_length} seconds — no dead air, no single looping action.
5. End on transition_out so the next segment can connect cleanly.

BAD (too static): "Character stands in studio, gives thumbs up, points to link."
GOOD (multi-beat): "{_beat_good_example(hyper_animation)}" """

    return beat_body, beat_rule

def _beat_good_example(hyper_animation: bool) -> str:
    if hyper_animation:
        return (
            "Beat 1: extreme close-up — cartoon parasites swarm out of character's belly button, "
            "skeleton recoils in horror. Smash cut to Beat 2: gut inflates like a balloon, "
            "character slaps it and parasites fly off in all directions. Cut to Beat 3: softgel "
            "drops from sky in slow-mo, explodes into golden shockwave that vaporizes parasites. "
            "Cut to Beat 4: character spins triumphantly, ribs glow neon green. Cut to Beat 5: "
            "zoom-punch in — character points emphatically at product bottle with both hands, freeze frame."
        )
    return (
        "Beat 1: close-up miming disgust at bloating. Cut to Beat 2: medium shot gesturing at gut. "
        "Cut to Beat 3: product softgel hero insert. Cut to Beat 4: relieved smile. "
        "Cut to Beat 5: character points down toward product on table, direct eye contact, hold for end frame."
    )

def _build_segment_seeddance_example(
    enabled: bool,
    beat_min: int,
    beat_max: int,
    segment_length: int,
    *,
    hyper_animation: bool = False,
) -> str:
    if not enabled:
        base = (
            "Self-contained Seeddance prompt. MUST start with the full character description "
            "from style_lock. Describe action, setting, camera angle, movement, and mood. "
            "End with style keywords."
        )
        return base + (
            " Each beat/scene: bizarre, surreal, action-packed, scroll-stopping animation."
            if hyper_animation
            else ""
        )
    beat_max = max(0, min(10, beat_max))
    if beat_max <= 1:
        base = (
            f"Start with the full character description from style_lock. One cohesive scene for "
            f"the full {segment_length}s — action, setting, framing, mood. End with style keywords."
        )
        return base + (
            " Exaggerated, surreal, high-energy motion — not a static talking shot."
            if hyper_animation
            else ""
        )
    base = (
        f"Start with the full character description from style_lock. Then a BEAT-BY-BEAT shot list "
        f"for the full {segment_length}s: label each beat (Beat 1, Beat 2, …), describe "
        "action/setting/framing/mood, and insert Cut to / Smash cut / Push-in / Insert between "
        "beats. Each beat should map to a phrase in the script. End with style keywords."
    )
    return base + (
        " Each beat: bizarre, surreal, action-packed — less logical, more visually extreme."
        if hyper_animation
        else ""
    )

@app.post("/api/segment")
async def segment(request: Request):
    data = await request.json()
    script_chunks = data.get("script_chunks", [])
    effective_duration = float(data.get("effective_duration", 60))
    brand_name = data.get("brand_name", "")
    product_description = data.get("product_description", "")
    segment_length = int(data.get("segment_length", 15))
    tone = data.get("tone", "")
    visual_description = data.get("visual_description", "")
    reference_images = data.get("reference_images", [])
    target_wpm = data.get("target_wpm")
    beat_requirement_enabled = bool(data.get("beat_requirement_enabled", True))
    beat_min = int(data.get("beat_min", 2))
    beat_max = int(data.get("beat_max", 5))
    hyper_animation_enabled = bool(data.get("hyper_animation_enabled", False))

    if not script_chunks:
        raise HTTPException(400, "script_chunks is required")

    if len(reference_images) > 5:
        raise HTTPException(400, "Maximum 5 reference images allowed.")

    client = _anthropic()

    tone_section = ""
    if tone:
        tone_section = f"""
== TONE / VOICE (this ad) ==
{tone}

Reflect this tone in style_lock.delivery and in every seeddance_prompt through on-screen
energy, facial expressions, body language, pacing, and mood.
"""

    visual_section = ""
    if visual_description:
        visual_section += f"""
== ORIGINAL AD — VISUAL DESCRIPTION ==
{visual_description}
"""
    if reference_images:
        visual_section += """
Reference screenshots of the original ad are attached below this text. Study them for
setting, framing, subject appearance, wardrobe, props, lighting, color grade, camera
style, and overall aesthetic. Your style_lock and segment prompts should mirror this
visual language (adapted for the new brand/product), not invent something unrelated.
"""

    chunks_block = ""
    for ch in script_chunks:
        num = ch.get("segment_number", "?")
        start = ch.get("start_time", "")
        end = ch.get("end_time", "")
        text = (ch.get("transcript") or "").strip()
        chunks_block += f"""
--- SEGMENT {num} ({start} → {end}) ---
SPOKEN SCRIPT (fixed — copy verbatim into this segment's transcript field):
{text}
"""

    pace_note = ""
    if target_wpm:
        pace_note = f"\nDelivery pace target: ~{target_wpm} WPM (scripts are already time-boxed).\n"

    beat_instructions, beat_critical_rule = _build_segment_beat_instructions(
        beat_requirement_enabled,
        beat_min,
        beat_max,
        segment_length,
        hyper_animation=hyper_animation_enabled,
    )
    seeddance_example = _build_segment_seeddance_example(
        beat_requirement_enabled,
        beat_min,
        beat_max,
        segment_length,
        hyper_animation=hyper_animation_enabled,
    )
    hyper_animation_block = _build_hyper_animation_block(hyper_animation_enabled)
    hyper_critical_rule = (
        "- HYPER mode on: every beat must be bizarre, action-packed, and visually extreme — "
        "avoid bland talking-head or single-pose shots.\n"
        if hyper_animation_enabled
        else ""
    )
    camera_example = (
        "Overall camera language for the segment AND any per-beat shifts (e.g. Beat 1 MCU → Beat 2 wide → Beat 3 product macro)"
        if beat_requirement_enabled and beat_max > 1
        else "Shot type and camera motion for this clip (e.g. medium close-up, slow push-in)"
    )
    density_rule = (
        "- Match visual density to script density — more sentences = more beats.\n"
        if beat_requirement_enabled and beat_max > 1
        else ""
    )
    length_rule = (
        "- seeddance_prompt length: roughly 120–220 words (longer when the script chunk is dense).\n"
        if beat_requirement_enabled and beat_max > 1
        else "- Keep each seeddance_prompt concise (roughly 80–150 words).\n"
    )

    try:
        prompt_text = f"""You are an expert AI-video production director who specialises in
creating shot-by-shot breakdowns for AI video generators like Seeddance on Higgsfield.

TASK: The ad script is ALREADY split into fixed {segment_length}-second chunks below
(~{int(effective_duration)}s total). Do NOT re-split or rewrite any spoken lines.
Generate a style_lock plus a Seeddance prompt, camera note, and transition_out for EACH chunk.
{beat_instructions}
{hyper_animation_block}
{NO_ONSCREEN_TEXT_BLOCK}

All spoken scripts below are ENGLISH voiceover only — never translate dialogue. The VO is added
separately; the video itself must contain NO readable on-screen text.

== BRAND CONTEXT ==
Brand: {brand_name}
Product: {product_description}
{pace_note}{tone_section}{visual_section}
== FIXED SCRIPT CHUNKS ({len(script_chunks)} segments) ==
{chunks_block}

== OUTPUT FORMAT (strict JSON, no markdown fences) ==
{{
  "style_lock": {{
    "character": "full physical description of the on-screen person that MUST appear verbatim in every segment prompt",
    "wardrobe": "clothing description, consistent across all segments",
    "lighting": "lighting setup used in every segment",
    "color_grade": "color palette / grade",
    "visual_style": "overall aesthetic (e.g. cinematic, UGC, studio, lifestyle)",
    "delivery": "on-camera tone, energy, and delivery style"
  }},
  "segments": [
    {{
      "segment_number": 1,
      "start_time": "{script_chunks[0].get('start_time', '0:00')}",
      "end_time": "{script_chunks[0].get('end_time', f'0:{segment_length:02d}')}",
      "transcript": "exact script chunk provided above — verbatim",
      "seeddance_prompt": "{seeddance_example}",
      "camera": "{camera_example}",
      "transition_out": "Exact last-frame composition for handoff to the next segment — pose, prop placement, background"
    }}
  ]
}}

CRITICAL RULES:
- Return exactly {len(script_chunks)} segments, in order, matching the script chunks above.
- Each transcript field MUST match the provided script chunk verbatim — do not edit copy.
- Copy the character description from style_lock into every seeddance_prompt.
- NEVER instruct on-screen text, captions, or typography in seeddance_prompt, camera, or transition_out.
- Describe scenes, characters, actions, props, and camera only — no readable words in the frame.
{beat_critical_rule}
{hyper_critical_rule}{density_rule}- Camera angles may change within a segment; lighting/color tags stay consistent with style_lock.
{length_rule}
Return ONLY the raw JSON object."""

        content_blocks: list = [{"type": "text", "text": prompt_text}]
        for img in reference_images:
            media_type = img.get("media_type", "image/jpeg")
            data_b64 = img.get("data", "")
            if not data_b64:
                continue
            if media_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
                media_type = "image/jpeg"
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data_b64,
                },
            })

        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=SEGMENT_MAX_TOKENS,
            messages=[{"role": "user", "content": content_blocks}],
        )

        if msg.stop_reason == "max_tokens":
            raise HTTPException(
                500,
                "Segment response was cut off — try again or use fewer reference images.",
            )

        try:
            result = _parse_json_object(msg.content[0].text)
        except json.JSONDecodeError:
            raise HTTPException(500, "Claude returned invalid JSON – please retry.")

        # Enforce exact transcripts from user-edited chunks
        for i, ch in enumerate(script_chunks):
            if i < len(result.get("segments") or []):
                result["segments"][i]["transcript"] = (ch.get("transcript") or "").strip()
                result["segments"][i]["segment_number"] = ch.get("segment_number", i + 1)
                result["segments"][i]["start_time"] = ch.get("start_time", result["segments"][i].get("start_time"))
                result["segments"][i]["end_time"] = ch.get("end_time", result["segments"][i].get("end_time"))

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Segment generation failed: {e}")

# ---------------------------------------------------------------------------
# 4. Generate SRT – word-level or segment-level timestamps → viral caption SRT
# ---------------------------------------------------------------------------

@app.post("/api/generate-srt")
async def generate_srt(request: Request):
    data = await request.json()
    words = data.get("words", [])
    segments = data.get("segments", [])
    style = data.get("style", "viral")  # "viral" (2-3 words) or "standard"

    srt_lines: list[str] = []
    counter = 1

    if words and style == "viral":
        # Viral-style: 2-3 word bursts from word-level timestamps
        chunk_size = 3
        for i in range(0, len(words), chunk_size):
            chunk = words[i : i + chunk_size]
            start_s = chunk[0]["start"]
            end_s = chunk[-1]["end"]
            text = " ".join(w["word"] for w in chunk).upper()
            srt_lines.append(f"{counter}")
            srt_lines.append(f"{_srt_ts(start_s)} --> {_srt_ts(end_s)}")
            srt_lines.append(text)
            srt_lines.append("")
            counter += 1
    elif segments:
        # Standard-style from segments
        for seg in segments:
            srt_lines.append(f"{counter}")
            srt_lines.append(
                f"{_srt_ts_from_label(seg['start_time'])} --> {_srt_ts_from_label(seg['end_time'])}"
            )
            srt_lines.append(seg.get("transcript", "").upper())
            srt_lines.append("")
            counter += 1

    return JSONResponse({"srt": "\n".join(srt_lines)})


def _srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _srt_ts_from_label(label: str) -> str:
    """Convert '0:15' or '1:05' to SRT timestamp."""
    parts = label.split(":")
    if len(parts) == 2:
        mins, secs = int(parts[0]), int(parts[1])
    else:
        mins, secs = 0, int(parts[0])
    total = mins * 60 + secs
    return _srt_ts(float(total))


# ---------------------------------------------------------------------------
# Static files (must be mounted LAST so API routes take priority)
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")
