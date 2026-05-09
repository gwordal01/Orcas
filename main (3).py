import sys
import os
import platform
import subprocess
import threading
import queue
import time
import re
import argparse

IS_WINDOWS = platform.system() == "Windows"

if not IS_WINDOWS:
    try:
        import pty
        import select
        HAS_PTY = True
    except ImportError:
        HAS_PTY = False
else:
    HAS_PTY = False

class Color:
    RESET = "\033[0m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"


def color_text(text, color):
    return f"{color}{text}{Color.RESET}"


MODEL = "qwen2.5:3b"


def check_ollama():
    """Verify ollama is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            return False, "ollama not running. Run: ollama serve"
        if "qwen2.5" not in result.stdout.lower():
            return False, "Model missing. Run: ollama pull qwen2.5:3b"
        return True, "ok"
    except FileNotFoundError:
        return False, "ollama not installed. Get it from https://ollama.com"
    except Exception as e:
        return False, str(e)


def ask_llm_for_fix(error_text):
    """
    Send an error to the LLM and get a fix suggestion.
    This is the 'recovery handler' in our watchdog analogy.
    """
    prompt = build_llm_prompt(error_text)

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        suggestion = format_llm_suggestion(result.stdout.strip())
        return suggestion
    except subprocess.TimeoutExpired:
        return "[LLM timeout - error too complex]"
    except Exception as e:
        return f"[LLM error: {e}]"


def format_llm_suggestion(raw_text):
    """Force a short, stable display format for inline suggestions."""
    if not raw_text:
        return "Fix: Re-run with full traceback and inspect the first failing line.\nCommand: N/A"

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    fix_line = next((line for line in lines if line.lower().startswith("fix:")), "")
    command_line = next((line for line in lines if line.lower().startswith("command:")), "")

    if not fix_line:
        fix_line = f"Fix: {lines[0]}"
    if not command_line:
        command_line = "Command: N/A"

    return f"{fix_line}\n{command_line}"


def build_llm_prompt(error_text):
    """Build the prompt sent to the LLM."""
    return f"""You are TerminalBrain, an expert CLI debugger.
Task: Provide ONE concrete fix for the error below.
Rules:
- Output exactly 2 lines:
  Fix: <single actionable fix>
  Command: <exact command or N/A>
- No greetings and no extra explanation.
- Prefer highest-probability fix.
- If import/module error: include install command.
- If file/path error: include path-check command.
- If syntax/type error: include the exact fix code.
Error:
{error_text}
"""

ERROR_PATTERNS = [
    # Python errors
    r"Traceback \(most recent call last\)",
    r"Error:",
    r"Exception:",
    r"ModuleNotFoundError",
    r"ImportError",
    r"NameError",
    r"SyntaxError",
    r"TypeError",
    r"ValueError",
    r"KeyError",
    r"AttributeError",
    r"FileNotFoundError",
    r"IndentationError",
    r"ZeroDivisionError",
    r"RuntimeError",
    r"AssertionError",

    # Shell errors
    r"command not found",
    r"No such file or directory",
    r"permission denied",
    r"cannot access",
    r"is not recognized as an internal or external command",
    r"cannot find the path specified",
    r"access is denied",
    r"unexpected token",
    r"syntax error near unexpected token",
    r"segmentation fault",
    r"core dumped",

    # Generic
    r"FAILED",
    r"FATAL",
    r"panic:",
    r"unhandled exception",
    r"ERROR \[",

]

ERROR_REGEX = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)


def is_error_line(line):
    """Check if a line of stderr looks like a real error."""
    if not line or not line.strip():
        return False
    # Filter common non-fatal noise so we avoid unnecessary LLM calls.
    if re.search(r"\b(warning|deprecated|notice|info)\b", line, re.IGNORECASE):
        return False
    return bool(ERROR_REGEX.search(line))



fix_cache = {}


def _extract_error_signature(error_text):
    """Build a stable key for semantically similar error blocks."""
    text = (error_text or "").strip()
    if not text:
        return ""

    keyword_match = ERROR_REGEX.search(text)
    keyword = keyword_match.group(0).lower() if keyword_match else "generic-error"

    quoted = re.findall(r"[\"']([^\"']+)[\"']", text)
    token = quoted[0].lower() if quoted else ""
    if not token:
        token_match = re.search(
            r"(?:module|package|file|path)\s+([A-Za-z0-9_./\\-]+)",
            text,
            re.IGNORECASE,
        )
        if token_match:
            token = token_match.group(1).lower()

    normalized = re.sub(r"0x[0-9a-fA-F]+|\d+", "#", text.lower())
    snippet = " ".join(normalized.split())[:80]
    return f"{keyword}|{token}|{snippet}"


def get_cached_fix(error_text):
    """Look up a cached fix for similar errors."""
    key = _extract_error_signature(error_text)
    return fix_cache.get(key)


def cache_fix(error_text, fix):
    """Store a fix for future identical errors."""
    key = _extract_error_signature(error_text)
    if key:
        fix_cache[key] = fix



def reader_thread(stream, output_queue, stream_name):
    """Read from a stream and push lines to a queue."""
    try:
        for line in iter(stream.readline, ''):
            if not line:
                break
            output_queue.put((stream_name, line))
    except Exception as e:
        output_queue.put(("error", f"[reader thread error: {e}]\n"))
    finally:
        try:
            stream.close()
        except Exception:
            pass



def run_with_brain(command):
    """
    Run a command, capture stdout/stderr live, analyze errors.
    """
    print(color_text("\n+- TerminalBrain wrapping: ", Color.CYAN), end="")
    print(color_text(" ".join(command), Color.BOLD))
    print(color_text("| stdout = white | stderr = red | brain = cyan", Color.DIM))
    print(color_text("+-" + "-" * 50, Color.CYAN))
    print()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )
    except FileNotFoundError:
        print(color_text(f"Command not found: {command[0]}", Color.RED))
        suggestion = ask_llm_for_fix(f"command not found: {command[0]}")
        print(color_text(f"\n[Brain] {suggestion}\n", Color.CYAN))
        return
    except Exception as e:
        print(color_text(f"Failed to start process: {e}", Color.RED))
        return

    output_queue = queue.Queue()

    stdout_thread = threading.Thread(
        target=reader_thread,
        args=(process.stdout, output_queue, "stdout"),
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=reader_thread,
        args=(process.stderr, output_queue, "stderr"),
        daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    error_buffer = []
    in_error_block = False
    error_count = 0
    last_error_time = 0.0
    llm_calls = 0
    cache_hits = 0

    # Process output until command finishes
    while True:
        try:
            stream_name, line = output_queue.get(timeout=0.1)
        except queue.Empty:
            # Flush a pending error block quickly even without stdout.
            if in_error_block and error_buffer and (time.time() - last_error_time) > 0.35:
                llm_used, cached_used = handle_error_block(error_buffer)
                llm_calls += llm_used
                cache_hits += cached_used
                error_buffer = []
                in_error_block = False

            # Check if process is still alive
            if process.poll() is not None and output_queue.empty():
                break
            continue

        if stream_name == "stdout":
            # Print stdout in white
            print(color_text(line.rstrip(), Color.WHITE))

            # If we were collecting an error, the error block ended
            if in_error_block and error_buffer:
                llm_used, cached_used = handle_error_block(error_buffer)
                llm_calls += llm_used
                cache_hits += cached_used
                error_buffer = []
                in_error_block = False

        elif stream_name == "stderr":
            # Print stderr in red
            print(color_text(line.rstrip(), Color.RED))

            # Check if this looks like an error
            if is_error_line(line):
                in_error_block = True
                error_count += 1
                last_error_time = time.time()

            if in_error_block:
                error_buffer.append(line)

                # Send error to brain after we've collected a few lines
                # (or after timeout - handled by the next iteration)

    # Wait for any final output
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)

    # Process any remaining error buffer
    if error_buffer:
        llm_used, cached_used = handle_error_block(error_buffer)
        llm_calls += llm_used
        cache_hits += cached_used

    # Final summary
    print()
    print(color_text("-" * 52, Color.DIM))
    exit_code = process.returncode
    status_color = Color.GREEN if exit_code == 0 else Color.RED
    print(color_text(f"Exit code: {exit_code}", status_color))
    print(color_text(f"Errors detected: {error_count}", Color.DIM))
    print(color_text(f"LLM calls: {llm_calls} | cache hits: {cache_hits}", Color.DIM))


def handle_error_block(lines):
    """Send a collected error block to the LLM for analysis."""
    error_text = "".join(lines).strip()
    if not error_text:
        return 0, 0

    # Check cache first
    cached = get_cached_fix(error_text)
    if cached:
        print()
        print(color_text(f"[Brain][cached] {cached}", Color.CYAN))
        print()
        return 0, 1

    # Show "thinking" indicator
    print()
    print(color_text("[Brain] analyzing...", Color.CYAN), end="", flush=True)

    fix = ask_llm_for_fix(error_text)

    # Clear the "analyzing" line
    print("\r" + " " * 30 + "\r", end="")

    print(color_text(f"[Brain] {fix}", Color.CYAN))
    print()

    # Cache for next time
    cache_fix(error_text, fix)
    return 1, 0


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TerminalBrain - wrap a command and analyze its errors with a local LLM"
    )
    raw_argv = sys.argv[1:]
    if len(raw_argv) == 1 and raw_argv[0] in ("-h", "--help"):
        parser.print_help()
        return

    command = list(raw_argv)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command to wrap")
    # Convenience: allow `hardcore10.py -c "..."` by assuming current Python.
    if command and command[0].startswith("-"):
        command = [sys.executable] + command

    # Check ollama
    ok, msg = check_ollama()
    if not ok:
        print(color_text(f"ERROR: {msg}", Color.RED))
        sys.exit(1)

    print(color_text("[OK] ollama ready", Color.GREEN))
    print(color_text(f"  Model: {MODEL}", Color.DIM))
    print(color_text(f"  Platform: {platform.system()}", Color.DIM))
    print(color_text(f"  pty available: {HAS_PTY}", Color.DIM))

    # Run the wrapped command
    run_with_brain(command)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print()
        print("TerminalBrain - wrap a command and get AI fix suggestions")
        print()
        print("Usage:")
        print("  python hardcore10.py <command> [args...]")
        print()
        print("Try one of these to test:")
        print("  python hardcore10.py python -c \"import nonexistent_module\"")
        print("  python hardcore10.py ls /nonexistent_directory")
        print("  python hardcore10.py python -c \"print(undefined_var)\"")
        print("  python hardcore10.py -c \"import idkwhy\"")
        print()
        sys.exit(0)

    main()


# ============================================================
# SAMPLE TESTS
# ============================================================
# These are common error cases where the LLM usually gives
# useful, concrete fixes.
#
# 1) Missing Python module (import error)
# py -3.11 hardcore10.py python -c "import nonexistent_module"
#
# 2) Undefined variable (name error)
# py -3.11 hardcore10.py python -c "print(undefined_var)"
#
# 3) Missing file (file/path error)
# py -3.11 hardcore10.py python -c "open('missing_file_12345.txt')"
#
# 4) Invalid pip package (package resolution error)
# py -3.11 hardcore10.py pip install package_that_should_not_exist_abcxyz
#
# 5) Invalid command/path in shell (command/path not found)
# py -3.11 hardcore10.py cmd /c dir C:\\definitely_missing_folder_12345
