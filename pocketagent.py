"""
EdgeAgent  —  Windows Edition
==============================
A CLI agent powered by a local 3B LLM via ollama.
Tools: read files, list directories, run safe shell commands, answer system questions.

Prerequisites:
    1. Install ollama:        https://ollama.com/download
    2. Start ollama server:   ollama serve          (in a separate terminal)
    3. Pull the model:        ollama pull qwen2.5:3b
    4. Install Python deps:   pip install ollama

Usage:
    python edgeagent.py
    python edgeagent.py --model llama3.2:3b
    python edgeagent.py --model qwen2.5:3b
"""

import argparse
import json
import os
import subprocess
import sys
import time
import platform
import shutil
from pathlib import Path

try:
    import ollama
except ImportError:
    print("\n  ERROR: ollama Python SDK not installed.")
    print("  Run:  pip install ollama\n")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen2.5:3b"

# Shell commands that are explicitly allowed (safety allowlist)
SAFE_COMMANDS = {
    "dir", "ls", "echo", "whoami", "hostname", "ipconfig", "ping",
    "systeminfo", "tasklist", "python", "python3", "pip", "where",
    "type", "cat", "more", "find", "tree", "date", "time", "ver",
    "wmic", "netstat", "curl", "powershell",
}

# Max output length returned to the model (avoid flooding context)
MAX_OUTPUT_CHARS = 3000

# ── Colours ───────────────────────────────────────────────────────────────────

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def c(text, color): return f"{color}{text}{RESET}"

# ── Tools ─────────────────────────────────────────────────────────────────────

def tool_read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"ERROR: File not found: {path}"
        if not p.is_file():
            return f"ERROR: Path is not a file: {path}"
        size = p.stat().st_size
        if size > 500_000:
            return f"ERROR: File too large ({size} bytes). Max 500KB."
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(text)} chars total]"
        return text
    except PermissionError:
        return f"ERROR: Permission denied: {path}"
    except Exception as e:
        return f"ERROR: {e}"


def tool_list_dir(path: str = ".") -> str:
    """List contents of a directory."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"ERROR: Directory not found: {path}"
        if not p.is_dir():
            return f"ERROR: Path is not a directory: {path}"
        items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        lines = [f"Contents of: {p}\n"]
        for item in items:
            if item.is_dir():
                lines.append(f"  [DIR]  {item.name}/")
            else:
                size = item.stat().st_size
                size_str = f"{size:>10,} bytes"
                lines.append(f"  [FILE] {item.name:<40} {size_str}")
        lines.append(f"\n{len(items)} items total.")
        return "\n".join(lines)
    except PermissionError:
        return f"ERROR: Permission denied: {path}"
    except Exception as e:
        return f"ERROR: {e}"


def tool_run_command(command: str) -> str:
    """Run a safe shell command and return output."""
    # Extract base command (first word)
    base = command.strip().split()[0].lower().rstrip(".exe")
    if base not in SAFE_COMMANDS:
        return (
            f"ERROR: Command '{base}' is not in the safe allowlist.\n"
            f"Allowed: {', '.join(sorted(SAFE_COMMANDS))}"
        )
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
        out = result.stdout + result.stderr
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n... [truncated]"
        return out.strip() if out.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out (15s limit)."
    except Exception as e:
        return f"ERROR: {e}"


def tool_system_info() -> str:
    """Return basic system information."""
    info = {
        "os":           platform.system(),
        "os_version":   platform.version(),
        "machine":      platform.machine(),
        "processor":    platform.processor(),
        "python":       sys.version.split()[0],
        "cwd":          str(Path.cwd()),
        "home":         str(Path.home()),
        "ollama":       shutil.which("ollama") or "not found in PATH",
    }
    lines = ["System Information:"]
    for k, v in info.items():
        lines.append(f"  {k:<14}: {v}")
    return "\n".join(lines)


# Tool registry — maps tool name → (function, description, args)
TOOLS = {
    "read_file": {
        "fn":   tool_read_file,
        "desc": "Read the contents of a file. Args: path (string)",
        "args": ["path"],
    },
    "list_dir": {
        "fn":   tool_list_dir,
        "desc": "List files and directories. Args: path (string, default '.')",
        "args": ["path"],
    },
    "run_command": {
        "fn":   tool_run_command,
        "desc": f"Run a safe shell command. Allowed bases: {', '.join(sorted(SAFE_COMMANDS))}. Args: command (string)",
        "args": ["command"],
    },
    "system_info": {
        "fn":   tool_system_info,
        "desc": "Get OS, Python, processor, and path information. No args needed.",
        "args": [],
    },
}

# ── System prompt ─────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    tool_docs = "\n".join(
        f"  - {name}: {info['desc']}"
        for name, info in TOOLS.items()
    )
    return f"""You are EdgeAgent, a helpful CLI assistant running locally on the user's laptop.
You have access to the following tools:

{tool_docs}

When you need to use a tool, respond with ONLY a JSON object in this exact format:
{{"tool": "<tool_name>", "args": {{"<arg_name>": "<value>"}}}}

Rules:
- Use a tool when the user asks you to read a file, list files, run a command, or get system info.
- After receiving tool output, give a clear, concise answer based on the result.
- If no tool is needed, just answer directly.
- Never make up file contents or command output — always use a tool to get real data.
- For list_dir with no path specified, use {{"path": "."}}
- Keep answers brief and focused.
- You are running on: {platform.system()} {platform.machine()}
"""

# ── Tool call parser ──────────────────────────────────────────────────────────

def parse_tool_call(text: str):
    """
    Try to extract a JSON tool call from the model's response.
    Returns (tool_name, args_dict) or (None, None).
    """
    text = text.strip()

    # Try to find JSON block (sometimes model wraps in ```json ... ```)
    for start_marker in ["```json", "```", ""]:
        if start_marker:
            if start_marker in text:
                start = text.index(start_marker) + len(start_marker)
                end   = text.find("```", start)
                if end == -1:
                    end = len(text)
                text = text[start:end].strip()
                break

    # Find first { ... }
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start == -1 or brace_end == -1:
        return None, None

    json_str = text[brace_start : brace_end + 1]
    try:
        data = json.loads(json_str)
        tool = data.get("tool")
        args = data.get("args", {})
        if tool and tool in TOOLS:
            return tool, args
    except json.JSONDecodeError:
        pass

    return None, None

# ── Streaming response with token/s ──────────────────────────────────────────

def stream_response(model: str, messages: list) -> tuple[str, float]:
    """
    Stream a response from ollama.
    Returns (full_text, tokens_per_second).
    """
    full_text  = ""
    token_count = 0
    start_time  = None

    print(f"\n  {c('EdgeAgent', CYAN)}{c(':', DIM)} ", end="", flush=True)

    try:
        stream = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                if start_time is None:
                    start_time = time.time()
                print(content, end="", flush=True)
                full_text   += content
                token_count += len(content.split())   # approximate

        elapsed = time.time() - (start_time or time.time())
        tps     = token_count / elapsed if elapsed > 0 else 0.0

    except ollama.ResponseError as e:
        print(f"\n  {c('ERROR', RED)}: {e}")
        full_text = ""
        tps = 0.0
    except Exception as e:
        print(f"\n  {c('ERROR', RED)}: {e}")
        full_text = ""
        tps = 0.0

    print()   # newline after streamed response
    return full_text, tps

# ── Banner & UI ───────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def banner(model: str):
    print(f"{BOLD}{CYAN}")
    print("  ███████╗██████╗  ██████╗ ███████╗")
    print("  ██╔════╝██╔══██╗██╔════╝ ██╔════╝")
    print("  █████╗  ██║  ██║██║  ███╗█████╗  ")
    print("  ██╔══╝  ██║  ██║██║   ██║██╔══╝  ")
    print("  ███████╗██████╔╝╚██████╔╝███████╗")
    print("  ╚══════╝╚═════╝  ╚═════╝ ╚══════╝")
    print(f"  {DIM}A G E N T{RESET}{BOLD}{CYAN}          Day 07 — BUILDCORED ORCAS{RESET}")
    print()
    print(f"  {BOLD}Model   :{RESET} {c(model, YELLOW)}")
    print(f"  {BOLD}Platform:{RESET} {platform.system()} {platform.machine()}")
    print(f"  {BOLD}Tools   :{RESET} {', '.join(TOOLS.keys())}")
    print()
    print(f"  {DIM}Type your message and press Enter. Type 'exit' to quit.{RESET}")
    print(f"  {DIM}Try: 'list my current directory' or 'what OS am I on?'{RESET}")
    print(f"  {'─' * 54}{RESET}")
    print()

# ── Main agent loop ───────────────────────────────────────────────────────────

def run_agent(model: str):
    clear()
    banner(model)

    # Verify ollama is reachable and model exists
    print(f"  {DIM}Connecting to ollama...{RESET}", end="", flush=True)
    try:
        models = ollama.list()
        available = [m["model"] for m in models.get("models", [])]
        # Strip tag for loose match
        base = model.split(":")[0]
        if not any(base in m for m in available):
            print(f"\n\n  {c('WARNING', YELLOW)}: Model '{model}' not found locally.")
            print(f"  Run this first:  {c(f'ollama pull {model}', GREEN)}\n")
        else:
            print(f" {c('OK', GREEN)}")
    except Exception as e:
        print(f"\n\n  {c('ERROR', RED)}: Cannot reach ollama server.")
        print(f"  Make sure ollama is running:  {c('ollama serve', GREEN)}")
        print(f"  Details: {e}\n")
        sys.exit(1)

    print()

    messages = [{"role": "system", "content": build_system_prompt()}]
    total_tps_samples = []

    while True:
        try:
            user_input = input(f"  {c('You', GREEN)}{c(':', DIM)} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {c('Goodbye!', CYAN)}\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print(f"\n  {c('Goodbye!', CYAN)}\n")
            break

        # Special commands
        if user_input.lower() == "clear":
            clear()
            banner(model)
            messages = [{"role": "system", "content": build_system_prompt()}]
            continue

        if user_input.lower() == "history":
            print(f"\n  {c('Conversation history:', CYAN)} {len(messages) - 1} messages")
            continue

        messages.append({"role": "user", "content": user_input})

        # ── Get model response ──
        response, tps = stream_response(model, messages)
        if not response:
            continue

        total_tps_samples.append(tps)

        # ── Check for tool call ──
        tool_name, tool_args = parse_tool_call(response)

        if tool_name:
            # Execute the tool
            tool_fn = TOOLS[tool_name]["fn"]
            arg_values = [tool_args.get(a, "") for a in TOOLS[tool_name]["args"]]

            print(f"\n  {c('→ Tool', YELLOW)}: {c(tool_name, BOLD)} ", end="")
            if tool_args:
                print(f"({', '.join(f'{k}={repr(v)}' for k, v in tool_args.items())})")
            else:
                print()

            t0         = time.time()
            tool_output = tool_fn(*arg_values)
            t1         = time.time()

            print(f"  {c('← Result', YELLOW)} {c(f'({t1-t0:.2f}s):', DIM)}")
            # Print first few lines of tool output
            lines = tool_output.splitlines()
            preview = lines[:12]
            for line in preview:
                print(f"    {c(line, DIM)}")
            if len(lines) > 12:
                print(f"    {c(f'... ({len(lines) - 12} more lines)', DIM)}")
            print()

            # Feed tool result back to model for a natural-language answer
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Tool '{tool_name}' returned:\n{tool_output}\n\nNow answer the user's original question based on this output."
            })

            followup, tps2 = stream_response(model, messages)
            total_tps_samples.append(tps2)
            messages.append({"role": "assistant", "content": followup})

        else:
            messages.append({"role": "assistant", "content": response})

        # ── Tokens/sec display ──
        avg_tps = sum(total_tps_samples) / len(total_tps_samples) if total_tps_samples else 0
        print(f"\n  {c(f'~{tps:.1f} tok/s  |  avg {avg_tps:.1f} tok/s  |  {len(messages)-1} messages in context', DIM)}")
        print(f"  {'─' * 54}")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EdgeAgent — local LLM CLI agent powered by ollama"
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    run_agent(args.model)


if __name__ == "__main__":
    main()