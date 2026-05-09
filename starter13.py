"""
BUILDCORED ORCAS - Day 13: DailyDebrief
Collect your day's activity and get an AI summary.

Hardware concept: Flight Data Recorder
Collect all streams -> compress -> report.

Run: python starter13.py
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    from git import InvalidGitRepositoryError, NoSuchPathError, Repo
    GITPYTHON_AVAILABLE = True
except Exception:
    Repo = None
    InvalidGitRepositoryError = NoSuchPathError = Exception
    GITPYTHON_AVAILABLE = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
except ImportError:
    print("pip install rich")
    sys.exit(1)

MODEL = "qwen2.5:3b"
ACTIVE_REPO_PATH = Path.cwd()


def check_ollama():
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        if "qwen2.5" not in r.stdout.lower():
            console.print("[red]Run: ollama pull qwen2.5:3b[/red]")
            sys.exit(1)
    except Exception:
        console.print("[red]ollama not found[/red]")
        sys.exit(1)


check_ollama()

# ====== DATA SOURCES ======


def ensure_test_repo():
    """
    If current dir is not a git repo, create a tiny local test repo
    so the debrief still has commit data.
    """
    cwd = Path.cwd()
    if not GITPYTHON_AVAILABLE:
        return None

    try:
        Repo(cwd, search_parent_directories=True)
        return None
    except (InvalidGitRepositoryError, NoSuchPathError):
        sandbox = cwd / "dailydebrief_test_repo"
        sandbox.mkdir(exist_ok=True)
        repo = Repo.init(sandbox)

        note = sandbox / "README.md"
        note.write_text(
            "# DailyDebrief Test Repo\n\nAuto-created because no git repo was found.\n",
            encoding="utf-8",
        )
        repo.index.add(["README.md"])
        repo.index.commit("chore: bootstrap daily debrief test repo")
        return sandbox


def get_git_commits(hours=24):
    """Last N hours of git commits from current repo."""
    if not GITPYTHON_AVAILABLE:
        try:
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            r = subprocess.run(
                ["git", "log", f"--since={since}", "--pretty=format:%h %s"],
                cwd=str(ACTIVE_REPO_PATH),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            lines = r.stdout.strip().splitlines() if r.stdout.strip() else []
            return lines[:20]
        except Exception:
            return []

    try:
        repo = Repo(ACTIVE_REPO_PATH, search_parent_directories=True)
        since = datetime.now() - timedelta(hours=hours)
        commits = []
        for c in repo.iter_commits("HEAD", max_count=50):
            if datetime.fromtimestamp(c.committed_date) < since:
                continue
            commits.append(f"{c.hexsha[:7]} {c.summary}")
            if len(commits) >= 20:
                break
        return commits
    except Exception:
        return []


def get_recent_files(hours=24):
    """Files modified in the last N hours in home dir."""
    home = Path.home()
    cutoff = time.time() - (hours * 3600)
    recent = []
    for p in home.rglob("*"):
        try:
            if p.is_file() and p.stat().st_mtime > cutoff:
                # Skip noisy/cache files
                if any(
                    x in str(p)
                    for x in [".cache", "node_modules", ".git", "__pycache__", "Library"]
                ):
                    continue
                recent.append(str(p.relative_to(home)))
                if len(recent) >= 30:
                    break
        except Exception:
            pass
    return recent


def get_shell_history(lines=30):
    """Last N lines of shell history (zsh/bash/PowerShell)."""
    history_paths = [
        Path.home() / ".zsh_history",
        Path.home() / ".bash_history",
        Path.home()
        / "AppData"
        / "Roaming"
        / "Microsoft"
        / "Windows"
        / "PowerShell"
        / "PSReadLine"
        / "ConsoleHost_history.txt",
    ]

    for path in history_paths:
        if path.exists():
            try:
                with open(path, "r", errors="ignore") as f:
                    all_lines = f.readlines()
                return [l.strip() for l in all_lines[-lines:] if l.strip()]
            except Exception:
                pass
    return []


def get_vscode_recent_workspaces(max_items=12):
    """
    4th data source:
    VS Code recently opened folders/workspaces.
    """
    storage = (
        Path.home()
        / "AppData"
        / "Roaming"
        / "Code"
        / "User"
        / "globalStorage"
        / "storage.json"
    )
    if not storage.exists():
        return []

    try:
        raw = json.loads(storage.read_text(encoding="utf-8", errors="ignore"))
        opened = raw.get("history.recentlyOpenedPathsList", {})
        entries = opened.get("entries", [])
        results = []
        for item in entries:
            path = item.get("folderUri") or item.get("workspace", {}).get("configPath")
            if not path:
                continue
            cleaned = path.replace("file://", "").replace("%3A", ":").replace("%20", " ")
            cleaned = cleaned.replace("/", "\\")
            results.append(cleaned)
            if len(results) >= max_items:
                break
        return results
    except Exception:
        return []


# ====== LLM SUMMARY ======

DEBRIEF_PROMPT = """You are a concise engineering debrief assistant for a curious developer.
Read the activity data and infer today's story.
Output EXACTLY 5 lines in this exact schema (no markdown, no bullets):

BUILT: <specific progress from evidence>
BROKE: <issue or risk from evidence, or "Nothing critical broke">
LEARNED: <concrete takeaway>
PATTERN: <dominant pattern/theme from the day>
NEXT: <single practical next action for tomorrow>

Hard rules:
- Exactly 5 lines.
- Max 20 words per line.
- Reference concrete clues (repo names, file names, or command hints) where possible.
- Do not invent dramatic failures.

Data:
{data}

5 lines only:"""


def get_debrief(data_text):
    prompt = DEBRIEF_PROMPT.format(data=data_text[:3000])
    try:
        r = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        return r.stdout.strip()
    except Exception:
        return "[LLM error]"


# ====== MAIN ======


test_repo = ensure_test_repo()
if test_repo:
    ACTIVE_REPO_PATH = test_repo
    console.print(
        Panel(
            f"No git repo found in [bold]{Path.cwd()}[/bold]. Created [bold]{test_repo}[/bold] for demo commits.",
            title="Repo Bootstrap",
            border_style="yellow",
        )
    )

banner = Text(
    "\n"
    "  ____        _ _       ____       _           _  __\n"
    " |  _ \\  __ _(_) |_   _|  _ \\  ___| |__  _ __ (_)/ _|\n"
    " | | | |/ _` | | | | | | | | |/ _ \\ '_ \\| '__|| | |_ \n"
    " | |_| | (_| | | | |_| | |_| |  __/ |_) | |   | |  _|\n"
    " |____/ \\__,_|_|_|\\__, |____/ \\___|_.__/|_|   |_|_|\n"
    "                  |___/                               \n",
    style="bold cyan",
)
console.print(banner)
console.print("[dim]Collecting your last 24 hours of builder telemetry...[/dim]\n")

commits = get_git_commits()
files = get_recent_files()
history = get_shell_history()
workspaces = get_vscode_recent_workspaces()

console.print(f"  Git commits:    {len(commits)}")
console.print(f"  Recent files:   {len(files)}")
console.print(f"  Shell commands: {len(history)}")
console.print(f"  VS Code recent: {len(workspaces)}")
console.print()

# Build data string for LLM
data = []
if commits:
    data.append("GIT COMMITS:\n" + "\n".join(commits[:10]))
if files:
    data.append("FILES MODIFIED:\n" + "\n".join(files[:15]))
if history:
    data.append("SHELL HISTORY:\n" + "\n".join(history[-15:]))
if workspaces:
    data.append("VS CODE RECENT WORKSPACES:\n" + "\n".join(workspaces[:10]))

if not data:
    console.print("[yellow]No data found. Make some git commits first![/yellow]")
    sys.exit(0)

combined = "\n\n".join(data)

console.print("[dim]Asking the brain...[/dim]")
start = time.time()
debrief = get_debrief(combined)
elapsed = time.time() - start

console.print(
    Panel(
        debrief,
        title=f"Your Daily Debrief ({elapsed:.1f}s)",
        subtitle="Flight Recorder Summary",
        border_style="bright_cyan",
    )
)
console.print("\n[bold green]See you tomorrow for Day 14![/bold green]")
