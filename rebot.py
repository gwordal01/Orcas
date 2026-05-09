"""
BUILDCORED ORCAS - Day 14: RegisterBot
Tiny CPU simulator with 8 registers, ALU, and a minimal instruction set.
Local LLM narrates each step.

Hardware concept: fetch-decode-execute cycle, registers, and the ALU.

Run: python rebot.py
Prereqs:
- ollama serve
- ollama pull qwen2.5:3b
"""

import subprocess
import sys
import time

try:
    from rich import box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
except ImportError as error:
    print(f"Rich import failed: {error}")
    print("Try: py -3.11 -m pip install rich")
    sys.exit(1)


MODEL = "qwen2.5:3b"
NARRATOR_TIMEOUT_SECONDS = 8
STEP_DELAY_SECONDS = 0.1

PALETTE = {
    "accent": "bright_cyan",
    "accent_2": "bright_magenta",
    "panel": "cyan",
    "panel_soft": "bright_blue",
    "good": "bright_green",
    "warn": "bright_yellow",
    "muted": "grey70",
    "title": "bold bright_white",
}


class CPU:
    def __init__(self):
        self.registers = [0] * 8
        self.pc = 0
        self.flags = {"ZERO": False, "NEG": False}
        self.halted = False
        self.program = []

    def reset(self):
        self.__init__()


def alu(op, a, b):
    """Arithmetic Logic Unit - does the math."""
    if op == "ADD":
        return a + b
    if op == "SUB":
        return a - b
    if op == "MUL":
        return a * b
    if op == "AND":
        return a & b
    if op == "OR":
        return a | b
    raise ValueError(f"Unknown ALU op: {op}")


def reg_index(s):
    """Convert R0 to 0, R1 to 1, etc."""
    return int(s[1:])


def is_register(s):
    return isinstance(s, str) and s.startswith("R") and s[1:].isdigit()


def get_value(cpu, arg):
    """If arg is a register name, return its value. Otherwise parse as int."""
    if is_register(arg):
        return cpu.registers[reg_index(arg)]
    return int(arg)


def parse_addr(arg):
    return int(arg)


def execute(cpu, instruction):
    """Execute one instruction and return a human-readable description."""
    op = instruction[0].upper()
    args = instruction[1:]

    if op == "MOV":
        dst = reg_index(args[0])
        value = get_value(cpu, args[1])
        cpu.registers[dst] = value
        cpu.pc += 1
        return f"R{dst} <- {value}"

    if op in {"ADD", "SUB", "MUL"}:
        dst = reg_index(args[0])
        left = cpu.registers[dst]
        right = get_value(cpu, args[1])
        cpu.registers[dst] = alu(op, left, right)
        cpu.pc += 1
        return f"R{dst} = {left} {op} {right} = {cpu.registers[dst]}"

    if op == "CMP":
        left = get_value(cpu, args[0])
        right = get_value(cpu, args[1])
        cpu.flags["ZERO"] = left == right
        cpu.flags["NEG"] = left < right
        cpu.pc += 1
        return f"CMP {left} vs {right} -> ZERO={cpu.flags['ZERO']}, NEG={cpu.flags['NEG']}"

    if op == "JMP":
        cpu.pc = parse_addr(args[0])
        return f"JMP -> line {cpu.pc}"

    if op == "JZ":
        if cpu.flags["ZERO"]:
            cpu.pc = parse_addr(args[0])
            return f"JZ taken -> line {cpu.pc}"
        cpu.pc += 1
        return "JZ not taken"

    if op == "JNZ":
        if not cpu.flags["ZERO"]:
            cpu.pc = parse_addr(args[0])
            return f"JNZ taken -> line {cpu.pc}"
        cpu.pc += 1
        return "JNZ not taken"

    if op == "HALT":
        cpu.halted = True
        return "HALT - CPU stopped"

    cpu.pc += 1
    return f"Unknown op: {op}"


NARRATOR_PROMPT = """You are narrating a tiny CPU simulator for a beginner.
Describe this single step in one short sentence, at most 14 words.
Use simple hardware language like register, ALU, flag, or program counter.
Do not restate every register value.

Instruction: {instr}
Effect: {effect}
Registers after: {regs}
"""


def fallback_narration(instr, effect):
    op = instr[0].upper()
    if op == "MOV":
        return "The control unit loaded a value into a register."
    if op in {"ADD", "SUB", "MUL"}:
        return f"The ALU executed {op.lower()} and stored the result in the destination register."
    if op == "CMP":
        return "The CPU compared two values and updated the status flags."
    if op in {"JMP", "JZ", "JNZ"}:
        return "The control unit changed the program counter for control flow."
    if op == "HALT":
        return "The CPU stopped the fetch-decode-execute cycle."
    return effect


def narrate(instr, effect, regs):
    try:
        result = subprocess.run(
            [
                "ollama",
                "run",
                MODEL,
                NARRATOR_PROMPT.format(instr=" ".join(map(str, instr)), effect=effect, regs=regs),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=NARRATOR_TIMEOUT_SECONDS,
        )
        narration = result.stdout.strip()
        return narration or fallback_narration(instr, effect)
    except Exception:
        return fallback_narration(instr, effect)


def render_banner(n):
    title = Text("REGISTERBOT", style="bold bright_white")
    subtitle = Text("tiny cpu simulator • fetch decode execute", style="bright_cyan")
    info = Text(f"computing {n}! with an assembly loop", style="grey70")
    content = Group(
        Align.center(title),
        Align.center(subtitle),
        Align.center(info),
    )
    console.print(
        Panel(
            content,
            border_style=PALETTE["panel"],
            box=box.DOUBLE,
            padding=(1, 4),
        )
    )


def build_register_table(cpu, step, instr):
    table = Table(
        title=f"Step {step:02d}  •  {' '.join(map(str, instr))}",
        title_style="bold bright_white",
        show_header=True,
        header_style="bold bright_cyan",
        box=box.ROUNDED,
        border_style=PALETTE["panel_soft"],
        pad_edge=False,
        expand=True,
    )
    for i in range(8):
        table.add_column(f"R{i}", justify="center")
    table.add_row(
        *[
            f"[bold {PALETTE['accent'] if value != 0 else PALETTE['muted']}]{value}[/]"
            for value in cpu.registers
        ]
    )
    return table


def build_status_panels(cpu, effect, narration):
    zero_style = PALETTE["good"] if cpu.flags["ZERO"] else PALETTE["muted"]
    neg_style = PALETTE["warn"] if cpu.flags["NEG"] else PALETTE["muted"]

    effect_panel = Panel(
        effect,
        title="Effect",
        title_align="left",
        border_style=PALETTE["accent"],
        box=box.ROUNDED,
        padding=(1, 2),
    )
    flags_panel = Panel(
        f"[bold {zero_style}]ZERO={cpu.flags['ZERO']}[/]\n"
        f"[bold {neg_style}]NEG={cpu.flags['NEG']}[/]\n"
        f"[bold {PALETTE['accent_2']}]PC={cpu.pc}[/]",
        title="Flags",
        title_align="left",
        border_style=PALETTE["accent_2"],
        box=box.ROUNDED,
        padding=(1, 2),
    )
    narrator_panel = Panel(
        narration,
        title="Narrator",
        title_align="left",
        border_style=PALETTE["good"],
        box=box.ROUNDED,
        padding=(1, 2),
    )
    return effect_panel, flags_panel, narrator_panel


def show_state(cpu, step, instr, effect, narration):
    register_table = build_register_table(cpu, step, instr)
    effect_panel, flags_panel, narrator_panel = build_status_panels(cpu, effect, narration)

    console.print(register_table)
    console.print(Columns([effect_panel, flags_panel], equal=True, expand=True))
    if narration:
        console.print(narrator_panel)
    console.print()


def build_factorial_program(n):
    """Build a tiny assembly program that computes n! into R1."""
    return [
        ["MOV", "R0", str(n)],
        ["MOV", "R1", "1"],
        ["MOV", "R2", "0"],
        ["MOV", "R3", "1"],
        ["CMP", "R0", "R2"],
        ["JZ", "9"],
        ["MUL", "R1", "R0"],
        ["SUB", "R0", "R3"],
        ["JMP", "4"],
        ["HALT"],
    ]


def parse_factorial_input():
    if len(sys.argv) > 1:
        raw_value = sys.argv[1]
    else:
        raw_value = "5"

    try:
        n = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Factorial input must be an integer, got {raw_value!r}.") from exc

    if n < 0:
        raise ValueError("Factorial input must be non-negative.")

    return n


def run_factorial_demo(n):
    console.print()
    render_banner(n)

    cpu = CPU()
    cpu.program = build_factorial_program(n)
    step = 0

    while not cpu.halted and cpu.pc < len(cpu.program) and step < 200:
        instr = cpu.program[cpu.pc]
        step += 1

        effect = execute(cpu, instr)
        regs_str = ", ".join(f"R{i}={value}" for i, value in enumerate(cpu.registers))
        narration = narrate(instr, effect, regs_str)

        show_state(cpu, step, instr, effect, narration)
        time.sleep(STEP_DELAY_SECONDS)

    expected = factorial_expected(n)
    matches = cpu.registers[1] == expected
    status_style = PALETTE["good"] if matches else "bold red"
    summary = Group(
        Align.center(Text("Execution Complete", style="bold bright_white")),
        Align.center(Text(f"steps: {step}", style="bright_cyan")),
        Align.center(Text(f"result: {cpu.registers[1]}", style=f"bold {PALETTE['good']}")),
        Align.center(Text(f"expected: {expected}", style="grey70")),
        Align.center(Text("status: verified" if matches else "status: mismatch", style=status_style)),
    )
    console.print(
        Panel(
            summary,
            title="Run Summary",
            title_align="left",
            border_style=PALETTE["good"] if matches else "red",
            box=box.DOUBLE,
            padding=(1, 4),
        )
    )
    console.print("[dim]See you tomorrow for Day 15![/dim]")


def factorial_expected(n):
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result


if __name__ == "__main__":
    try:
        run_factorial_demo(parse_factorial_input())
    except ValueError as error:
        console.print(
            Panel(
                f"{error}\n\nUsage: python rebot.py 5",
                title="Input Error",
                title_align="left",
                border_style="red",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
