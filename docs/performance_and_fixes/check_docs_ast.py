#!/usr/bin/env python3
"""
check_docs_ast.py: Automated AST Syntax Verification Harness for Documentation Code Blocks.

Iterates over Markdown files in docs/performance_and_fixes/ (00~07, 10, README.md),
extracts all ```python ... ``` code blocks, and validates syntax integrity via ast.parse().

Usage:
    python check_docs_ast.py                  # Verify canonical volumes (00~07, 10, README.md)
    python check_docs_ast.py --all            # Verify all .md files in docs/performance_and_fixes/
    python check_docs_ast.py <file_or_dir>    # Verify specific file or custom directory
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import sys
import textwrap


CANONICAL_TARGETS = [
    "00_HUMAN_MASTER_GUIDE.md",
    "01_architecture_and_decoupling_overview.md",
    "02_environment_and_geometry_optimizations.md",
    "03_cpp_acceleration_and_compilation.md",
    "04_concurrency_ipc_and_memory.md",
    "05_critical_bug_fixes_and_stability.md",
    "06_empirical_benchmarks_and_metrics.md",
    "07_safe_migration_and_verification_guide.md",
    "10_v13_codex_reaudit_and_optimal_fixes_synthesis.md",
    "README.md",
]


def extract_python_blocks(md_path: Path) -> list[tuple[int, str]]:
    """Extract all ```python ... ``` blocks with 1-indexed starting line numbers."""
    text = md_path.read_text(encoding="utf-8")
    pattern = r"```python\s*\n(.*?)```"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    blocks = []
    for m in matches:
        line_num = text[:m.start()].count("\n") + 1
        code = m.group(1)
        blocks.append((line_num, code))
    return blocks


def check_markdown_file(md_path: Path, verbose: bool = True) -> tuple[int, int]:
    """
    Check a single Markdown file for Python AST syntax validity.
    Returns: (total_blocks, error_count)
    """
    if not md_path.exists():
        if verbose:
            print(f"[SKIP] {md_path.name} (file does not exist)")
        return 0, 0

    blocks = extract_python_blocks(md_path)
    if verbose:
        print(f"Checking {md_path.name}: found {len(blocks)} python code blocks.")

    errors = 0
    for idx, (line_num, raw_code) in enumerate(blocks, start=1):
        dedented_code = textwrap.dedent(raw_code)
        try:
            ast.parse(dedented_code)
        except SyntaxError as e:
            errors += 1
            if verbose:
                print(f"  [SYNTAX ERROR] Block #{idx} (MD line {line_num}, code line {e.lineno}): {e.msg}")
                preview_lines = dedented_code.splitlines()[:5]
                preview = "\n".join(preview_lines)
                print(f"    Code snippet preview:\n{textwrap.indent(preview, '      ')}")

    if verbose:
        if errors == 0:
            print(f"  [PASS] All {len(blocks)} python blocks valid.")
        else:
            print(f"  [FAIL] {errors}/{len(blocks)} blocks failed AST parsing.")

    return len(blocks), errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated AST Syntax Verification for Doc Code Blocks")
    parser.add_argument("target", nargs="?", default=None, help="Specific file or directory to check")
    parser.add_argument("--all", action="store_true", help="Check all .md files in the documentation directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed block-level logs")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.target:
        target_path = Path(args.target).resolve()
        if target_path.is_file():
            md_files = [target_path]
        elif target_path.is_dir():
            md_files = sorted(target_path.glob("*.md"))
        else:
            print(f"Error: Target '{args.target}' not found.", file=sys.stderr)
            return 2
    elif args.all:
        md_files = sorted(base_dir.glob("*.md"))
    else:
        # Default: Canonical volumes
        md_files = []
        for name in CANONICAL_TARGETS:
            p = base_dir / name
            if p.exists():
                md_files.append(p)

    print("=" * 78)
    print(" NewCppRL Documentation AST Syntax Verification Suite")
    print(f" Directory: {base_dir}")
    print(f" Scope: {'All Markdown files' if args.all else 'Canonical Target Volumes'}")
    print("=" * 78)

    total_files = len(md_files)
    total_blocks = 0
    total_errors = 0
    files_with_errors = 0

    for md_file in md_files:
        b_count, err_count = check_markdown_file(md_file, verbose=not args.quiet)
        total_blocks += b_count
        total_errors += err_count
        if err_count > 0:
            files_with_errors += 1

    print("=" * 78)
    print(f"Summary: {total_blocks} python blocks across {total_files} files checked.")
    print(f"Result: {total_errors} syntax errors in {files_with_errors} files.")
    print("=" * 78)

    if total_errors == 0:
        print("[SUCCESS] All documentation code blocks 100% compliant with Python 3 AST.")
        return 0
    else:
        print(f"[FAILURE] {total_errors} AST syntax errors detected across documentation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
