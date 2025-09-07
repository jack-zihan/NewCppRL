#!/usr/bin/env python3
"""
Minimal MCP server: "context7"

Capabilities (read-first, safe-by-default):
1) fs_read:    Read text files under a root dir
2) fs_glob:    List files by glob pattern under root
3) http_fetch: Simple HTTP GET with optional headers (urllib)
4) sqlite_query: Read-only SELECT on SQLite DB
5) git_diff:   Run git diff for a path (if git present)
6) git_show:   Show a file at revision (if git present)
7) search_text: Simple recursive substring search under root

Notes:
- Requires `mcp` Python package (Model Context Protocol SDK).
- Uses stdio transport; configure in your MCP client (Codex) via servers.json.
- Keeps a strict root sandbox; blocks paths escaping the root.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
except Exception as e:  # pragma: no cover
    print(
        "[context7] Missing `mcp` package. Install with: pip install mcp",
        file=sys.stderr,
    )
    raise


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


@dataclass
class Ctx:
    root: Path


def build_server(ctx: Ctx) -> Server:
    server = Server("context7")

    @server.tool()
    async def fs_read(path: str, encoding: str = "utf-8") -> str:
        """Read a text file under the sandbox root.

        Args:
            path: Relative file path under root.
            encoding: Text encoding.
        Returns:
            File content as string.
        """
        p = (ctx.root / path).resolve()
        if not _is_subpath(p, ctx.root):
            raise ValueError("path escapes root sandbox")
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p.read_text(encoding=encoding)

    @server.tool()
    async def fs_glob(pattern: str) -> List[str]:
        """Glob files relative to root; returns relative POSIX paths."""
        matches = [
            str(Path(os.path.relpath(p, ctx.root)).as_posix())
            for p in ctx.root.glob(pattern)
            if p.is_file()
        ]
        return sorted(matches)

    @server.tool()
    async def search_text(query: str, exts: Optional[List[str]] = None, limit: int = 200) -> List[Dict[str, str]]:
        """Naive recursive substring search under root (small repos).

        Args:
            query: Substring to search (case sensitive).
            exts: Optional list of file extensions to include (e.g., [".py", ".md"]).
            limit: Max number of hits to return.
        Returns: List of {path, line_no, line} results.
        """
        results: List[Dict[str, str]] = []
        count = 0
        for dirpath, _, filenames in os.walk(ctx.root):
            for fn in filenames:
                p = Path(dirpath) / fn
                if exts and p.suffix not in exts:
                    continue
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if query in line:
                                results.append(
                                    {
                                        "path": str(Path(os.path.relpath(p, ctx.root)).as_posix()),
                                        "line_no": str(i),
                                        "line": line.rstrip("\n"),
                                    }
                                )
                                count += 1
                                if count >= limit:
                                    return results
                except Exception:
                    continue
        return results

    @server.tool()
    async def http_fetch(url: str, headers: Optional[Dict[str, str]] = None, timeout_s: float = 10.0) -> Dict[str, str]:
        """HTTP GET via urllib (no extra deps).

        Returns dict with {status, headers, body}.
        """
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            try:
                if "charset=" in ctype:
                    encoding = ctype.split("charset=")[-1]
                    body_text = body.decode(encoding, errors="replace")
                else:
                    body_text = body.decode("utf-8", errors="replace")
            except Exception:
                body_text = body.decode("utf-8", errors="replace")
            return {
                "status": str(resp.status),
                "headers": json.dumps(dict(resp.headers)),
                "body": body_text,
            }

    @server.tool()
    async def sqlite_query(db_path: str, sql: str, params_json: str = "[]") -> Dict[str, object]:
        """Run a read-only SELECT on SQLite. Blocks write statements."""
        if any(kw in sql.lower() for kw in ("update", "insert", "delete", "create", "drop", "alter")):
            raise ValueError("Only SELECT queries are allowed")
        db_p = (ctx.root / db_path).resolve()
        if not _is_subpath(db_p, ctx.root):
            raise ValueError("db_path escapes root")
        if not db_p.exists():
            raise FileNotFoundError(str(db_p))
        params = json.loads(params_json or "[]")
        con = sqlite3.connect(str(db_p))
        con.row_factory = sqlite3.Row
        try:
            cur = con.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
            return {"rows": rows, "count": len(rows)}
        finally:
            con.close()

    def _git_available() -> bool:
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            return True
        except Exception:
            return False

    @server.tool()
    async def git_diff(rev_range: str = "HEAD~1..HEAD", path: str = ".") -> str:
        """Return `git diff` output if git is available."""
        if not _git_available():
            return "git not available"
        proc = await asyncio.create_subprocess_exec(
            "git",
            "diff",
            rev_range,
            "--",
            path,
            cwd=str(ctx.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(err.decode())
        return out.decode()

    @server.tool()
    async def git_show(rev: str, file_path: str) -> str:
        """Show file content at revision if git is available."""
        if not _git_available():
            return "git not available"
        proc = await asyncio.create_subprocess_exec(
            "git",
            "show",
            f"{rev}:{file_path}",
            cwd=str(ctx.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(err.decode())
        return out.decode()

    return server


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="context7 MCP server (stdio)")
    parser.add_argument("--root", default=os.getcwd(), help="sandbox root directory")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    ctx = Ctx(root=root)
    server = build_server(ctx)

    async def runner():
        async with stdio_server() as (rx, tx):
            await server.run(rx, tx)

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

