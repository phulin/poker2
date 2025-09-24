#!/usr/bin/env python3
"""
Convert relative imports to absolute imports within a package tree.

Usage:
  python scripts/convert_relative_imports.py --path src --write

Notes:
  - Detects the top-level package for each file by walking up parents that
    contain an __init__.py; the highest such directory is treated as the
    package root, and its directory name becomes the top-level package name.
  - Rewrites only relative ImportFrom nodes (level > 0). Absolute imports are untouched.
  - Requires Python 3.9+ (uses ast.unparse).
"""

import argparse
import ast
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class PackageContext:
    package_root: Path
    top_package_name: str
    module_package_parts: List[str]


def find_package_root(file_path: Path) -> Optional[Path]:
    """Return the highest ancestor directory containing __init__.py.

    If no ancestor has __init__.py, return None.
    """
    cur = file_path.parent
    highest: Optional[Path] = None
    while True:
        init_file = cur / "__init__.py"
        if init_file.exists():
            highest = cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return highest


def compute_module_package_parts(package_root: Path, file_path: Path) -> List[str]:
    """Given package_root and file_path, return the module's package parts (excluding the module file name).

    Example: package_root=/repo/src/mypkg, file=/repo/src/mypkg/sub/thing/util.py
             -> ["sub", "thing"]
    """
    rel = file_path.parent.relative_to(package_root)
    if rel == Path(""):
        return []
    return list(rel.parts)


def resolve_absolute_module(
    ctx: PackageContext, level: int, module: Optional[str]
) -> str:
    """Resolve the absolute module string from a relative import.

    - level=N means go up N levels from current module package
    - module may be None or a dotted path
    """
    # Base parts start from top-level package name
    parts: List[str] = [ctx.top_package_name]

    # Compute current package parts and move up `level` directories
    if level > 0:
        if level > len(ctx.module_package_parts) + 1:
            # If import goes beyond the package root, clamp at root
            up_to = 0
        else:
            up_to = max(0, len(ctx.module_package_parts) - (level - 1))
        parts.extend(ctx.module_package_parts[:up_to])

    # Append the requested submodule path if provided
    if module:
        parts.extend(module.split("."))

    # Normalize by removing empty segments
    parts = [p for p in parts if p]
    return ".".join(parts)


def convert_file(
    path: Path, write: bool, make_backup: bool
) -> Tuple[bool, Optional[str]]:
    """Convert a single file. Returns (changed, error_message)."""
    try:
        src = path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Failed to read {path}: {e}"

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"Syntax error in {path}: {e}"

    pkg_root = find_package_root(path)
    if pkg_root is None:
        # Not inside a package; skip conversions for this file
        return False, None

    top_package_name = pkg_root.name
    module_parts = compute_module_package_parts(pkg_root, path)
    ctx = PackageContext(pkg_root, top_package_name, module_parts)

    changed = False

    class Rewriter(ast.NodeTransformer):
        def visit_ImportFrom(self, node: ast.ImportFrom):  # type: ignore[override]
            nonlocal changed
            if node.level and node.level > 0:
                abs_module = resolve_absolute_module(ctx, node.level, node.module)
                new_node = ast.ImportFrom(module=abs_module, names=node.names, level=0)
                changed = True
                return ast.copy_location(new_node, node)
            return node

    new_tree = Rewriter().visit(tree)
    ast.fix_missing_locations(new_tree)

    if not changed:
        return False, None

    try:
        new_code = ast.unparse(new_tree)
    except Exception as e:
        return False, f"Failed to unparse {path}: {e}"

    if write:
        if make_backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            try:
                shutil.copy2(path, backup_path)
            except Exception as e:
                return False, f"Failed to write backup for {path}: {e}"
        try:
            path.write_text(
                new_code + ("\n" if not new_code.endswith("\n") else ""),
                encoding="utf-8",
            )
        except Exception as e:
            return False, f"Failed to write {path}: {e}"

    return True, None


def iter_py_files(base: Path) -> List[Path]:
    return [p for p in base.rglob("*.py") if p.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert relative imports to absolute within a package tree."
    )
    parser.add_argument(
        "--path", default="src", help="Directory to process (default: src)"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes back to files (default: dry-run)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write .bak backups alongside modified files",
    )

    args = parser.parse_args()

    base = Path(args.path).resolve()
    if not base.exists() or not base.is_dir():
        print(f"Error: path not found or not a directory: {base}", file=sys.stderr)
        return 2

    files = iter_py_files(base)
    total = 0
    changed = 0
    failed = 0
    for f in files:
        total += 1
        was_changed, err = convert_file(f, write=args.write, make_backup=args.backup)
        rel = f.relative_to(base)
        if err:
            failed += 1
            print(f"[ERROR] {rel}: {err}")
        elif was_changed:
            changed += 1
            action = "UPDATED" if args.write else "WOULD-UPDATE"
            print(f"[{action}] {rel}")

    print(
        f"Processed {total} files. {'Changed' if args.write else 'Would change'} {changed}. Failures: {failed}."
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
