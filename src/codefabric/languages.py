"""Language detection and per-language chunking hints."""
from __future__ import annotations

import os

# Extension -> language id. Covers the mainstream enterprise stack.
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".scala": "scala",
    ".swift": "swift",
    ".sh": "shell",
    ".bash": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".rst": "markdown",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".tf": "terraform",
    ".proto": "proto",
}

# Languages we treat as "code" (get structural chunking + symbols).
CODE_LANGUAGES = {
    "python", "javascript", "typescript", "java", "kotlin", "go", "rust",
    "c", "cpp", "csharp", "ruby", "php", "scala", "swift",
}

# Directories that never contain useful source for retrieval.
DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache", ".idea",
    ".vscode", "target", ".codefabric", ".next", ".terraform", "vendor",
    "site-packages", ".eggs",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar", ".gz",
    ".whl", ".so", ".dylib", ".dll", ".exe", ".bin", ".pyc", ".class",
    ".jar", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".sqlite",
    ".db", ".parquet", ".npy", ".npz", ".onnx", ".pt", ".lock",
}


def detect_language(path: str) -> str | None:
    """Return the language id for a path, or None if not indexable."""
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in BINARY_EXTENSIONS:
        return None
    return EXTENSION_MAP.get(ext)
