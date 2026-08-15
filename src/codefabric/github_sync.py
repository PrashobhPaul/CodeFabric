"""GitHub account connector: mirror every repo you own into a local
workspace, then index the workspace as one searchable fabric.

Design constraints:
  - Pure stdlib (urllib + tarfile) — no git binary required. Repos are
    fetched as tarballs via the GitHub REST API.
  - Works unauthenticated for public repos (``--user NAME``) and with a
    token (env ``GITHUB_TOKEN`` / ``CODEFABRIC_GITHUB_TOKEN`` or
    ``--token``) for private repos across the whole account.
  - Incremental: a repo is re-downloaded only when its ``pushed_at``
    timestamp changed since the last sync.
  - Safe extraction: path traversal, symlinks, and absolute paths in
    tarballs are rejected.

The network layer is injectable (``fetcher``) so tests run offline.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field

API_URL = "https://api.github.com"
DEFAULT_WORKSPACE = os.path.join("~", "codefabric-workspace")
USER_AGENT = "codefabric"


class GitHubError(RuntimeError):
    pass


@dataclass
class SyncStats:
    synced: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def default_fetcher(url: str, headers: dict[str, str]) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        raise GitHubError(f"network error reaching GitHub: {e.reason}") from e


def resolve_token(explicit: str | None = None) -> str | None:
    return (
        explicit
        or os.environ.get("CODEFABRIC_GITHUB_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or None
    )


def resolve_workspace(explicit: str | None = None) -> str:
    path = explicit or os.environ.get("CODEFABRIC_WORKSPACE") or DEFAULT_WORKSPACE
    return os.path.abspath(os.path.expanduser(path))


class GitHubSync:
    def __init__(
        self,
        token: str | None = None,
        workspace: str | None = None,
        api_url: str = API_URL,
        fetcher=default_fetcher,
    ):
        self.token = resolve_token(token)
        self.workspace = resolve_workspace(workspace)
        self.api_url = api_url.rstrip("/")
        self.fetcher = fetcher

    # -- API helpers ---------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get_json(self, path: str):
        status, body = self.fetcher(f"{self.api_url}{path}", self._headers())
        if status == 401:
            raise GitHubError("GitHub rejected the token (401). Check GITHUB_TOKEN.")
        if status == 403:
            raise GitHubError("GitHub API rate limit or permissions error (403). "
                              "Set GITHUB_TOKEN to raise the rate limit.")
        if status == 404:
            raise GitHubError(f"GitHub returned 404 for {path}.")
        if status != 200:
            raise GitHubError(f"GitHub API error {status} for {path}.")
        return json.loads(body.decode("utf-8"))

    # -- repo listing --------------------------------------------------

    def list_repos(self, user: str | None = None,
                   include_forks: bool = False) -> list[dict]:
        """Every repo on the account, private included when a token is
        set. Falls back to public repos of ``user`` without a token."""
        repos: list[dict] = []
        page = 1
        while True:
            if self.token and not user:
                path = (f"/user/repos?per_page=100&page={page}"
                        f"&affiliation=owner&sort=pushed")
            elif user:
                path = f"/users/{user}/repos?per_page=100&page={page}&sort=pushed"
            else:
                raise GitHubError(
                    "No GitHub token found and no --user given. Either export "
                    "GITHUB_TOKEN (all your repos, private included) or pass "
                    "--user <github-username> (public repos only)."
                )
            batch = self._get_json(path)
            if not batch:
                break
            repos.extend(batch)
            if len(batch) < 100:
                break
            page += 1

        out = []
        for r in repos:
            if r.get("fork") and not include_forks:
                continue
            out.append(
                {
                    "full_name": r["full_name"],
                    "owner": r["owner"]["login"],
                    "name": r["name"],
                    "default_branch": r.get("default_branch") or "main",
                    "pushed_at": r.get("pushed_at") or "",
                    "private": bool(r.get("private")),
                    "size_kb": r.get("size", 0),
                }
            )
        return out

    # -- sync ----------------------------------------------------------

    def sync(self, user: str | None = None, include_forks: bool = False,
             log=print) -> SyncStats:
        os.makedirs(self.workspace, exist_ok=True)
        manifest_path = os.path.join(self.workspace, ".codefabric-sync.json")
        manifest: dict[str, str] = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)

        stats = SyncStats()
        repos = self.list_repos(user=user, include_forks=include_forks)
        log(f"found {len(repos)} repositories")

        for repo in repos:
            full = repo["full_name"]
            dest = os.path.join(self.workspace, f"{repo['owner']}__{repo['name']}")
            stamp = repo["pushed_at"]
            if manifest.get(full) == stamp and os.path.isdir(dest):
                stats.skipped.append(full)
                continue
            try:
                log(f"  syncing {full} ...")
                data = self._download_tarball(repo)
                self._extract_tarball(data, dest)
                manifest[full] = stamp
                stats.synced.append(full)
            except (GitHubError, tarfile.TarError, OSError) as e:
                log(f"  FAILED {full}: {e}")
                stats.failed.append(full)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        return stats

    def _download_tarball(self, repo: dict) -> bytes:
        path = (f"/repos/{repo['owner']}/{repo['name']}/tarball/"
                f"{repo['default_branch']}")
        status, body = self.fetcher(f"{self.api_url}{path}", self._headers())
        if status != 200:
            raise GitHubError(f"tarball download failed ({status}) for "
                              f"{repo['full_name']}")
        return body

    @staticmethod
    def _extract_tarball(data: bytes, dest: str) -> None:
        """Extract, stripping the top-level ``owner-repo-sha/`` directory
        and refusing anything unsafe."""
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.makedirs(dest, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            for member in tar:
                if not (member.isfile() or member.isdir()):
                    continue  # skip symlinks, devices, etc.
                parts = member.name.split("/")
                if len(parts) < 2:
                    continue  # the top-level wrapper dir itself
                rel = "/".join(parts[1:])
                if not rel or rel.startswith("/") or ".." in rel.split("/"):
                    continue
                target = os.path.join(dest, *rel.split("/"))
                if not os.path.abspath(target).startswith(os.path.abspath(dest)):
                    continue
                if member.isdir():
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    with open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
