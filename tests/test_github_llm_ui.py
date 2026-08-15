import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
import urllib.request

from codefabric.github_sync import GitHubError, GitHubSync
from codefabric.indexer import Indexer, IndexConfig
from codefabric.search import SearchEngine
from codefabric import llm


def make_tarball(files: dict[str, str], top: str = "owner-repo-abc123") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=f"{top}/{rel}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeGitHub:
    """Offline stand-in for the GitHub REST API."""

    def __init__(self):
        self.repos = [
            {
                "full_name": "alice/webapp", "name": "webapp",
                "owner": {"login": "alice"}, "default_branch": "main",
                "pushed_at": "2026-08-01T00:00:00Z", "private": False,
                "fork": False, "size": 10,
            },
            {
                "full_name": "alice/oldfork", "name": "oldfork",
                "owner": {"login": "alice"}, "default_branch": "main",
                "pushed_at": "2026-01-01T00:00:00Z", "private": False,
                "fork": True, "size": 5,
            },
        ]
        self.tarballs = {
            "alice/webapp": make_tarball({
                "app/main.py":
                    'def handle_login(user, password):\n'
                    '    """Validate a login attempt."""\n'
                    '    return check_password(user, password)\n\n\n'
                    'def check_password(user, password):\n'
                    '    """Compare against the stored hash."""\n'
                    '    return True\n',
                "README.md": "# webapp\n",
                # unsafe members that must be rejected:
                "../evil.py": "print('escape')\n",
            }),
        }
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append(url)
        if "/users/alice/repos" in url or "/user/repos" in url:
            if "page=1" in url:
                return 200, json.dumps(self.repos).encode()
            return 200, b"[]"
        for full, blob in self.tarballs.items():
            if f"/repos/{full}/tarball/" in url:
                return 200, blob
        return 404, b"{}"


class GitHubSyncTests(unittest.TestCase):
    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="codefabric_ws_")
        self.fake = FakeGitHub()
        self.sync = GitHubSync(token=None, workspace=self.ws,
                               fetcher=self.fake)

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_requires_user_or_token(self):
        saved = {v: os.environ.pop(v, None)
                 for v in ("GITHUB_TOKEN", "CODEFABRIC_GITHUB_TOKEN")}
        try:
            sync = GitHubSync(token=None, workspace=self.ws, fetcher=self.fake)
            with self.assertRaises(GitHubError):
                sync.list_repos()
        finally:
            for var, val in saved.items():
                if val is not None:
                    os.environ[var] = val

    def test_list_repos_filters_forks(self):
        repos = self.sync.list_repos(user="alice")
        self.assertEqual([r["full_name"] for r in repos], ["alice/webapp"])
        repos = self.sync.list_repos(user="alice", include_forks=True)
        self.assertEqual(len(repos), 2)

    def test_sync_extracts_and_indexes_end_to_end(self):
        stats = self.sync.sync(user="alice", log=lambda *_: None)
        self.assertEqual(stats.synced, ["alice/webapp"])
        extracted = os.path.join(self.ws, "alice__webapp", "app", "main.py")
        self.assertTrue(os.path.exists(extracted))
        # path traversal member must NOT land outside the workspace
        self.assertFalse(os.path.exists(os.path.join(self.ws, "..", "evil.py"))
                         and os.path.exists(os.path.join(
                             os.path.dirname(self.ws), "evil.py")))

        # index the workspace and run a federated search
        Indexer(self.ws, IndexConfig()).build()
        engine = SearchEngine(self.ws)
        results = engine.search("validate login attempt password", k=5)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.symbol, "handle_login")
        self.assertTrue(results[0].chunk.file_path.startswith("alice__webapp/"))

    def test_sync_is_incremental(self):
        self.sync.sync(user="alice", log=lambda *_: None)
        stats2 = self.sync.sync(user="alice", log=lambda *_: None)
        self.assertEqual(stats2.synced, [])
        self.assertEqual(stats2.skipped, ["alice/webapp"])


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="codefabric_llm_")
        with open(os.path.join(self.root, "billing.py"), "w") as f:
            f.write('def compute_tax(amount):\n'
                    '    """Apply the 18 percent tax rate."""\n'
                    '    return amount * 0.18\n')
        Indexer(self.root, IndexConfig()).build()
        self.engine = SearchEngine(self.root)
        # make sure no real provider is picked up from the environment
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "CODEFABRIC_LLM_BASE_URL"):
            os.environ.pop(var, None)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_extractive_answer_without_llm(self):
        result = llm.ask(self.engine, "how is tax computed", provider="auto")
        self.assertEqual(result["mode"], "extractive")
        self.assertIn("compute_tax", result["answer"])
        self.assertTrue(result["sources"])
        self.assertEqual(result["sources"][0]["file"], "billing.py")

    def test_provider_none_forces_extractive(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        try:
            result = llm.ask(self.engine, "tax rate", provider="none")
            self.assertEqual(result["mode"], "extractive")
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_complete_raises_without_provider(self):
        with self.assertRaises(llm.LLMUnavailable):
            llm.complete("hello", provider=None)

    def test_build_prompt_contains_snippets_and_citations(self):
        results = self.engine.search("tax", k=2)
        prompt = llm.build_prompt("what is the tax rate?", results)
        self.assertIn("[1] billing.py", prompt)
        self.assertIn("compute_tax", prompt)
        self.assertIn("citing snippet numbers", prompt)


class WebUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from codefabric.webui import serve

        cls.root = tempfile.mkdtemp(prefix="codefabric_ui_")
        with open(os.path.join(cls.root, "orders.py"), "w") as f:
            f.write('def total_price(items):\n'
                    '    """Sum item prices."""\n'
                    '    return sum(i["price"] for i in items)\n')
        Indexer(cls.root, IndexConfig()).build()
        cls.server = serve(cls.root, port=0, open_browser=False,
                           background=True)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        shutil.rmtree(cls.root, ignore_errors=True)

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return r.status, r.read()

    def test_page_served(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"CodeFabric", body)

    def test_api_search(self):
        status, body = self._get("/api/search?q=sum+item+prices&k=5")
        self.assertEqual(status, 200)
        results = json.loads(body)
        self.assertTrue(results)
        self.assertEqual(results[0]["chunk"]["symbol"], "total_price")

    def test_api_stats(self):
        status, body = self._get("/api/stats")
        stats = json.loads(body)
        self.assertGreaterEqual(stats["chunks"], 1)
        self.assertGreaterEqual(stats["symbols"], 1)

    def test_api_ask_extractive(self):
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "CODEFABRIC_LLM_BASE_URL"):
            os.environ.pop(var, None)
        req = urllib.request.Request(
            self.base + "/api/ask",
            data=json.dumps({"question": "how are prices summed"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        self.assertEqual(data["mode"], "extractive")
        self.assertIn("total_price", data["answer"])


if __name__ == "__main__":
    unittest.main()
