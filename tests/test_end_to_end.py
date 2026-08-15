import json
import os
import shutil
import tempfile
import unittest

from codefabric.indexer import Indexer, IndexConfig
from codefabric.search import SearchEngine

AUTH_PY = '''"""Authentication helpers."""
import hashlib

from .storage import load_user


def hash_password(password: str, salt: str) -> str:
    """Hash a password with a salt using sha256."""
    return hashlib.sha256((salt + password).encode()).hexdigest()


def authenticate(username: str, password: str) -> bool:
    """Check a username/password pair against stored credentials."""
    user = load_user(username)
    if user is None:
        return False
    return hash_password(password, user["salt"]) == user["hash"]
'''

STORAGE_PY = '''"""User storage backed by a JSON file."""
import json
import os

DB_PATH = os.environ.get("USER_DB", "users.json")


def load_user(username: str):
    """Load a single user record from the JSON database."""
    if not os.path.exists(DB_PATH):
        return None
    with open(DB_PATH) as f:
        users = json.load(f)
    return users.get(username)


def save_user(username: str, record: dict) -> None:
    """Persist a user record."""
    users = {}
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            users = json.load(f)
    users[username] = record
    with open(DB_PATH, "w") as f:
        json.dump(users, f)
'''

BILLING_JS = '''import { getUser } from "./users";

export function calculateInvoiceTotal(items) {
  let total = 0;
  for (const item of items) {
    total += item.price * item.quantity;
  }
  return total;
}

export class InvoiceBuilder {
  constructor(customer) {
    this.customer = customer;
    this.items = [];
  }
}
'''


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="codefabric_e2e_")
        pkg = os.path.join(cls.root, "app")
        os.makedirs(pkg)
        open(os.path.join(pkg, "__init__.py"), "w").close()
        with open(os.path.join(pkg, "auth.py"), "w") as f:
            f.write(AUTH_PY)
        with open(os.path.join(pkg, "storage.py"), "w") as f:
            f.write(STORAGE_PY)
        with open(os.path.join(cls.root, "billing.js"), "w") as f:
            f.write(BILLING_JS)
        cls.stats = Indexer(cls.root, IndexConfig()).build()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_index_stats(self):
        self.assertGreaterEqual(self.stats.files_indexed, 4)
        self.assertGreater(self.stats.chunks, 4)
        self.assertGreater(self.stats.symbols, 5)

    def test_hybrid_search_finds_relevant_function(self):
        engine = SearchEngine(self.root)
        results = engine.search("verify user password credentials", k=5)
        self.assertTrue(results)
        top_symbols = [r.chunk.symbol for r in results[:3]]
        self.assertTrue(
            any(s in ("authenticate", "hash_password") for s in top_symbols),
            f"expected auth symbols in top hits, got {top_symbols}",
        )

    def test_search_cross_language(self):
        engine = SearchEngine(self.root)
        results = engine.search("invoice total price", k=5, language="javascript")
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.language, "javascript")
        self.assertEqual(results[0].chunk.symbol, "calculateInvoiceTotal")

    def test_graph_context_on_results(self):
        engine = SearchEngine(self.root)
        results = engine.search("load user record json database", k=5)
        hit = next((r for r in results if r.chunk.symbol == "load_user"), None)
        self.assertIsNotNone(hit)
        relations = {(rel["relation"], rel["symbol"]) for rel in hit.related}
        self.assertIn(("caller", "authenticate"), relations)

    def test_callers_and_impact(self):
        engine = SearchEngine(self.root)
        callers = engine.callers("hash_password")
        self.assertTrue(any(c["symbol"] == "authenticate" for c in callers))
        impact = engine.impact("load_user")
        names = {r["name"] for r in impact}
        self.assertIn("authenticate", names)

    def test_outline(self):
        engine = SearchEngine(self.root)
        outline = engine.outline("app/auth.py")
        names = [o["name"] for o in outline]
        self.assertEqual(names, ["hash_password", "authenticate"])

    def test_incremental_reindex(self):
        # Second run with no changes: everything skipped.
        stats2 = Indexer(self.root, IndexConfig()).build()
        self.assertEqual(stats2.files_indexed, 0)
        self.assertGreaterEqual(stats2.files_skipped, 4)

        # Touch one file: only it re-indexes; search picks up the change.
        with open(os.path.join(self.root, "app", "storage.py"), "a") as f:
            f.write('\n\ndef purge_users():\n    """Delete every user record."""\n'
                    "    return None\n")
        stats3 = Indexer(self.root, IndexConfig()).build()
        self.assertEqual(stats3.files_indexed, 1)
        engine = SearchEngine(self.root)
        results = engine.search("delete every user record purge", k=5)
        self.assertTrue(any(r.chunk.symbol == "purge_users" for r in results))
        syms = engine.find_symbol("purge_users")
        self.assertEqual(len(syms), 1)

    def test_index_dir_is_transparent_json(self):
        idx_dir = os.path.join(self.root, ".codefabric")
        for name in ("manifest.json", "graph.json", "bm25.json", "vectors.json"):
            with open(os.path.join(idx_dir, name)) as f:
                json.load(f)  # must be valid JSON

    def test_gitignore_and_excludes_respected(self):
        with open(os.path.join(self.root, ".gitignore"), "w") as f:
            f.write("generated/\n")
        gen = os.path.join(self.root, "generated")
        os.makedirs(gen, exist_ok=True)
        with open(os.path.join(gen, "junk.py"), "w") as f:
            f.write("def junk():\n    pass\n")
        files = Indexer(self.root, IndexConfig()).discover_files()
        self.assertNotIn("generated/junk.py", files)


class CLITests(unittest.TestCase):
    def test_cli_smoke(self):
        import contextlib
        import io

        from codefabric.cli import main

        root = tempfile.mkdtemp(prefix="codefabric_cli_")
        try:
            with open(os.path.join(root, "m.py"), "w") as f:
                f.write('def ping():\n    """Reply pong."""\n    return "pong"\n')
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                main(["index", root])
                main(["search", "reply pong", "--path", root, "--k", "3"])
                main(["symbol", "ping", "--path", root])
                main(["stats", "--path", root])
            out = buf.getvalue()
            self.assertIn("indexed 1 files", out)
            self.assertIn("m.py", out)
            self.assertIn('"chunks"', out)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
