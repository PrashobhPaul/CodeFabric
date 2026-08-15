import unittest

from codefabric.chunking.python_chunker import PythonChunker
from codefabric.chunking.heuristic_chunker import HeuristicChunker

PY_SOURCE = '''"""Module docstring."""
import os
from collections import Counter


def top_level(a, b):
    """Add two numbers."""
    return a + b


class Greeter:
    """Says hello."""

    prefix = "Hello"

    def greet(self, name):
        """Greet someone."""
        return f"{self.prefix}, {name}!"

    def shout(self, name):
        return self.greet(name).upper()


class LoudGreeter(Greeter):
    def greet(self, name):
        return super().greet(name).upper()
'''

JS_SOURCE = '''import { thing } from "./util";
const helper = require("./helper");

export function processOrder(order) {
  if (!order.id) {
    throw new Error("no id");
  }
  return order;
}

export class OrderService {
  constructor(repo) {
    this.repo = repo;
  }
}

const fmt = (x) => x.toString();
'''


class PythonChunkerTests(unittest.TestCase):
    def setUp(self):
        self.chunker = PythonChunker()
        self.chunks, self.symbols, self.imports = self.chunker.parse(
            "pkg/mod.py", PY_SOURCE
        )

    def test_symbols_extracted(self):
        names = {s.qualified_name: s for s in self.symbols}
        self.assertIn("top_level", names)
        self.assertIn("Greeter", names)
        self.assertIn("Greeter.greet", names)
        self.assertIn("LoudGreeter", names)
        self.assertEqual(names["Greeter.greet"].kind, "method")
        self.assertEqual(names["top_level"].kind, "function")
        self.assertEqual(names["LoudGreeter"].bases, ["Greeter"])

    def test_docstrings_and_signatures(self):
        greet = next(s for s in self.symbols if s.qualified_name == "Greeter.greet")
        self.assertEqual(greet.docstring, "Greet someone.")
        self.assertTrue(greet.signature.startswith("def greet"))

    def test_calls_captured(self):
        shout = next(s for s in self.symbols if s.qualified_name == "Greeter.shout")
        self.assertIn("greet", shout.calls)

    def test_imports(self):
        self.assertIn("os", self.imports)
        self.assertIn("collections", self.imports)

    def test_chunk_boundaries_cover_definitions(self):
        sym_chunks = {c.symbol for c in self.chunks if c.symbol}
        self.assertIn("top_level", sym_chunks)
        self.assertIn("Greeter", sym_chunks)
        # No chunk cuts through top_level's body.
        tl = next(c for c in self.chunks if c.symbol == "top_level")
        self.assertIn("return a + b", tl.text)

    def test_large_class_split_into_methods(self):
        methods = "\n\n".join(
            f'    def m{i}(self):\n        """doc {i}"""\n        return {i} '
            + "+ 1" * 120
            for i in range(10)
        )
        src = f"class Big:\n    \"\"\"Big class.\"\"\"\n\n{methods}\n"
        chunks, symbols, _ = PythonChunker(max_chunk_chars=800).parse("big.py", src)
        kinds = {c.kind for c in chunks}
        self.assertIn("class", kinds)  # header chunk
        method_chunks = [c for c in chunks if c.kind == "method"]
        self.assertGreaterEqual(len(method_chunks), 5)
        # Method chunks carry class context.
        self.assertTrue(any("# context: class Big" in c.text for c in method_chunks))

    def test_syntax_error_returns_empty(self):
        chunks, symbols, imports = self.chunker.parse("bad.py", "def broken(:\n")
        self.assertEqual((chunks, symbols, imports), ([], [], []))


class HeuristicChunkerTests(unittest.TestCase):
    def setUp(self):
        self.chunker = HeuristicChunker()
        self.chunks, self.symbols, self.imports = self.chunker.parse(
            "src/order.js", JS_SOURCE, "javascript"
        )

    def test_symbols_found(self):
        names = {s.name for s in self.symbols}
        self.assertIn("processOrder", names)
        self.assertIn("OrderService", names)
        self.assertIn("fmt", names)

    def test_block_boundaries(self):
        fn = next(c for c in self.chunks if c.symbol == "processOrder")
        self.assertIn('throw new Error("no id")', fn.text)
        self.assertTrue(fn.text.rstrip().endswith("}"))

    def test_imports_extracted(self):
        self.assertIn("./util", self.imports)
        self.assertIn("./helper", self.imports)

    def test_unknown_language_windows(self):
        text = "\n".join(f"line {i}" for i in range(200))
        chunks, symbols, _ = self.chunker.parse("notes.txt", text, "text")
        self.assertFalse(symbols)
        self.assertGreater(len(chunks), 1)
        # Full coverage: first chunk starts at 1, last ends at 200.
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[-1].end_line, 200)


if __name__ == "__main__":
    unittest.main()
