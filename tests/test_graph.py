import unittest

from codefabric.graph import GraphBuilder, KnowledgeGraph
from codefabric.models import Symbol


def sym(name, file_path, kind="function", calls=(), bases=(), qualified=None):
    return Symbol(
        name=name,
        qualified_name=qualified or name,
        kind=kind,
        file_path=file_path,
        start_line=1,
        end_line=5,
        calls=list(calls),
        bases=list(bases),
    )


class GraphBuilderTests(unittest.TestCase):
    def setUp(self):
        b = GraphBuilder()
        b.add_file(
            "app/db.py", "python",
            [sym("connect", "app/db.py"), sym("query", "app/db.py", calls=["connect"])],
            [],
        )
        b.add_file(
            "app/api.py", "python",
            [sym("handler", "app/api.py", calls=["query"]),
             sym("Base", "app/api.py", kind="class")],
            ["app.db"],
        )
        b.add_file(
            "app/views.py", "python",
            [sym("View", "app/views.py", kind="class", bases=["Base"]),
             sym("render", "app/views.py", calls=["handler"])],
            ["app.api"],
        )
        self.graph = b.finalize()

    def test_contains_edges(self):
        syms = self.graph.neighbors("file:app/db.py", "contains")
        self.assertIn("sym:app/db.py::connect", syms)

    def test_import_resolution(self):
        self.assertIn(
            "file:app/db.py",
            self.graph.neighbors("file:app/api.py", "imports"),
        )
        self.assertIn("file:app/api.py", self.graph.importers_of("app/db.py"))

    def test_call_edges_and_callers(self):
        callers = self.graph.callers_of("sym:app/db.py::query")
        self.assertIn("sym:app/api.py::handler", callers)
        callees = self.graph.callees_of("sym:app/db.py::query")
        self.assertIn("sym:app/db.py::connect", callees)

    def test_inheritance(self):
        subs = self.graph.subclasses_of("sym:app/api.py::Base")
        self.assertIn("sym:app/views.py::View", subs)

    def test_impact_transitive(self):
        impact = self.graph.impact("sym:app/db.py::connect")
        names = {node for node, _, _ in impact}
        self.assertIn("sym:app/db.py::query", names)      # direct caller
        self.assertIn("sym:app/api.py::handler", names)   # depth 2
        self.assertIn("sym:app/views.py::render", names)  # depth 3
        depths = {node: d for node, d, _ in impact}
        self.assertLess(depths["sym:app/db.py::query"],
                        depths["sym:app/views.py::render"])

    def test_roundtrip(self):
        g2 = KnowledgeGraph.from_dict(self.graph.to_dict())
        self.assertEqual(
            g2.callers_of("sym:app/db.py::query"),
            self.graph.callers_of("sym:app/db.py::query"),
        )
        self.assertEqual(set(g2.nodes), set(self.graph.nodes))

    def test_remove_file(self):
        self.graph.remove_file("app/api.py")
        self.assertNotIn("file:app/api.py", self.graph.nodes)
        self.assertNotIn("sym:app/api.py::handler",
                         self.graph.callers_of("sym:app/db.py::query"))


if __name__ == "__main__":
    unittest.main()
