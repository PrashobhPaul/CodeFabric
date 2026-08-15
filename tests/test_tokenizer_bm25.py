import unittest

from codefabric.tokenizer import split_identifier, tokenize
from codefabric.retrieval.bm25 import BM25Index
from codefabric.retrieval.fusion import reciprocal_rank_fusion


class TokenizerTests(unittest.TestCase):
    def test_split_camel_and_snake(self):
        self.assertEqual(split_identifier("getUserById"),
                         ["get", "user", "by", "id"])
        self.assertEqual(split_identifier("parse_config_file"),
                         ["parse", "config", "file"])
        self.assertEqual(split_identifier("HTTPServerError"),
                         ["http", "server", "error"])

    def test_tokenize_emits_original_and_subwords(self):
        toks = tokenize("def getUserById(user_id):")
        self.assertIn("getuserbyid", toks)
        self.assertIn("user", toks)
        self.assertIn("id", toks)
        # stopword 'def' dropped
        self.assertNotIn("def", toks)


class BM25Tests(unittest.TestCase):
    def _index(self):
        idx = BM25Index()
        idx.add("doc_auth", tokenize("def authenticate_user(password, token): verify login"))
        idx.add("doc_db", tokenize("def connect_database(url): open connection pool"))
        idx.add("doc_misc", tokenize("print hello world example"))
        idx.finalize()
        return idx

    def test_ranking_relevance(self):
        idx = self._index()
        results = idx.search(tokenize("user authentication login"), k=3)
        self.assertTrue(results)
        self.assertEqual(results[0][0], "doc_auth")

    def test_roundtrip_persistence(self):
        idx = self._index()
        idx2 = BM25Index.from_dict(idx.to_dict())
        r1 = idx.search(tokenize("database connection"), k=2)
        r2 = idx2.search(tokenize("database connection"), k=2)
        self.assertEqual(r1, r2)
        self.assertEqual(r1[0][0], "doc_db")

    def test_empty_query(self):
        self.assertEqual(self._index().search([], k=5), [])


class FusionTests(unittest.TestCase):
    def test_rrf_prefers_agreement(self):
        rankings = {
            "bm25": [("a", 10.0), ("b", 5.0), ("c", 1.0)],
            "dense": [("b", 0.9), ("a", 0.8), ("d", 0.7)],
        }
        fused = reciprocal_rank_fusion(rankings)
        ids = [doc for doc, _, _ in fused]
        # a and b (in both rankings) must outrank c and d.
        self.assertLess(ids.index("a"), ids.index("c"))
        self.assertLess(ids.index("b"), ids.index("d"))
        srcs = {doc: s for doc, _, s in fused}
        self.assertEqual(sorted(srcs["a"]), ["bm25", "dense"])
        self.assertEqual(srcs["c"], ["bm25"])


if __name__ == "__main__":
    unittest.main()
