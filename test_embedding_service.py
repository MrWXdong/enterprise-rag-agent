from embedding_service import embed_texts, get_embedding_model


class TestEmbeddingService:
    """测试 Embedding 服务"""

    def test_empty_texts_returns_empty_list(self):
        assert embed_texts([]) == []

    def test_embedding_dimension(self):
        vectors = embed_texts(["测试文本"])
        expected_dimension = get_embedding_model().get_embedding_dimension()

        assert len(vectors) == 1
        assert len(vectors[0]) == expected_dimension

    def test_related_texts_are_closer(self):
        vectors = embed_texts(
            [
                "住宿费怎么报销",
                "酒店费用如何申请",
                "电脑采购流程",
            ]
        )

        related_score = sum(a * b for a, b in zip(vectors[0], vectors[1]))
        unrelated_score = sum(a * b for a, b in zip(vectors[0], vectors[2]))

        assert related_score > unrelated_score
