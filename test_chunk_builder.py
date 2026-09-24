from chunk_builder import build_chunks


class TestBuildChunks:
    """测试 build_chunks 函数"""

    def test_build_chunks_basic(self):
        """测试基本切分和结构化字段"""
        chunks = build_chunks(
            "doc-1",
            "ABCDEFGHIJ",
            chunk_size=6,
            overlap=3,
        )

        assert len(chunks) == 3
        assert [chunk["content"] for chunk in chunks] == [
            "ABCDEF",
            "DEFGHI",
            "GHIJ",
        ]
        assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]

    def test_chunk_ids(self):
        """测试 chunk_id 的生成规则"""
        chunks = build_chunks(
            "doc-1",
            "ABCDEFGHIJ",
            chunk_size=6,
            overlap=3,
        )

        assert chunks[0]["chunk_id"] == "doc-1_chunk_000000"
        assert chunks[1]["chunk_id"] == "doc-1_chunk_000001"
        assert chunks[2]["chunk_id"] == "doc-1_chunk_000002"

    def test_previous_and_next_links(self):
        """测试相邻 chunk 的前后关系"""
        chunks = build_chunks(
            "doc-1",
            "ABCDEFGHIJ",
            chunk_size=6,
            overlap=3,
        )

        assert chunks[0]["previous_chunk_id"] is None
        assert chunks[0]["next_chunk_id"] == "doc-1_chunk_000001"

        assert chunks[1]["previous_chunk_id"] == "doc-1_chunk_000000"
        assert chunks[1]["next_chunk_id"] == "doc-1_chunk_000002"

        assert chunks[2]["previous_chunk_id"] == "doc-1_chunk_000001"
        assert chunks[2]["next_chunk_id"] is None

    def test_empty_text(self):
        """测试空文本返回空列表"""
        chunks = build_chunks("doc-1", "")

        assert chunks == []

    def test_different_documents_have_different_ids(self):
        """测试不同文档生成的 chunk_id 不会冲突"""
        chunks_a = build_chunks("doc-a", "hello")
        chunks_b = build_chunks("doc-b", "hello")

        assert chunks_a[0]["chunk_id"] == "doc-a_chunk_000000"
        assert chunks_b[0]["chunk_id"] == "doc-b_chunk_000000"
        assert chunks_a[0]["chunk_id"] != chunks_b[0]["chunk_id"]
