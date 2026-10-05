from chunk_builder import (
    build_chunks,
    build_structure_chunks,
    build_structure_sentence_chunks,
)


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


class TestBuildStructureChunks:
    """测试结构感知切分"""

    def test_preserve_heading_metadata(self):
        text = """第一章 差旅制度
第一条 适用范围
本制度适用于全体员工。
第二条 住宿标准
一类城市每晚最高六百元。
"""

        chunks = build_structure_chunks(
            "doc-1",
            text,
            chunk_size=500,
            overlap=50,
        )

        assert len(chunks) == 2

        assert chunks[0]["chapter"] == "第一章 差旅制度"
        assert chunks[0]["heading"] == "第一条 适用范围"
        assert chunks[0]["heading_type"] == "article"
        assert chunks[0]["content"].startswith("第一条 适用范围")

        assert chunks[1]["chapter"] == "第一章 差旅制度"
        assert chunks[1]["heading"] == "第二条 住宿标准"
        assert chunks[1]["heading_type"] == "article"

    def test_chunk_links_are_correct(self):
        text = """第一章 差旅制度
第一条 适用范围
本制度适用于全体员工。
第二条 住宿标准
一类城市每晚最高六百元。
"""

        chunks = build_structure_chunks(
            "doc-1",
            text,
            chunk_size=500,
            overlap=50,
        )

        assert chunks[0]["chunk_id"] == "doc-1_chunk_000000"
        assert chunks[0]["previous_chunk_id"] is None
        assert chunks[0]["next_chunk_id"] == "doc-1_chunk_000001"

        assert chunks[1]["chunk_id"] == "doc-1_chunk_000001"
        assert chunks[1]["previous_chunk_id"] == "doc-1_chunk_000000"
        assert chunks[1]["next_chunk_id"] is None

    def test_long_section_creates_multiple_chunks(self):
        text = """第一章 差旅制度
第四条 住宿费用标准
""" + "住宿费用标准相关内容。" * 50

        chunks = build_structure_chunks(
            "doc-1",
            text,
            chunk_size=100,
            overlap=10,
        )

        assert len(chunks) > 1

        for chunk in chunks:
            assert chunk["chapter"] == "第一章 差旅制度"
            assert chunk["heading"] == "第四条 住宿费用标准"
            assert chunk["heading_type"] == "article"
            assert chunk["content"].startswith("第四条 住宿费用标准")

    def test_empty_text_returns_empty_list(self):
        chunks = build_structure_chunks(
            "doc-1",
            "",
            chunk_size=500,
            overlap=50,
        )

        assert chunks == []


class TestBuildStructureSentenceChunks:
    """测试结构加句子感知的最终 chunks"""

    def test_preserve_metadata_and_links(self):
        text = """第一章 差旅制度
第一条 适用范围
本制度适用于全体员工。
第二条 住宿标准
一类城市每晚最高六百元。
"""

        chunks = build_structure_sentence_chunks(
            "doc-1",
            text,
            chunk_size=500,
            overlap=50,
        )

        assert len(chunks) == 2
        assert chunks[0]["chapter"] == "第一章 差旅制度"
        assert chunks[0]["heading"] == "第一条 适用范围"
        assert chunks[0]["heading_type"] == "article"
        assert chunks[0]["chunk_id"] == "doc-1_chunk_000000"
        assert chunks[0]["next_chunk_id"] == "doc-1_chunk_000001"

        assert chunks[1]["previous_chunk_id"] == "doc-1_chunk_000000"
        assert chunks[1]["next_chunk_id"] is None

    def test_chunk_content_ends_with_punctuation_when_possible(self):
        text = """第一章 差旅制度
第四条 住宿费用标准
第一句话。第二句话！第三句话？第四句话。
"""

        chunks = build_structure_sentence_chunks(
            "doc-1",
            text,
            chunk_size=50,
            overlap=5,
        )

        assert len(chunks) > 1
        assert (
            chunks[0]["content"].endswith("。")
            or chunks[0]["content"].endswith("！")
            or chunks[0]["content"].endswith("？")
        )
