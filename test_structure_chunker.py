import pytest
from structure_chunker import (
    detect_heading_type,
    is_heading,
    split_into_sections,
    split_section_by_size,
    split_section_by_sentences_size,
)


class TestHeadingDetection:
    """测试标题识别和标题类型"""

    def test_detect_chapter(self):
        assert is_heading("第一章 公司制度") is True
        assert detect_heading_type("第一章 公司制度") == "chapter"

    def test_detect_article(self):
        assert is_heading("第四条 住宿费用标准") is True
        assert detect_heading_type("第四条 住宿费用标准") == "article"

    def test_detect_question(self):
        assert is_heading("问题一：如何报销？") is True
        assert detect_heading_type("问题一：如何报销？") == "question"

    def test_detect_markdown(self):
        assert is_heading("## 数据管理制度") is True
        assert detect_heading_type("## 数据管理制度") == "markdown"

    def test_detect_normal_text(self):
        assert is_heading("住宿费用按照城市分类执行。") is False
        assert detect_heading_type("住宿费用按照城市分类执行。") is None


class TestSplitIntoSections:
    """测试章节切分"""

    def test_empty_text_returns_empty_list(self):
        assert split_into_sections("") == []

    def test_text_without_heading(self):
        sections = split_into_sections("这是一段普通正文。")

        assert len(sections) == 1
        assert sections[0]["chapter"] is None
        assert sections[0]["heading"] is None
        assert sections[0]["heading_type"] is None
        assert sections[0]["body"] == "这是一段普通正文。"

    def test_chapter_without_subheading(self):
        text = """第一章 公司制度
这是第一章的正文。
"""

        sections = split_into_sections(text)

        assert len(sections) == 1
        assert sections[0]["chapter"] == "第一章 公司制度"
        assert sections[0]["heading"] is None
        assert sections[0]["heading_type"] is None
        assert sections[0]["body"] == "这是第一章的正文。"

    def test_chapter_and_articles(self):
        text = """第一章 差旅制度
第一条 适用范围
本制度适用于全体员工。
第二条 住宿标准
一类城市每晚最高六百元。
第二章 采购制度
第三条 采购申请
采购前必须提交申请。
"""

        sections = split_into_sections(text)

        assert len(sections) == 3

        assert sections[0]["chapter"] == "第一章 差旅制度"
        assert sections[0]["heading"] == "第一条 适用范围"
        assert sections[0]["heading_type"] == "article"
        assert sections[0]["body"] == "本制度适用于全体员工。"

        assert sections[1]["chapter"] == "第一章 差旅制度"
        assert sections[1]["heading"] == "第二条 住宿标准"
        assert sections[1]["heading_type"] == "article"
        assert sections[1]["body"] == "一类城市每晚最高六百元。"

        assert sections[2]["chapter"] == "第二章 采购制度"
        assert sections[2]["heading"] == "第三条 采购申请"
        assert sections[2]["heading_type"] == "article"
        assert sections[2]["body"] == "采购前必须提交申请。"

    def test_question_sections(self):
        text = """问题一：可以修改城市分类吗？
不可以，城市分类由财务统一维护。
问题二：能否重复报销？
不可以，相同费用不能重复报销。
"""

        sections = split_into_sections(text)

        assert len(sections) == 2

        assert sections[0]["heading"] == "问题一：可以修改城市分类吗？"
        assert sections[0]["heading_type"] == "question"
        assert sections[0]["body"] == "不可以，城市分类由财务统一维护。"

        assert sections[1]["heading"] == "问题二：能否重复报销？"
        assert sections[1]["heading_type"] == "question"
        assert sections[1]["body"] == "不可以，相同费用不能重复报销。"

    def test_empty_body_is_skipped(self):
        text = """第一条 适用范围
第二条 住宿标准
一类城市每晚最高六百元。
"""

        sections = split_into_sections(text)

        assert len(sections) == 1
        assert sections[0]["heading"] == "第二条 住宿标准"
        assert sections[0]["body"] == "一类城市每晚最高六百元。"


class TestSplitSectionBySize:
    """测试结构 section 的长度控制切分"""

    def test_short_section_returns_one_chunk(self):
        section = {
            "chapter": "第一章 公司制度",
            "heading": "第四条 住宿费用标准",
            "heading_type": "article",
            "body": "一类城市每晚最高六百元。",
        }

        chunks = split_section_by_size(
            section,
            chunk_size=500,
            overlap=50,
        )

        assert len(chunks) == 1
        assert chunks[0]["chapter"] == "第一章 公司制度"
        assert chunks[0]["heading"] == "第四条 住宿费用标准"
        assert chunks[0]["heading_type"] == "article"
        assert chunks[0]["chunk_index_in_section"] == 0
        assert chunks[0]["content"].startswith("第四条 住宿费用标准")
        assert "一类城市每晚最高六百元" in chunks[0]["content"]

    def test_long_section_returns_multiple_chunks(self):
        section = {
            "chapter": "第一章 公司制度",
            "heading": "第四条 住宿费用标准",
            "heading_type": "article",
            "body": "住宿费用标准相关内容。" * 50,
        }

        chunks = split_section_by_size(
            section,
            chunk_size=100,
            overlap=10,
        )

        assert len(chunks) > 1
        assert [chunk["chunk_index_in_section"] for chunk in chunks] == list(
            range(len(chunks))
        )

        for chunk in chunks:
            assert chunk["chapter"] == "第一章 公司制度"
            assert chunk["heading"] == "第四条 住宿费用标准"
            assert chunk["heading_type"] == "article"
            assert chunk["content"].startswith("第四条 住宿费用标准")
            assert len(chunk["content"]) <= 100

    def test_uses_chapter_when_heading_is_none(self):
        section = {
            "chapter": "第一章 公司制度",
            "heading": None,
            "heading_type": None,
            "body": "这是章节正文。",
        }

        chunks = split_section_by_size(
            section,
            chunk_size=500,
            overlap=50,
        )

        assert len(chunks) == 1
        assert chunks[0]["heading"] is None
        assert chunks[0]["heading_type"] is None
        assert chunks[0]["content"].startswith("第一章 公司制度")

    def test_empty_body_returns_empty_list(self):
        section = {
            "chapter": "第一章 公司制度",
            "heading": None,
            "heading_type": None,
            "body": "   ",
        }

        assert split_section_by_size(section) == []

    def test_chunk_size_too_small_raises_error(self):
        section = {
            "chapter": "第一章 公司制度",
            "heading": "第四条 住宿费用标准",
            "heading_type": "article",
            "body": "正文内容",
        }

        with pytest.raises(ValueError):
            split_section_by_size(
                section,
                chunk_size=5,
                overlap=1,
            )

    def test_overlap_too_large_raises_error(self):
        section = {
            "chapter": "第一章 公司制度",
            "heading": "第四条 住宿费用标准",
            "heading_type": "article",
            "body": "正文内容",
        }

        with pytest.raises(ValueError):
            split_section_by_size(
                section,
                chunk_size=20,
                overlap=16,
            )


class TestSplitSectionBySentencesSize:
    """测试结构 section 的句子感知切分"""

    class TestSplitSectionBySentencesSize:
        """测试结构 section 的句子感知切分"""

        def test_preserve_heading_metadata(self):
            section = {
                "chapter": "第一章 公司制度",
                "heading": "第四条 住宿费用标准",
                "heading_type": "article",
                "body": "第一句话。第二句话！第三句话？",
            }

            chunks = split_section_by_sentences_size(
                section,
                chunk_size=30,
                overlap=5,
            )

            assert len(chunks) >= 1

            for chunk in chunks:
                assert chunk["chapter"] == "第一章 公司制度"
                assert chunk["heading"] == "第四条 住宿费用标准"
                assert chunk["heading_type"] == "article"
                assert chunk["content"].startswith("第四条 住宿费用标准")

                body_text = chunk["content"].split("\n", 1)[-1]

                assert body_text.endswith(
                    (
                        "。",
                        "！",
                        "？",
                        "；",
                        "…",
                        "。”",
                        "！”",
                        "？”",
                    )
                )

    def test_empty_body_returns_empty_list(self):
        section = {
            "chapter": "第一章 公司制度",
            "heading": None,
            "heading_type": None,
            "body": "   ",
        }

        assert split_section_by_sentences_size(section) == []

    def test_quoted_sentence_is_not_cut(self):
        section = {
            "chapter": "第一章 对话测试",
            "heading": "第一条 问候",
            "heading_type": "article",
            "body": "你好：“你好。”我们继续。",
        }

        chunks = split_section_by_sentences_size(
            section,
            chunk_size=100,
            overlap=10,
        )

        combined = "".join(chunk["content"] for chunk in chunks)

        assert "你好：“你好。”" in combined
