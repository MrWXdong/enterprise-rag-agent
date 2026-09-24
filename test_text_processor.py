import pytest

from text_processor import split_text


class TestSplitText:
    """测试 split_text 函数"""

    def test_split_text_with_overlap(self):
        """测试相邻 chunk 保留 overlap"""
        result = split_text(
            "ABCDEFGHIJ",
            chunk_size=6,
            overlap=3,
        )

        assert result == ["ABCDEF", "DEFGHI", "GHIJ"]

    def test_split_text_without_overlap(self):
        """测试 overlap 为 0 时文本不重叠"""
        result = split_text(
            "ABCDEFGHIJ",
            chunk_size=6,
            overlap=0,
        )

        assert result == ["ABCDEF", "GHIJ"]

    def test_chunk_size_larger_than_text(self):
        """测试 chunk_size 大于文本长度"""
        result = split_text(
            "hello",
            chunk_size=100,
            overlap=10,
        )

        assert result == ["hello"]

    def test_empty_text_returns_empty_list(self):
        """测试空文本返回空列表"""
        result = split_text("", chunk_size=100, overlap=10)

        assert result == []

    def test_invalid_chunk_size_raises_error(self):
        """测试 chunk_size 小于等于 0"""
        with pytest.raises(ValueError):
            split_text("hello", chunk_size=0, overlap=0)

    def test_negative_overlap_raises_error(self):
        """测试 overlap 小于 0"""
        with pytest.raises(ValueError):
            split_text("hello", chunk_size=10, overlap=-1)

    def test_overlap_not_smaller_than_chunk_size(self):
        """测试 overlap 大于等于 chunk_size"""
        with pytest.raises(ValueError):
            split_text("hello", chunk_size=10, overlap=10)