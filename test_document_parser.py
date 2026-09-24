import tempfile
from pathlib import Path

import pytest

from document_parser import (
    UnsupportedFileTypeError,
    parse_document,
)


class TestParseDocument:
    """测试 parse_document 函数"""

    def test_parse_txt_file(self):
        """测试正常解析 TXT 文件"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.txt"
            file_path.write_text("hello", encoding="utf-8")

            result = parse_document(file_path)

            assert result == "hello"

    def test_parse_uppercase_txt_extension(self):
        """测试大写 TXT 后缀"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "TEST.TXT"
            file_path.write_text("hello", encoding="utf-8")

            result = parse_document(file_path)

            assert result == "hello"

    def test_parse_gbk_txt_file(self):
        """测试指定 GBK 编码读取 TXT"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "gbk.txt"
            file_path.write_bytes("中文内容".encode("gbk"))

            result = parse_document(file_path, encoding="gbk")

            assert result == "中文内容"

    def test_parse_missing_file_raises_error(self):
        """测试文件不存在时抛出 FileNotFoundError"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "missing.txt"

            with pytest.raises(FileNotFoundError):
                parse_document(file_path)

    def test_parse_unsupported_file_type_raises_error(self):
        """测试不支持的文件类型抛出 UnsupportedFileTypeError"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.zip"
            file_path.write_bytes(b"zip content")

            with pytest.raises(UnsupportedFileTypeError):
                parse_document(file_path)