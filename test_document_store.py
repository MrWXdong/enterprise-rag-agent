import json
import tempfile
from pathlib import Path

import pytest

from document_store import (
    calculate_sha256,
    save_document,
    list_documents,
    update_document_metadata,
    save_chunks,
    find_document_by_hash,
    add_document_alias,
)


class TestCalculateSha256:
    """测试 calculate_sha256 函数"""

    def test_calculate_sha256_normal(self):
        """测试正常输入：计算字节内容的SHA256哈希值"""
        content = b"Hello, World!"
        result = calculate_sha256(content)
        # 验证返回的是64字符的十六进制字符串
        assert len(result) == 64
        assert result.isalnum()

    def test_calculate_sha256_empty_content(self):
        """测试边界值：空字节内容"""
        content = b""
        result = calculate_sha256(content)
        assert len(result) == 64
        # 空字符串的SHA256哈希值是已知的
        assert (
            result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_calculate_sha256_unicode_content(self):
        """测试Unicode内容"""
        content = "中文测试内容".encode("utf-8")
        result = calculate_sha256(content)
        assert len(result) == 64

    def test_calculate_sha256_large_content(self):
        """测试大文件内容"""
        content = b"A" * 1024 * 1024  # 1MB
        result = calculate_sha256(content)
        assert len(result) == 64

    def test_calculate_sha256_deterministic(self):
        """测试相同输入产生相同输出"""
        content = b"test content"
        result1 = calculate_sha256(content)
        result2 = calculate_sha256(content)
        assert result1 == result2

    def test_calculate_sha256_different_inputs(self):
        """测试不同输入产生不同输出"""
        result1 = calculate_sha256(b"content1")
        result2 = calculate_sha256(b"content2")
        assert result1 != result2


class TestSaveDocument:
    """测试 save_document 函数"""

    def test_save_document_normal(self):
        """测试正常保存文档"""
        with tempfile.TemporaryDirectory() as temp_dir:
            document_dir = Path(temp_dir)
            original_name = "test_document.pdf"
            content = b"PDF content here"
            file_hash = calculate_sha256(content)

            result = save_document(document_dir, original_name, content, file_hash)

            # 验证返回的元数据
            assert "document_id" in result
            assert result["original_name"] == "test_document.pdf"
            assert result["stored_file_name"] == "original.pdf"
            assert result["extension"] == ".pdf"
            assert result["file_hash"] == file_hash
            assert result["status"] == "upload"
            assert "created_at" in result

            # 验证文件已保存
            doc_id = result["document_id"]
            saved_file = document_dir / doc_id / "original.pdf"
            assert saved_file.exists()
            assert saved_file.read_bytes() == content

            # 验证元数据文件已保存
            metadata_file = document_dir / doc_id / "metadata.json"
            assert metadata_file.exists()
            saved_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            assert saved_metadata == result

    def test_save_document_with_uppercase_extension(self):
        """测试大写扩展名会被转换为小写"""
        with tempfile.TemporaryDirectory() as temp_dir:
            document_dir = Path(temp_dir)
            original_name = "doc.TXT"
            content = b"text content"
            file_hash = calculate_sha256(content)

            result = save_document(document_dir, original_name, content, file_hash)

            assert result["extension"] == ".txt"
            assert result["stored_file_name"] == "original.txt"

    def test_save_document_no_extension(self):
        """测试无扩展名的文件名"""
        with tempfile.TemporaryDirectory() as temp_dir:
            document_dir = Path(temp_dir)
            original_name = "noextensionfile"
            content = b"content without extension"
            file_hash = calculate_sha256(content)

            result = save_document(document_dir, original_name, content, file_hash)

            assert result["extension"] == ""
            assert result["stored_file_name"] == "original"

    def test_save_document_with_path_separator(self):
        """测试文件名包含路径分隔符时的安全性"""
        with tempfile.TemporaryDirectory() as temp_dir:
            document_dir = Path(temp_dir)
            original_name = "../../../etc/passwd"
            content = b"malicious content"
            file_hash = calculate_sha256(content)

            result = save_document(document_dir, original_name, content, file_hash)

            # 验证只保存了文件名部分，不包含路径
            assert result["original_name"] == "passwd"
            assert result["stored_file_name"] == "original"
            assert ".." not in result["original_name"]

    def test_save_document_various_extensions(self):
        """测试各种扩展名"""
        extensions = [
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".jpg",
            ".png",
            ".gif",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            for ext in extensions:
                document_dir = Path(temp_dir)
                original_name = f"file{ext}"
                content = f"content for {ext}".encode()
                file_hash = calculate_sha256(content)

                result = save_document(document_dir, original_name, content, file_hash)

                assert result["extension"] == ext.lower()
                assert result["stored_file_name"] == f"original{ext.lower()}"

    def test_save_document_empty_content(self):
        """测试保存空内容"""
        with tempfile.TemporaryDirectory() as temp_dir:
            document_dir = Path(temp_dir)
            original_name = "empty.pdf"
            content = b""
            file_hash = calculate_sha256(content)

            result = save_document(document_dir, original_name, content, file_hash)

            doc_id = result["document_id"]
            saved_file = document_dir / doc_id / "original.pdf"
            assert saved_file.exists()
            assert saved_file.read_bytes() == b""

    def test_save_document_creates_unique_ids(self):
        """测试每次调用生成唯一的document_id"""
        with tempfile.TemporaryDirectory() as temp_dir:
            document_dir = Path(temp_dir)
            content = b"test"
            file_hash = calculate_sha256(content)

            result1 = save_document(document_dir, "file1.pdf", content, file_hash)
            result2 = save_document(document_dir, "file2.pdf", content, file_hash)

            assert result1["document_id"] != result2["document_id"]

    def test_save_document_creates_directory(self):
        """测试文档目录是否正确创建"""
        with tempfile.TemporaryDirectory() as temp_dir:
            document_dir = Path(temp_dir) / "documents"
            content = b"test"
            file_hash = calculate_sha256(content)

            result = save_document(document_dir, "test.pdf", content, file_hash)

            doc_dir = document_dir / result["document_id"]
            assert doc_dir.exists()
            assert doc_dir.is_dir()


class TestListDocuments:
    """测试 list_documents 函数"""

    def test_list_documents_empty_directory(self):
        """测试空目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)
            result = list_documents(documents_dir)
            assert result == []

    def test_list_documents_nonexistent_directory(self):
        """测试不存在的目录"""
        documents_dir = Path("/nonexistent/path/to/documents")
        result = list_documents(documents_dir)
        assert result == []

    def test_list_documents_with_documents(self):
        """测试列出包含文档的目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)

            # 创建两个文档
            content1 = b"doc1 content"
            hash1 = calculate_sha256(content1)
            result1 = save_document(documents_dir, "doc1.pdf", content1, hash1)

            content2 = b"doc2 content"
            hash2 = calculate_sha256(content2)
            result2 = save_document(documents_dir, "doc2.txt", content2, hash2)

            # 列出文档
            documents = list_documents(documents_dir)

            assert len(documents) == 2
            # 验证返回的文档包含必要的字段
            for doc in documents:
                assert "document_id" in doc
                assert "original_name" in doc
                assert "file_hash" in doc
                assert "status" in doc
                assert "created_at" in doc

    def test_list_documents_sorted(self):
        """测试文档列表是否按目录名排序"""
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)

            # 创建多个文档
            for i in range(5):
                content = f"doc{i} content".encode()
                hash_val = calculate_sha256(content)
                save_document(documents_dir, f"doc{i}.pdf", content, hash_val)

            documents = list_documents(documents_dir)

            assert len(documents) == 5
            # 验证列表按目录名排序（字典序）
            doc_ids = [doc["document_id"] for doc in documents]
            assert doc_ids == sorted(doc_ids)

    def test_list_documents_ignores_files(self):
        """测试忽略非目录文件"""
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)

            # 创建文档
            content = b"doc content"
            hash_val = calculate_sha256(content)
            save_document(documents_dir, "doc.pdf", content, hash_val)

            # 在根目录添加一个普通文件
            (documents_dir / "random_file.txt").write_text("not a document")

            documents = list_documents(documents_dir)

            assert len(documents) == 1
            assert documents[0]["original_name"] == "doc.pdf"

    def test_list_documents_ignores_missing_metadata(self):
        """测试忽略缺少元数据文件的目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)

            # 创建文档
            content = b"doc content"
            hash_val = calculate_sha256(content)
            save_document(documents_dir, "doc.pdf", content, hash_val)

            # 创建一个没有metadata.json的目录
            invalid_dir = documents_dir / "invalid_doc"
            invalid_dir.mkdir()
            (invalid_dir / "some_file.txt").write_text("content")

            documents = list_documents(documents_dir)

            # 应该只返回有效文档，忽略无效目录
            assert len(documents) == 1

    def test_list_documents_metadata_integrity(self):
        """测试元数据完整性"""
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)

            original_name = "test_document.pdf"
            content = b"test content for integrity check"
            file_hash = calculate_sha256(content)

            saved_metadata = save_document(
                documents_dir, original_name, content, file_hash
            )
            documents = list_documents(documents_dir)

            assert len(documents) == 1
            listed_metadata = documents[0]

            # 验证所有字段都正确返回
            assert listed_metadata["document_id"] == saved_metadata["document_id"]
            assert listed_metadata["original_name"] == saved_metadata["original_name"]
            assert (
                listed_metadata["stored_file_name"]
                == saved_metadata["stored_file_name"]
            )
            assert listed_metadata["extension"] == saved_metadata["extension"]
            assert listed_metadata["file_hash"] == saved_metadata["file_hash"]
            assert listed_metadata["status"] == saved_metadata["status"]
            assert listed_metadata["created_at"] == saved_metadata["created_at"]

    def test_update_document_metadata_normal(self):
        # 准备：使用临时目录，不污染真实 data/
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)

            content = b"hello"
            file_hash = calculate_sha256(content)

            original_metadata = save_document(
                documents_dir,
                "test.txt",
                content,
                file_hash,
            )

            # 执行：更新已有文档的 metadata
            updated_metadata = update_document_metadata(
                documents_dir,
                original_metadata["document_id"],
                {
                    "status": "processed",
                    "character_count": 5,
                    "chunk_count": 1,
                },
            )

            # 断言：新字段已经更新
            assert updated_metadata["status"] == "processed"
            assert updated_metadata["character_count"] == 5
            assert updated_metadata["chunk_count"] == 1

            # 断言：旧字段没有丢失
            assert updated_metadata["document_id"] == original_metadata["document_id"]
            assert (
                updated_metadata["original_name"] == original_metadata["original_name"]
            )
            assert updated_metadata["file_hash"] == original_metadata["file_hash"]

            # 断言：更新结果真的写入了磁盘
            metadata_path = (
                documents_dir / original_metadata["document_id"] / "metadata.json"
            )

            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            assert saved_metadata["status"] == "processed"
            assert saved_metadata["character_count"] == 5
            assert saved_metadata["chunk_count"] == 1

    def test_update_document_metadata_missing_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)

            with pytest.raises(FileNotFoundError):
                update_document_metadata(
                    documents_dir,
                    "missing-document-id",
                    {
                        "status": "processed",
                    },
                )

    class TestSaveChunks:
        """测试 save_chunks 函数"""

        def test_save_chunks_normal(self):
            """测试正常保存 chunks 并返回数量"""
            with tempfile.TemporaryDirectory() as temp_dir:
                documents_dir = Path(temp_dir)
                content = b"hello"
                file_hash = calculate_sha256(content)

                metadata = save_document(
                    documents_dir,
                    "test.txt",
                    content,
                    file_hash,
                )

                chunks = [
                    {
                        "chunk_id": f"{metadata['document_id']}_chunk_000000",
                        "document_id": metadata["document_id"],
                        "chunk_index": 0,
                        "content": "hello",
                        "heading": None,
                        "section_id": None,
                        "previous_chunk_id": None,
                        "next_chunk_id": None,
                    }
                ]

                chunk_count = save_chunks(
                    documents_dir,
                    metadata["document_id"],
                    chunks,
                )

                assert chunk_count == 1

                chunks_path = documents_dir / metadata["document_id"] / "chunks.json"
                assert chunks_path.exists()

                saved_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

                assert saved_chunks == chunks

        def test_save_chunks_missing_document_raises(self):
            """测试文档目录不存在时抛出 FileNotFoundError"""
            with tempfile.TemporaryDirectory() as temp_dir:
                documents_dir = Path(temp_dir)

                with pytest.raises(FileNotFoundError):
                    save_chunks(
                        documents_dir,
                        "missing-document-id",
                        [],
                    )


class TestDocumentDeduplication:
    """测试文件哈希去重和文件名别名"""

    def test_find_document_by_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)
            content = b"same content"
            file_hash = calculate_sha256(content)

            original = save_document(
                documents_dir,
                "original.txt",
                content,
                file_hash,
            )

            found = find_document_by_hash(
                documents_dir,
                file_hash,
            )

            assert found is not None
            assert found["document_id"] == original["document_id"]
            assert found["original_name"] == "original.txt"

    def test_find_document_by_hash_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)

            found = find_document_by_hash(
                documents_dir,
                "missing-hash",
            )

            assert found is None

    def test_add_document_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)
            content = b"same content"
            file_hash = calculate_sha256(content)

            original = save_document(
                documents_dir,
                "original.txt",
                content,
                file_hash,
            )

            updated = add_document_alias(
                documents_dir,
                original["document_id"],
                "another-name.txt",
            )

            assert updated["original_name"] == "original.txt"
            assert updated["aliases"] == ["another-name.txt"]

    def test_add_document_alias_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)
            content = b"same content"
            file_hash = calculate_sha256(content)

            original = save_document(
                documents_dir,
                "original.txt",
                content,
                file_hash,
            )

            add_document_alias(
                documents_dir,
                original["document_id"],
                "another-name.txt",
            )

            updated = add_document_alias(
                documents_dir,
                original["document_id"],
                "another-name.txt",
            )

            assert updated["aliases"] == ["another-name.txt"]

    def test_original_name_is_not_added_as_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)
            content = b"same content"
            file_hash = calculate_sha256(content)

            original = save_document(
                documents_dir,
                "original.txt",
                content,
                file_hash,
            )

            updated = add_document_alias(
                documents_dir,
                original["document_id"],
                "original.txt",
            )

            assert updated["aliases"] == []
