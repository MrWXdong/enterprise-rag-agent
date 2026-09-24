from pathlib import Path
from text_processor import read_text_file

class DocumentError(Exception):
    pass

class UnsupportedFileTypeError(DocumentError):
    pass

class DocumentParserError(DocumentError):
    pass
def parse_document(
            file_path:Path,
            encoding:str="utf-8"
) ->str:

    if not file_path.exists():
        raise FileNotFoundError(f'文件不存在:{file_path}')

    suffix=file_path.suffix.lower()

    if suffix ==".txt":
        return read_text_file(file_path,encoding)
    raise UnsupportedFileTypeError(f"暂不支持文件格式：{suffix or '无后缀'}")