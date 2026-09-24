from pathlib import Path

def read_text_file(file_path:Path,
                   encoding:str="utf-8"
 )   ->str:
    return file_path.read_text(encoding=encoding)

def split_text(
        text: str,
        chunk_size:int =500,
        overlap:int = 50,
)->list[str]:

    if chunk_size<=0:
        raise ValueError("chunk_size必须大于0")
    if overlap<0:
        raise ValueError ("overlap不能小于0")
    if chunk_size<=overlap:
        raise ValueError("chunk_size必须大于overlap")
    chunks:list[str]=[]
    step = chunk_size - overlap
    start = 0

    while start<len(text):
        end = min(start+chunk_size,len(text))
        chunk=text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end>=len(text):
            break
        start+=step
    return chunks
