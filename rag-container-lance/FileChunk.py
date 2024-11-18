class FileChunk:
    filename: str = None
    filepath: str = None
    chunk: str = None

    def __init__(self, filename: str, filepath: str, chunk: str):
        self.filename = filename
        self.filepath = filepath
        self.chunk = chunk
