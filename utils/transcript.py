"""Transcript utilities for capturing and recording demo outputs.

Ground Rule 6: Every demo writes a transcript to transcripts/<name>.txt
(stdout is teed to the file).
"""

import sys
from contextlib import contextmanager
from pathlib import Path
import config


class TeeStream:
    """Tees output to both the original stream and a transcript file."""

    def __init__(self, original_stream, file_obj):
        self.original_stream = original_stream
        self.file_obj = file_obj

    def write(self, data):
        self.original_stream.write(data)
        self.file_obj.write(data)

    def flush(self):
        self.original_stream.flush()
        self.file_obj.flush()


@contextmanager
def tee_transcript(name: str):
    """Context manager that mirrors stdout to transcripts/<name>.txt (UTF-8, overwrite each run)."""
    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = name if name.endswith(".txt") else f"{name}.txt"
    filepath = config.TRANSCRIPTS_DIR / filename

    original_stdout = sys.stdout
    with open(filepath, "w", encoding="utf-8") as f:
        tee = TeeStream(original_stdout, f)
        sys.stdout = tee
        try:
            yield filepath
        finally:
            sys.stdout.flush()
            sys.stdout = original_stdout


# Backward-compatible alias
record_transcript = tee_transcript
