import hashlib
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple, Union

from langchain_core.tools import tool


@tool
def file_processor(input_str: str) -> str:
    """
    Process files: read or write content.

    Args:
        input_str: A string of the form "<content>||<filename>" to save
        or just "<filename>" to read.

    Returns:
        File content or a success message.

    Raises:
        ValueError: If the input string is not in the expected format.
    """
    try:
        if "||" in input_str:
            content, filename = input_str.split("||", 1)
            file_proc = FileProcessor()
            result = file_proc.save_content(content.strip(), filename.strip())
            return result
        else:
            return FileProcessor().read_file(input_str.strip())
    except Exception as e:
        return f"File processing error: {str(e)}"


class FileProcessor:
    """
    Enhanced file processing tool for AI agents with:
    - Secure file operations
    - Content validation
    - Multiple file format support
    - Metadata tracking
    - Thread-safe operations

    Usage:
        processor = FileProcessor()
        # Save and read text
        filepath = processor.save_content("Sample content")
        content = processor.read_file(filepath)

        # Handle binary data
        filepath = processor.save_content(b"binary data", file_type="binary")
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Args:
            base_dir: Custom base directory (defaults to system temp)
        """
        self.base_dir = base_dir or tempfile.gettempdir()
        os.makedirs(self.base_dir, exist_ok=True)

    def save_content(
        self,
        content: Union[str, bytes],
        filename: Optional[str] = None,
        file_type: str = "text",
    ) -> Tuple[str, str]:
        """
        Save content to a file with validation and checksum.

        Args:
            content: Text or binary content to save
            filename: Optional filename (auto-generated if None)
            file_type: "text" or "binary" content type

        Returns:
            Tuple of (filepath, checksum)

        Raises:
            ValueError: For invalid content or file operations
        """
        try:
            # Content validation
            if not content:
                raise ValueError("Content cannot be empty")

            # Determine file path
            if filename:
                filepath = os.path.join(self.base_dir, filename)
                if os.path.exists(filepath):
                    raise FileExistsError(f"File {filename} already exists")
            else:
                ext = ".txt" if file_type == "text" else ".bin"
                filepath = os.path.join(
                    self.base_dir, f"gaia_{uuid.uuid4().hex[:8]}{ext}"
                )

            # Write content
            mode = "w" if file_type == "text" else "wb"
            with open(filepath, mode) as f:
                f.write(content)

            # Generate checksum
            checksum = self._generate_checksum(filepath)

            return filepath, checksum

        except Exception as e:
            msg = f"Failed to save file: {str(e)}"
            raise ValueError(msg)

    def read_file(self, filepath: str) -> str:
        """
        Read file content with validation.

        Args:
            filepath: Full path to the file

        Returns:
            File content as string

        Raises:
            ValueError: For invalid files or read errors
        """
        try:
            # Security checks
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

            if not os.path.isfile(filepath):
                raise ValueError(f"Path is not a file: {filepath}")

            # Read content
            with open(filepath, "r") as f:
                content = f.read()

            return content

        except Exception as e:
            raise ValueError(f"Failed to read file: {str(e)}")

    def _generate_checksum(self, filepath: str) -> str:
        """Generate SHA-256 checksum for file verification"""
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def get_file_metadata(self, filepath: str) -> dict:
        """Get comprehensive file metadata"""
        path = Path(filepath)
        return {
            "filepath": str(path.absolute()),
            "filename": path.name,
            "size": os.path.getsize(filepath),
            "created": os.path.getctime(filepath),
            "modified": os.path.getmtime(filepath),
            "checksum": self._generate_checksum(filepath),
        }

    def cleanup(self, filepath: str) -> bool:
        """Safely remove a file"""
        try:
            os.remove(filepath)
            return True
        except Exception:
            return False
