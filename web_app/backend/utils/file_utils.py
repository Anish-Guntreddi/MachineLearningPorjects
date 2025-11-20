"""
File Utility Functions
Handles file uploads, validation, and cleanup
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile, HTTPException
import aiofiles


# Allowed file types by category
ALLOWED_EXTENSIONS = {
    'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'},
    'audio': {'.wav', '.mp3', '.flac', '.ogg', '.m4a'},
    'text': {'.txt', '.csv', '.json'},
    'video': {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
}

# Max file sizes (in bytes)
MAX_FILE_SIZES = {
    'image': 10 * 1024 * 1024,  # 10MB
    'audio': 50 * 1024 * 1024,  # 50MB
    'text': 5 * 1024 * 1024,    # 5MB
    'video': 100 * 1024 * 1024  # 100MB
}


def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """
    Validate if file extension is in allowed types

    Args:
        filename: Name of the file
        allowed_types: List of allowed categories (e.g., ['image', 'audio'])

    Returns:
        True if valid, False otherwise
    """
    ext = Path(filename).suffix.lower()

    for file_type in allowed_types:
        if file_type in ALLOWED_EXTENSIONS:
            if ext in ALLOWED_EXTENSIONS[file_type]:
                return True

    return False


def get_file_category(filename: str) -> Optional[str]:
    """
    Determine the category of a file based on its extension

    Args:
        filename: Name of the file

    Returns:
        Category name or None if not recognized
    """
    ext = Path(filename).suffix.lower()

    for category, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return category

    return None


async def save_upload_file(upload_file: UploadFile, destination: Optional[Path] = None) -> Path:
    """
    Save an uploaded file to disk

    Args:
        upload_file: FastAPI UploadFile object
        destination: Optional destination path, uses temp dir if None

    Returns:
        Path to saved file

    Raises:
        HTTPException: If file is too large or invalid
    """
    # Validate file type
    category = get_file_category(upload_file.filename)
    if not category:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {Path(upload_file.filename).suffix}"
        )

    # Create destination path
    if destination is None:
        temp_dir = Path(tempfile.mkdtemp())
        destination = temp_dir / upload_file.filename

    # Save file
    try:
        async with aiofiles.open(destination, 'wb') as f:
            content = await upload_file.read()

            # Check file size
            max_size = MAX_FILE_SIZES.get(category, 10 * 1024 * 1024)
            if len(content) > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
                )

            await f.write(content)

        return destination

    except Exception as e:
        if destination.exists():
            destination.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )


async def save_multiple_files(upload_files: List[UploadFile]) -> List[Path]:
    """
    Save multiple uploaded files

    Args:
        upload_files: List of FastAPI UploadFile objects

    Returns:
        List of paths to saved files
    """
    saved_paths = []

    try:
        for upload_file in upload_files:
            path = await save_upload_file(upload_file)
            saved_paths.append(path)

        return saved_paths

    except Exception as e:
        # Cleanup any files that were saved before the error
        cleanup_temp_files(saved_paths)
        raise e


def cleanup_temp_files(file_paths: List[Path]) -> None:
    """
    Delete temporary files and their parent directories if empty

    Args:
        file_paths: List of file paths to delete
    """
    for path in file_paths:
        try:
            if path.exists():
                path.unlink()

                # Try to remove parent directory if it's empty
                parent = path.parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()

        except Exception as e:
            # Log but don't raise - cleanup is best effort
            print(f"Warning: Failed to cleanup {path}: {e}")


def get_file_info(file_path: Path) -> dict:
    """
    Get information about a file

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information
    """
    stat = file_path.stat()

    return {
        'name': file_path.name,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'extension': file_path.suffix,
        'category': get_file_category(file_path.name),
        'modified': stat.st_mtime
    }
