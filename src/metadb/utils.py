"""Utility functions for the MetaDB package."""

import random
import datetime
import hashlib
import json
from typing import Dict, List, Any, Union, Optional


def generate_unique_id(prefix: str = "id") -> str:
    """
    Generate a unique identifier with an optional prefix.
    
    Args:
        prefix (str): Optional prefix for the ID. Defaults to "id".
    
    Returns:
        str: A unique string identifier
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    random_part = str(random.randint(1000, 9999))
    return f"{prefix}_{timestamp}_{random_part}"


def hash_data(data: Any) -> str:
    """
    Create a SHA-256 hash of any serializable data.
    
    Args:
        data: Any data that can be serialized to JSON
        
    Returns:
        str: Hexadecimal digest of the hash
    """
    if not isinstance(data, str):
        data = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def format_timestamp(timestamp: Optional[float] = None,
                    format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp into a human-readable string.
    
    Args:
        timestamp (float, optional): Unix timestamp to format. 
                                    Defaults to current time if None.
        format_string (str): Format string for the output. 
                            Defaults to "%Y-%m-%d %H:%M:%S".
    
    Returns:
        str: Formatted timestamp string
    """
    if timestamp is None:
        dt = datetime.datetime.now()
    else:
        dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime(format_string)


def chunk_list(input_list: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        input_list (List[Any]): Input list to be chunked
        chunk_size (int): Size of each chunk
        
    Returns:
        List[List[Any]]: List of chunks
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def merge_dicts(dict1: Dict[str, Any], 
               dict2: Dict[str, Any], 
               overwrite: bool = True) -> Dict[str, Any]:
    """
    Merge two dictionaries, with optional overwrite behavior.
    
    Args:
        dict1 (Dict[str, Any]): First dictionary
        dict2 (Dict[str, Any]): Second dictionary
        overwrite (bool): Whether to overwrite values in dict1 with values 
                         from dict2 if keys clash. Defaults to True.
                         
    Returns:
        Dict[str, Any]: Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and not overwrite:
            continue
        if isinstance(value, dict) and isinstance(result.get(key, None), dict):
            result[key] = merge_dicts(result[key], value, overwrite)
        else:
            result[key] = value
            
    return result