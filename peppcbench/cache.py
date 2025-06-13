from pathlib import Path
from typing import Dict, Optional, Tuple
from diskcache import Cache
from loguru import logger


class ChainCache:
    def __init__(self, cache_dir: Path, size_limit: int = 1 << 40):
        self.cache = Cache(cache_dir, size_limit=size_limit)

    def auto_get(self, key: str, sample_dir: Path) -> Optional[str]:
        if self.cache.get(key):
            logger.debug(f"cache hit for {key}")
            return self.cache.get(key)
        else:
            return None

    def get(self, key: str) -> Optional[Tuple[str, str, Dict[str, str]]]:
        if self.cache.get(key):
            logger.debug(f"cache hit for {key}")
            pdb_model_processed_str, pdb_ref_processed_str, chain_mapping = (
                self.cache.get(key)
            )
            return pdb_model_processed_str, pdb_ref_processed_str, chain_mapping
        else:
            return None

    def close(self):
        self.cache.close()
