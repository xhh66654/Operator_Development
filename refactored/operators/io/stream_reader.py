"""流式数据源抽象：按块/按行迭代，避免全量加载"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List


class StreamableDataSource(ABC):
    """流式数据源抽象"""

    @abstractmethod
    def iter_chunks(self, chunk_size: int = 10000) -> Iterator[List[Dict[str, Any]]]:
        """按块迭代"""
        pass

    @abstractmethod
    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        """按行迭代"""
        pass
