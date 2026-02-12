from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EpisodeLog:
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, **kwargs: Any) -> None:
        self.entries.append(dict(kwargs))

    def as_dict_of_lists(self) -> Dict[str, List[Any]]:
        # Convert list-of-dicts to dict-of-lists for plotting.
        out: Dict[str, List[Any]] = {}
        for row in self.entries:
            for k, v in row.items():
                out.setdefault(k, []).append(v)
        return out