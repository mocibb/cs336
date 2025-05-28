import heapq
from typing import Optional, TypeVar, Generic, Any, Tuple, Hashable

K = TypeVar('K', bound=Hashable)  # 限定key为可哈希类型

class QueueElement(Generic[K]):
    def __init__(self, elem):
        self.elem = elem

    def __lt__(self, other):
        return self.elem > other.elem

    def __str__(self):
        return str(self.elem)
    
class PriorityDict:
    def __init__(self):
        self._heap = [] 
        self._entry_map = {}
    
    def __setitem__(self, key: K, value: int) -> None:
        self._entry_map[key] = value
        heapq.heappush(self._heap, QueueElement((value, key)))

    def __getitem__(self, key: K) -> int:
        if key in self._entry_map:
            return self._entry_map[key]
        else:
            self.__setitem__(key, 0)
            return self._entry_map[key]

    def pop(self) -> tuple[K, int]:
        while self._heap:
            value, key = heapq.heappop(self._heap).elem
            if key in self._entry_map and self._entry_map[key] == value:
                del self._entry_map[key]
                return (key, value)
        raise KeyError("PriorityDict is empty")

    def get(self) -> Optional[tuple[K, int]]:
        while self._heap:
            value, key = self._heap[0]
            if key in self._entry_map and self._entry_map[key] == value:
                return (key, value)
            else:
                heapq.heappop(self._heap)
        return None

    def __contains__(self, key: K) -> bool:
        return key in self._entry_map

    def __len__(self) -> int:
        return len(self._entry_map)
    
    def print(self) -> None:
        print(self._entry_map)
        # for k, v in self._entry_map.items():
        #     print(f"k = {k}, v = {v}")
    


if __name__ == "__main__":
    pd = PriorityDict()
    print(pd["aa"])
    pd["a"] = 3
    pd["b"] = 2
    pd["c"] = 1
    pd["a"] = 2
    pd["a"] = 3
    print(pd.pop())
    print(pd.pop())
    print(pd.pop())
    print(pd.pop())
