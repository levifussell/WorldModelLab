
class Buffer:

    def __init__(self, max_len):
        
        self._max_len = max_len

    def __len__(self):
        return self._max_len