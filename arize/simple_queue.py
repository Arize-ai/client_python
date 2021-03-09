import queue


class SimpleQueue(queue.Queue):
    def add(self, data):
        if data:
            self.put(data)

    def close(self):
        self.put(None)
        return self

    def __iter__(self):
        return iter(self.get, None)
