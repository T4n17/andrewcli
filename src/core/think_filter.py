class ThinkFilter:
    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self):
        self.in_think = False
        self._buffer = ""

    def process(self, text):
        self._buffer += text
        results = []
        while self._buffer:
            tag = self.CLOSE_TAG if self.in_think else self.OPEN_TAG
            idx = self._buffer.find(tag)
            if idx != -1:
                before = self._buffer[:idx]
                if before:
                    results.append((before, self.in_think))
                self._buffer = self._buffer[idx + len(tag):]
                self.in_think = not self.in_think
            else:
                held = self._hold_partial(tag)
                safe = self._buffer[:len(self._buffer) - held] if held else self._buffer
                if safe:
                    results.append((safe, self.in_think))
                self._buffer = self._buffer[len(self._buffer) - held:] if held else ""
                break
        return results

    def _hold_partial(self, tag):
        for i in range(min(len(tag) - 1, len(self._buffer)), 0, -1):
            if self._buffer[-i:] == tag[:i]:
                return i
        return 0