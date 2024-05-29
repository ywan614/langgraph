from contextlib import contextmanager
from typing import Iterator, Sequence

import automerge.core
import automerge.document

from langgraph.channels.base import BaseChannel


class Document(automerge.document.MapReadProxy):
    def __init__(self, doc: automerge.core.Document) -> None:
        super().__init__(doc, automerge.core.ROOT, None)

    @contextmanager
    def change(self) -> Iterator[automerge.document.MapWriteProxy]:
        with self._doc.transaction() as tx:
            yield automerge.document.MapWriteProxy(tx, automerge.core.ROOT, None)


class AutomergeValue(BaseChannel):
    document: automerge.core.Document

    def checkpoint(self) -> bytes:
        return self.document.save()

    @contextmanager
    def from_checkpoint(self, checkpoint: bytes) -> Iterator["AutomergeValue"]:
        it = self.__class__()
        it.document = automerge.core.Document.load(checkpoint)
        try:
            yield it
        finally:
            del it.document

    def get(self) -> Document:
        return Document(self.document.fork())

    def update(self, values: Sequence[Document]) -> None:
        for value in values:
            self.document.merge(value._doc)
