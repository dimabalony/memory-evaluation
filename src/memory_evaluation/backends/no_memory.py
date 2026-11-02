"""Independent baseline: no memory store, extraction, or embedding calls."""

from contextlib import asynccontextmanager

from memory_evaluation.contracts import BackendDescriptor, Retrieval


class NoMemoryBackend:
    def describe(self):
        return BackendDescriptor("no_memory", "1", configuration={"storage": "absent"})

    @asynccontextmanager
    async def open_fixture(self, fixture, config):
        yield self

    async def retrieve(self, request, question):
        return Retrieval("")

    async def state_receipt(self):
        return "no-memory-store/v1"
