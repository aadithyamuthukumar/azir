from abc import ABC, abstractmethod

from schemas import ChatRequest, ChatResponse


class Provider(ABC):

    @abstractmethod
    async def complete(self, request: ChatRequest) -> ChatResponse:
        pass