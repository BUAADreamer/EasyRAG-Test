class BaseRetriever:
    def __init__(self):
        pass

    def augment(self,
                query: str,
                prompt: str
                ) -> str:
        raise NotImplementedError

    def retrieve(self,
                 query: str,
                 topk: int
                 ) -> tuple[list[str], list[float]]:
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
