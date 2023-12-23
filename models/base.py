from abc import ABC, abstractmethod


class RALLM(ABC):
    def __init__(self,
                 cfg: dict,
                 ):
        self.cfg = cfg

    @abstractmethod
    def retrieve_and_read(self,
                          prompt: str,
                          data: dict  # contains 'query' key
                          ) -> None:
        """retrieval and read"""

    @abstractmethod
    def generate_choice(self) -> None:
        """generate choice"""

    @abstractmethod
    def simple_read(self) -> None:
        """
        only for without retrieval method
        """
