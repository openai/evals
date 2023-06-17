"""
Base class for all memory data structures.
"""
import abc

class Memory(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def create(self, text):
        pass 

    @abc.abstractmethod
    def sample(self, query):
        pass
