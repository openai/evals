import abc

class Memory(abc.ABC):
    """
    Base class for all memory data structures.
    This class defines the interface for a memory module.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the memory module.
        """
        pass

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        """
        Add a new text entry to the memory.
        
        Args:
            text (str): The text to add to the memory.
        """
        pass 

    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        """
        Sample an entry from the memory based on a query.
        
        Args:
            query (str): The query to use for sampling from the memory.
        
        Returns:
            str: A text entry from the memory.
        """
        pass
