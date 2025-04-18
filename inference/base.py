from abc import abstractmethod

class BaseInference:
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def infer(self):
        pass

    @abstractmethod
    def verify(self):
        pass