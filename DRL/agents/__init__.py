from abc import ABCMeta, abstractmethod


class Agent(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError


    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def get_action(self):
        raise NotImplementedError