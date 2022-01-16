from abc import ABC, abstractmethod


class BaseMixin(ABC):

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def compile(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load(self, *args, **kwargs):
        raise NotImplementedError
