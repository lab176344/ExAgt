import abc

class Scheduler(abc.ABC):
    def __init__(self) -> None:
        self.name = None
        self.size = None
        self.n_samples = None
        self.description = None
        # TODO

    @abc.abstractmethod
    def get_lr_for_iter(self, iteration):
        pass
    def _get_description(self):
        #TODO
        description = {'name':self.name,}
        return description