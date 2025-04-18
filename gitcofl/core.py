from abc import ABC, abstractmethod
from .scheduler import DefaultScheduler

class FLClient(ABC):
    def __init__(self, interval=10, scheduler=None):
        """
        Initializes federated learning with an optional custom scheduler.
        :param interval: Time interval for checking updates (in seconds)
        :param scheduler: An instance of a scheduler, defaults to `DefaultScheduler`.
        """
        self.scheduler = scheduler or DefaultScheduler(interval=interval, task=self.process)

    @abstractmethod
    def pull_global_weights(self):
        pass

    @abstractmethod
    def after_pull_global_weights(self):
        pass

    @abstractmethod
    def push_local_weights(self):
        pass

    @abstractmethod
    def load_weights(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def save_weight(self):
        pass

    def process(self):
        try:
            self.pull_global_weights()
            self.after_pull_global_weights()
        except:
            pass

        is_loaded = self.load_weights()
        if is_loaded:
            self.train()
            self.test()
            self.save_weight()
            self.push_local_weights()

            if self.count_fl_round > self.total_fl_rounds:
                print("Total number of rounds reached. Stopping the scheduler...")
                self.scheduler.stop()

            
        else:
            print("global weights not comming yet. Skipping to next schedule...")

    def default_training_round(self):
        print("Performing a training federated learning round...")
    
    def default_testing_round(self):
        print("Performing a testing federated learning round...")

    def start(self):
        print("Starting federated learning with scheduling...")
        self.scheduler.start()