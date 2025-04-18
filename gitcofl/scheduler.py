import time
import schedule


class DefaultScheduler:
    def __init__(self, interval=10, task=None):
        """
        Initializes the scheduler with a default interval.
        :param interval: Time interval (seconds) between federated training rounds.
        :param task: The function to execute on schedule.
        """
        self.interval = interval
        self.task = task or self.default_task
        schedule.every(self.interval).seconds.do(self.task)

    def default_task(self):
        print("Running default federated learning round...")


    def start(self):
        """
        Starts the scheduler in a blocking loop.
        """
        print(f"Scheduler started with interval {self.interval} seconds...")
        while True:
            schedule.run_pending()
            time.sleep(1)  # Prevents busy-waiting

    def stop(self):
        """
        Stops the scheduler and exits the program.
        """
        print("Scheduler stopped.")
        exit(0)