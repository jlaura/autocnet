import os
import shutil
import threading
import queue


class FileCopy(threading.Thread):
    def __init__(self, queue):
        """
        Instantiate a FileCopy worker.

        Parameters
        ----------
        queue : queue.Queue
                The processing queue from which work is pulled.
        """
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True

    def run(self):
        """
        When the thread launches, pull messages off of 
        the queue, execute a file copy for the message
        and then alert done (iterate). If the queue is empty
        the thread will exit the run and terminate.
        """
        while True:
            try:
                oldnew = self.queue.get()
            except self.queue.Empty:
                return

            try:
                old, new = oldnew
                shutil.copy(old, new)
            except IOError as e:
                self.queue.put(e)
            self.queue.task_done()