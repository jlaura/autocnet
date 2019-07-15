import os
import shutil
import threading
import queue


class FileCopy(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True

    def run(self):
        # This puts one object into the queue for each file
        while True:
            try:
                oldnew = self.queue.get()
            except self.queue.Empty:
                return
            
            if oldnew == None:
                break

            try:
                old, new = oldnew
                shutil.copy(old, new)
            except IOError as e:
                self.queue.put(e)
            self.queue.task_done()