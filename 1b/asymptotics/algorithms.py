from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta

class AbstractAlgorithm(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def invoke(self, n = 1):
        pass

class Constant(AbstractAlgorithm):

    def invoke(self, n = 1):

        # To follow the example in Goodrich, et al. (p. 130), create a list, and retrieve the
        # length of the list
        temp_list = ["Hello"] * n
        start_time = datetime.now()
        len(temp_list)
        return (datetime.now() - start_time) / timedelta(microseconds = 1)