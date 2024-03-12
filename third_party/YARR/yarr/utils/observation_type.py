from typing import Type
import numpy as np


class ObservationElement(object):

    def __init__(self, name: str, shape: tuple, type: Type[np.dtype]):
        self.name = name
        self.shape = shape
        self.type = type
