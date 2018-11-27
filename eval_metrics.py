import numpy as np

class IOU(object):
    """description of class"""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.I = np.zeros(num_classes)
        self.U = np.zeros(num_classes)

    def accumulate(self, label, pred):
        for i in range(0, self.num_classes):
            lb = np.copy(label)
            lb[label == i] = 1
            lb[label != i] = 0
            
            pb = np.copy(pred)
            pb[pred == i] = 1
            pb[pred != i] = 0
                
            self.I[i] += np.count_nonzero(np.logical_and(lb, pb))
            self.U[i] += np.count_nonzero(np.logical_or(lb, pb))

    def calculate(self):
        return self.I / self.U