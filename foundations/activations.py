import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        result = []
        for z_ in z:
            result.append(round(1 / (1+np.e**(-z_)), 5))
        return np.array(result)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        result = []
        for z_ in z:
            result.append(max(0, z_))
        return np.array(result, dtype=np.float64)
        
