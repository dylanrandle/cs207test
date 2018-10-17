import pytest
import numpy as np
from L2 import L2


def test_L2_result():
    assert L2([1,1], [1,1]) == np.sqrt(2)
    
def test_L2_weight():
    assert L2([1,1], [2,2]) == 2*np.sqrt(2)
    
def test_L2_dims():
    with pytest.raises(ValueError):
        L2([1,1,1], [2,2])

def test_L2_zerolen():
    assert L2([])==0.0
