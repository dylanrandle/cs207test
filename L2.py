
import numpy as np

def L2(v, *args):
    """
    Function to compute the (possibly weighted) L2 norm of a vector
    
    INPUTS
    ======
    v: list of floats, required
       vector for wich we compute the L2 norm
    *args: additional arguments, optional
           could be a list of floats, respresenting weights (if so, must be first element)
    
    RETURNS
    =======
    L2-norm: float
             Raises ValueError if weight vector and vector have different length
             
    NOTES
    =====
    PRE:
        - v is a list of floats
        - *args has a weight vector equal in length to v in the first element if provided
    POST:
        - v, *args is unchanged
        - raises ValueError if len(v) != len(*args[0])
        - returns a float representing the L2 norm
    
    EXAMPLES
    ========
 
    """
    s = 0.0 # Initialize sum
    if len(args) == 0: # No weight vector
        for vi in v:
            s += vi * vi
    else: # Weight vector present
        w = args[0] # Get the weight vector
        if (len(w) != len(v)): # Check lengths of lists
            raise ValueError("Length of list of weights must match length of target list.")
        for i, vi in enumerate(v):
            s += w[i] * w[i] * vi * vi
    return np.sqrt(s)
