import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    print("the x=======>",x)
    x=np.array(x)
    e_x = np.exp(x - np.max(x,axis=-1))
    return e_x / e_x.sum(axis=-1)

