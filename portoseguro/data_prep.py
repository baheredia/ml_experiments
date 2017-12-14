import bcolz

def save_array(path, array):
    c = bcolz.carray(array, rootdir=path, mode='w')
    c.flush()
    
def load_array(path):
    return bcolz.open(path)[:]
