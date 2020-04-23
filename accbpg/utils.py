import os.path
import numpy as np
import scipy.sparse as sparse


def _open_file(filename):

    _, ext = os.path.splitext(filename)
    if ext == '.gz':
        import gzip
        return gzip.open(filename, 'rt')
    elif ext == '.bz2':
        import bz2
        return bz2.open(filename, 'rt')
    else:
        return open(filename, 'r')


def load_libsvm_file(filename, dtype=np.float64, 
                     n_features=None, zero_based="auto"):
    """
    Load dataset in svmlight / libsvm format into sparse CSR matrix.

    Inputs:
    filename: a string including file path and name
    dtype: numpy dtype of feature values
    n_features: number of features, optional
    zero_based: boolean or "auto", optional

    Returns:
    X: scipy.sparse.csr_matrix of shape (n_samples, n_features)
    y: numpy.ndarray of shape (n_samples,)

    """

    labels = []
    data = []
    indptr = []
    indices = []
    
    with _open_file(filename) as f:

        for line in f:
            
            # skip comments in the line
            idx_comment = line.find('#')
            if idx_comment >= 0:
                line = line[:idx_comment]

            line_parts = line.split()
            if len(line_parts) == 0:
                continue

            labels.append(float(line_parts[0]))
            indptr.append(len(data))

            prev_idx = -1
            for i in range(1,len(line_parts)):
                idx_str, value = line_parts[i].split(':',1)
                idx = int(idx_str)
                if idx < 0 or (not zero_based and idx == 0):
                    raise ValueError(
                      "Invalid index {0:d} in LibSVM data file.".format(idx))
                if idx <= prev_idx:
                    raise ValueError("Feature indices in LibSVM data file"
                                     "should be sorted and unique.")
                indices.append(idx)
                data.append(dtype(value))
                prev_idx = idx

    # construct data arrays
    indptr.append(len(data))

    data = np.array(data)
    indptr = np.array(indptr)
    indices = np.array(indices)

    if (zero_based is False or zero_based == "auto" and indices.min() > 0):
        indices -= 1
    if n_features is None:
        n_features = indices.max() + 1
    else:
        if n_features < indices.max() + 1:
            n_features = indices.max() + 1
            print("Warning: n_features increased to match data.")

    shape = (indptr.shape[0] - 1, n_features)
    X = sparse.csr_matrix((data, indices, indptr), shape)
    X.sort_indices()
    y = np.array(labels)

    return X, y


def shuffle_data(X, y):
    '''
    We need to return here since whole array assignment in numpy does not 
    change input arguments, i.e., it does NOT behaves as passing by reference
    '''
    index = np.arange(len(y))
    np.random.shuffle(index)
    X = X[index,:]
    y = y[index]
    return X, y     


def mnist_2digits(X, y, d1, d2):
    index1 = np.nonzero(y==d1)
    index2 = np.nonzero(y==d2)
    ycopy = y.copy()
    ycopy[index1] =  1
    ycopy[index2] = -1
    index = np.concatenate((index1[0], index2[0]))
    np.random.shuffle(index)
    Xd1d2 = X[index, :]
    yd1d2 = ycopy[index]
    return Xd1d2, yd1d2


def binary_error_rate(X, y, w, bias=0):
    if sparse.isspmatrix(X):
        yp = np.sign( X * w + bias )
    else:
        yp = np.sign( np.dot(X, w) + bias )

    return (1 - np.dot(yp, y)/len(y))/2


def rmse(X, y, w, bias=0):
    if sparse.isspmatrix(X):
        yp = X * w + bias
    else:
        yp = np.dot(X, w) + bias

    error2 = (yp - y)**2
    return np.sqrt(error2.mean())


def row_norm_squared(X):
    "return squared 2-norms of each row"
    X2 = sparse.csr_matrix((X.data**2, X.indices, X.indptr), X.shape)
    return np.squeeze(np.asarray(X2.sum(1))) 


def load_sido(filename):
    with np.load(filename) as D:
        data = D['Xdata']
        indptr = D['Xindptr']
        indices = D['Xindices']
        y = D['y']
        shape = D['shape']

    X = sparse.csr_matrix((data, indices, indptr), shape)

    return X, y
