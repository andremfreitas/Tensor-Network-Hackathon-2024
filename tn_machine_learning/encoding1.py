def encoding1(array: np.array):
    new_array = np.zeros((array.shape[0],2))
    for i in range(len(array)):
        p_i = np.sqrt(array[i])
        new_array[i] = (p_i, np.sqrt(1-p_i))
    return new_array.flatten()

n_sites = 16
prova = np.random.rand(n_sites)
encoding1(prova)
