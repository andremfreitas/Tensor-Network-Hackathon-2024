def classifier_list(max_bond, encoding=1, num_sweeps=100, n_jobs=20):
        # Convert dataset into a list of MPS
    # --------------------------------------
    X_train_mps = []
    X_test_mps = []

    if encoding == 1:
        for i in range(x_test.shape[0]):
            X_train_mps.append(MPS.from_tensor_list(encoding1(x_train[i, :]), conv_params=TNConvergenceParameters()))
            X_test_mps.append(MPS.from_tensor_list(encoding1(x_test[i]), conv_params=TNConvergenceParameters()))
    elif encoding == 2:
        for i in range(x_test.shape[0]):
            X_train_mps.append(MPS.from_statevector(x_train[i, :], conv_params=TNConvergenceParameters()))
            X_test_mps.append(MPS.from_statevector(x_test[i], conv_params=TNConvergenceParameters()))
    
        # Optimize MPS

    # get number of sites

    # define max bond dimension
    if encoding == 1:
        num_sites = x_train.shape[1]
    elif encoding == 2: 
        num_sites = int(np.log2(x_train.shape[1]))

    # define batch size, learning rate and number of sweeps
    batch_size = 50
    learning_rate = 1e-4
    

    # initialize MPS for the classifier
    conv_params = TNConvergenceParameters(
                        max_bond_dimension=max_bond)
    tn_classifier = MPS(num_sites, conv_params, dtype=float)

    svd, loss = tn_classifier.ml_optimize_mps(X_train_mps,
                                    y_train,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    num_sweeps=num_sweeps,
                                    n_jobs=n_jobs,
                                    verbose=False)
    # Predict labels with the trained MPS
    y_train_pred = tn_classifier.ml_predict(X_train_mps, n_jobs=1)
    y_test_pred = tn_classifier.ml_predict(X_test_mps, n_jobs=1)
    # Get accuracy
    accuracy_train = len(np.where(y_train == np.real(y_train_pred))[0]) / len(y_train)
    print(f"Accuracy_train: {accuracy_train}")

    accuracy_test = len(np.where(y_test == np.real(y_test_pred))[0]) / len(y_test)
    print(f"Accuracy_test: {accuracy_test}")

    entropy = tn_classifier.meas_bond_entropy()
    entropy_l = entropy.values()

    print(f"Bond entropy: {entropy_l}")
    # x_array = np.arange(len(entropy_l))

    return accuracy_test, accuracy_train, entropy, entropy_l

res_enc1 = {}

res_enc2 = {}

for max_bond in range(2, 13):
    print(f"Max bond: {max_bond} - Encoding 1")
    value_1 = classifier_list(max_bond, encoding=1, num_sweeps=100, n_jobs=2)
    res_enc1[max_bond] = {}
    res_enc1[max_bond]['accuracy_test']     = value_1[0]
    res_enc1[max_bond]['accuracy_train']    = value_1[1]
    res_enc1[max_bond]['entropy']           = value_1[2]
    res_enc1[max_bond]['entropy_l']         = value_1[3]

    print(f"Max bond: {max_bond} - Encoding 2")
    value_2 = classifier_list(max_bond, encoding=2, num_sweeps=1000, n_jobs=2)
    res_enc2[max_bond] = {}
    res_enc2[max_bond]['accuracy_test']     = value_2[0]
    res_enc2[max_bond]['accuracy_train']    = value_2[1]
    res_enc2[max_bond]['entropy']           = value_2[2]
    res_enc2[max_bond]['entropy_l']         = value_2[3]




