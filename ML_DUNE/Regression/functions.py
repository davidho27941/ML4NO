import numpy as np

def load_train_data(learn_target):
    if '3flavor' in learn_target:
        training_data = np.load("../../Data/n1000000_0910_all_flat.npz")
        data_all = np.column_stack([training_data['ve_dune'][:,:36], training_data['vu_dune'][:,:36], training_data['vebar_dune'][:,:36], training_data['vubar_dune'][:,:36]])

        # theta13, theta23, cos(delta), sin(delta)
        target = np.column_stack([training_data["theta13"]/180*np.pi, training_data["theta23"]/180*np.pi,
                                np.cos(training_data["delta"]/180*np.pi), np.sin(training_data["delta"]/180*np.pi)])
        split = 900000

    else:
        training_data = np.load("../../Data/nsi_data/sample_nsi_regression_1e7_v2.npz") #v1 is used for test
        data_all = np.column_stack([training_data['ve_dune'][:,:36], training_data['vu_dune'][:,:36], training_data['vebar_dune'][:,:36], training_data['vubar_dune'][:,:36]])

        # theta13, theta23, cos(delta), sin(delta), mumu, emu, etau
        target = np.column_stack([training_data["theta13"]/180*np.pi, training_data["theta23"]/180*np.pi,
                                np.cos(training_data["delta"]/180*np.pi), np.sin(training_data["delta"]/180*np.pi),
                                training_data["mumu"], training_data["emu"],training_data["etau"]])
        split = 9000000
        
    x_train = data_all[:split]
    y_train = target[:split]
    x_val = data_all[split:]
    y_val = target[split:]

    if 'clean' in learn_target: return x_train, y_train, x_val, y_val
    else: return np.random.poisson(x_train), y_train, np.random.poisson(x_val), y_val

# x_train, y_train, x_val, y_val = load_train_data(learn_target)

def load_test_data(learn_target):
    if '3flavor' in learn_target:
        data = np.load('../../Data/sample_NuFit0911.npz')
        data_all = np.column_stack([data['ve_dune'][:,:36], data['vu_dune'][:,:36], data['vebar_dune'][:,:36], data['vubar_dune'][:,:36]])
    else:
        data_index = 1
        training_data = np.load("../../Data/nsi_data/sample_nsi_regression_1e7_v1.npz")
        data_all = np.column_stack([training_data['ve_dune'][data_index:data_index+1,:36], training_data['vu_dune'][data_index:data_index+1,:36],
                                    training_data['vebar_dune'][data_index:data_index+1,:36], training_data['vubar_dune'][data_index:data_index+1,:36]])
    return data_all
# data_all = load_test_data(learn_target)

def chi2_graph(learn_target):
    if '3flavor' in learn_target:
        N_DUNE = 92
        theta23_DUNE, delta_cp_DUNE, chi_DUNE = [], [], []
        f_DUNE = open("../../Data/chi_square-4-2_figB_DUNE.txt")
        for i in range(N_DUNE):
            s = f_DUNE.readline().split()
            array = []
            for j in range(len(s)):
                array.append(float(s[j]))
            theta23_DUNE.append(array[0])
            delta_cp_DUNE.append(array[1]) 
            chi_DUNE.append(array[2])
        f_DUNE.close()

        theta23_DUNE = np.array(theta23_DUNE)
        delta_cp_DUNE = np.array(delta_cp_DUNE)
        chi_DUNE = np.array(chi_DUNE)
        
        x = np.linspace(min(theta23_DUNE)-3, max(theta23_DUNE)+3, 68)
        y = np.linspace(min(delta_cp_DUNE)-6, max(delta_cp_DUNE)+6, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((len(X),len(X[0])))

        for i in range(len(theta23_DUNE)):
            Z[np.where(Y == delta_cp_DUNE[i])[0][0]][np.where(X == theta23_DUNE[i])[1][0]] = 1
        return x, y, X, Y, Z # x, y, X, Y, Z = chi2_graph()
    else:
        pass
