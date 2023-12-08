from modules import *


def createFGAModel(X, y, verbose=False, labels=None, labels_predefined=None):
    """
    Create a Fuzzy Genetic Algorithm model for regression.

    Parameters
    ----------
    X : pandas.DataFrame
        The features of the data.
        df[['data_in1', 'data_in2', ...]]
    y : pandas.Series
        The target variable of the data.
        df['data_out']
    verbose : bool, optional
        Whether to print the parameters of the model. The default is False.
    labels : list, optional
        The labels for the fuzzy sets. The default is None and made in code.
    labels_predefined : list, optional
        The labels for the fuzzy sets rules. The default is None and made in code.

    Returns
    -------
    results : list
        The predicted values of the model.
    simulation : ctrl.ControlSystemSimulation
        The simulation of the model.
    """

    # =================================================================
    # ============================ DATA ===============================
    # =================================================================
    tmp, rh = X["TMP"].values, X["RH"].values
    pm10 = y.values

    # =================================================================
    # ============================= LABELS ============================
    # =================================================================
    if labels is None:
        labels = ["mbaja", "baja", "media", "alta", "malta"]
        labels_predefined = [["mbaja", "alta", "alta", "media", "malta"],["alta", "alta", "alta", "baja", "mbaja"],["malta", "media", "baja", "mbaja", "mbaja"],["media", "malta", "baja", "baja", "mbaja"],["media", "alta", "media", "mbaja", "baja"]]
        labels_predefined = [['mbaja', 'alta', 'malta', 'media', 'malta'],['alta', 'alta', 'alta', 'baja', 'mbaja'],['malta', 'media', 'baja', 'mbaja', 'mbaja'],['malta', 'malta', 'baja', 'baja', 'mbaja'],['malta', 'alta', 'mbaja', 'mbaja', 'baja']]
    num_labels = len(labels)

    # =================================================================
    # =================== ANTECEDENTS & CONSEQUENTS ===================
    # =================================================================
    samples = 100
    tmp_universe = np.linspace(min(tmp), max(tmp), num=samples)
    tmp_ctrl = ctrl.Antecedent(tmp_universe, "tmp")
    rh_universe = np.linspace(min(rh), max(rh), num=samples)
    rh_ctrl = ctrl.Antecedent(rh_universe, "rh")
    pm10_universe = np.linspace(min(pm10), max(pm10), num=samples)
    pm10_ctrl = ctrl.Consequent(pm10_universe, "pm10")

    # =================================================================
    # ====================== MEMBERSHIP FUNCTIONS =====================
    # =================================================================
    create_automf(rh_ctrl, labels, num_labels)
    create_automf(tmp_ctrl, labels, num_labels)
    create_automf(pm10_ctrl, labels, num_labels)
    control_vars = [tmp_ctrl, rh_ctrl, pm10_ctrl, labels]

    # =================================================================
    # ============================= RULES =============================
    # =================================================================
    rules = create_rules(tmp_ctrl, rh_ctrl, pm10_ctrl, labels, labels_predefined)

    # =================================================================
    # ========================= FUZZY SYSTEM ==========================
    # =================================================================
    system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(system)

    # =================================================================
    # ========================= FL OPERATIONS =========================
    # =================================================================
    results = []
    for i in range(len(tmp)):
        # print(f"{i+1}/{len(tmp)}", end="\r")

        simulation.input["tmp"] = tmp[i]
        simulation.input["rh"] = rh[i]
        simulation.compute()
        out = simulation.output["pm10"]
        results.append(out)

    # =================================================================
    # ========================= PLOT RESULTS ==========================
    # =================================================================
    if verbose:
        print("\nMSE: ", mean_squared_error(pm10, results))
        print("R2: ", r2_score(pm10, results))
        plt.plot(pm10, label="real")
        plt.plot(results, label="predicted")
        plt.legend()
        plt.show()

    # =================================================================
    # ============================ RETURN =============================
    # =================================================================
    return results, simulation


def predictFGAModel(tmp, rh, simulation):
    """
    Predict the output of a Fuzzy Genetic Algorithm model for regression.

    Parameters
    ----------
    tmp : float
        The temperature value.
    rh : float
        The relative humidity value.

    Returns
    -------
    out : float
        The predicted value of the model.
    """

    # =================================================================
    # ========================= FL OPERATIONS =========================
    # =================================================================
    simulation.input["tmp"] = tmp
    simulation.input["rh"] = rh
    simulation.compute()
    out = simulation.output["pm10"]

    # =================================================================
    # ============================ RETURN =============================
    # =================================================================
    return out
