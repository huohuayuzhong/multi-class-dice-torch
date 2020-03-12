# multi-class-dice-torch
Calculate dice for multi class with soft terms, weights and ignore indices

    Multi class dice loss
    :param pred: [n_batch, n_class, ...] 1D to 3D inputs ranged in [0, 1] (as prob)
    :param mask: [n_batch, n_class, ...] 1D to 3D inputs as a 0/1 mask
    :param fp_weight: float [0,1], penalty for fp preds, may work in data with heavy fg/bg imbalances
    :param label_smooth: float (0, inf), power of the denominator
    :param eps: epsilon, avoiding zero-divide
    :param class_weight: list [float], weights for classes
    :param ignore_index: int [0, n_class), num of classes to ignore; or list [int], indices to ignore
    :param per_instance: boolean, if True, dice was calculated per instance instead of per batch
    :return: dice score.
