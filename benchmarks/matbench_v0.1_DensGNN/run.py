import os.path
import argparse
import pandas as pd
import tensorflow as tf
from matbench.bench import MatbenchBenchmark
from kgcnn.data.crystal import CrystalDataset
from kgcnn.literature.DenseGNN import make_model_asu

from sklearn.preprocessing import StandardScaler
from kgcnn.training.schedule import LinearWarmupExponentialDecay
from kgcnn.training.scheduler import LinearLearningRateScheduler
import kgcnn.training.callbacks
from kgcnn.utils.devices import set_devices_gpu
import numpy as np
from copy import deepcopy
from hyper import *

parser = argparse.ArgumentParser(description='Train DenseGNN.')
parser.add_argument("--gpu", required=False, help="GPU index used for training.",
                    default=None, nargs="+", type=int)
args = vars(parser.parse_args())
print("Input of argparse:", args)
gpu_to_use = args["gpu"]
set_devices_gpu(gpu_to_use)

subsets_compatible = ["matbench_jdft2d", "matbench_phonons", "matbench_mp_e_form", "matbench_mp_gap", 
                      "matbench_perovskites",
                      "matbench_log_kvrh", "matbench_log_gvrh", "matbench_dielectric"]
mb = MatbenchBenchmark(subset=subsets_compatible, autoload=False)

callbacks = {
    "graph_labels": lambda st, ds: np.expand_dims(ds, axis=-1),
    "node_coordinates": lambda st, ds: np.array(st.cart_coords, dtype="float"),
    "node_frac_coordinates": lambda st, ds: np.array(st.frac_coords, dtype="float"),
    "graph_lattice": lambda st, ds: np.ascontiguousarray(np.array(st.lattice.matrix), dtype="float"),
    "abc": lambda st, ds: np.array(st.lattice.abc),
    "charge": lambda st, ds: np.array([st.charge], dtype="float"),
    "volume": lambda st, ds: np.array([st.lattice.volume], dtype="float"),
    "node_number": lambda st, ds: np.array(st.atomic_numbers, dtype="int"),
}

hyper_all = {
    "matbench_jdft2d": hyper_1,
    "matbench_phonons": hyper_2,
    "matbench_mp_e_form": hyper_3,
    "matbench_mp_gap": hyper_4,
    "matbench_perovskites": hyper_5,
    "matbench_log_kvrh": hyper_6,
    "matbench_log_gvrh": hyper_7,
    "matbench_dielectric": hyper_8,
}

restart_training = True
remove_invalid_graphs_on_predict = True

for idx_task, task in enumerate(mb.tasks):
    task.load()
    for i, fold in enumerate(task.folds):
        hyper = deepcopy(hyper_all[task.dataset_name])

        # Define loss for either classification or regression
        loss = {
            "class_name": "BinaryCrossentropy", "config": {"from_logits": True}
        } if task.metadata["task_type"] == "classification" else "mean_absolute_error"
        hyper["training"]["compile"]["loss"] = loss

        if restart_training and os.path.exists(
                "%s_predictions_%s_fold_%s.npy" % (task.dataset_name, hyper["model"]["config"]["name"], i)):
            predictions = np.load(
                "%s_predictions_%s_fold_%s.npy" % (task.dataset_name, hyper["model"]["config"]["name"], i)
            )
            task.record(fold, predictions)
            continue

        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        data_train = CrystalDataset()

        data_train._map_callbacks(train_inputs, pd.Series(train_outputs.values), callbacks)
        print("Making graph... (this may take a while)")
        data_train.set_methods(hyper["data"]["dataset"]["methods"])
        data_train.clean(hyper["model"]["config"]["inputs"])

        y_train = np.array(data_train.get("graph_labels"))
        x_train = data_train.tensor(hyper["model"]["config"]["inputs"])

        if task.metadata["task_type"] == "classification":
            scaler = None
        else:
            scaler = StandardScaler(**hyper["training"]["scaler"]["config"])
            y_train = scaler.fit_transform(y_train)
        print(y_train.shape)

        # train and validate your model
        model = make_model_asu(**hyper["model"]["config"])
        model.compile(
            loss=tf.keras.losses.get(hyper["training"]["compile"]["loss"]),
            optimizer=tf.keras.optimizers.get(hyper["training"]["compile"]["optimizer"])
        )
        hist = model.fit(
            x_train, y_train,
            batch_size=hyper["training"]["fit"]["batch_size"],
            epochs=hyper["training"]["fit"]["epochs"],
            verbose=hyper["training"]["fit"]["verbose"],
            callbacks=[tf.keras.utils.deserialize_keras_object(x) for x in hyper["training"]["fit"]["callbacks"]]
        )

        # Get testing data
        test_inputs = task.get_test_data(fold, include_target=False)
        data_test = CrystalDataset()
        data_test._map_callbacks(test_inputs, pd.Series(np.zeros(len(test_inputs))), callbacks)
        print("Making graph... (this may take a while)")
        data_test.set_methods(hyper["data"]["dataset"]["methods"])

        if remove_invalid_graphs_on_predict:
            removed = data_test.clean(hyper["model"]["config"]["inputs"])
            np.save(
                "%s_predictions_invalid_%s_fold_%s.npy" % (task.dataset_name, hyper["model"]["config"]["name"], i),
                removed
            )
        else:
            removed = None

        # Predict on the testing data
        x_test = data_test.tensor(hyper["model"]["config"]["inputs"])
        predictions_model = model.predict(x_test)

        if remove_invalid_graphs_on_predict:
            indices_test = [j for j in range(len(test_inputs))]
            for j in removed:
                indices_test.pop(j)
            predictions = np.expand_dims(np.zeros(len(test_inputs), dtype="float"), axis=-1)
            predictions[np.array(indices_test)] = predictions_model
        else:
            predictions = predictions_model

        if task.metadata["task_type"] == "classification":
            def np_sigmoid(x):
                return np.exp(-np.logaddexp(0, -x))
            predictions = np_sigmoid(predictions)
        else:
            predictions = scaler.inverse_transform(predictions)

        if predictions.shape[-1] == 1:
            predictions = np.squeeze(predictions, axis=-1)

        np.save(
            "%s_predictions_%s_fold_%s.npy" % (task.dataset_name, hyper["model"]["config"]["name"], i),
            predictions
        )

        # Record data!
        task.record(fold, predictions)

# Save your results
mb.to_file("results_densegnn.json.gz")

for key, values in mb.scores.items():
    factor = 1000.0 if key in ["matbench_jdft2d"] else 1.0
    if key not in ["matbench_mp_is_metal"]:
        print(key, factor*values["mae"]["mean"], factor*values["mae"]["std"])
    else:
        print(key, values["rocauc"]["mean"],  values["rocauc"]["std"])
