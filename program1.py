import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_executor as runtime
import tvm.auto_scheduler as auto_scheduler
import sqlite3
import os
from pathlib import Path
import onnx
import torch
from torchvision import models
from torch import nn
import argparse

def main(model, target_device, batch_size):
    if target_device == "cuda":
        target = tvm.target.Target("cuda")
    else:
        target = tvm.target.Target("llvm")

    db_path = os.path.expanduser('~/automate_tvm.db')

    tuning_option = {
        "log_filename": f"{model}.log",
        "tuner": "xgb_rank_itervar",
        "n_trial": 1500,
        "early_stopping": 300,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=5, repeat=1, timeout=4, min_repeat_ms=150),
        ),
    }
    lib = tune(tuning_option, batch_size, model, target_device, db_path, target)
    if lib:
        print("Tuning and compilation successful.")

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        if n_layer == 18:
            model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').eval()
        elif n_layer == 34:
            model = models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1').eval()
        elif n_layer == 50:
            model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1').eval()
        elif n_layer == 101:
            model = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V1').eval()
        elif n_layer == 152:
            model = models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V1').eval()
        else:
            raise ValueError("Unsupported model layers: " + str(n_layer))

    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        if n_layer == 11:
            model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1').eval()
        elif n_layer == 13:
            model = models.vgg13(weights='VGG13_Weights.IMAGENET1K_V1').eval()
        elif n_layer == 16:
            model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1').eval()
        elif n_layer == 19:
            model = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').eval()
        else:
            raise ValueError("Unsupported model layers: " + str(n_layer))

    elif name == "mobilenet":
        model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1').eval()
    
    elif "squeezenet" in name:
        model = models.squeezenet1_1(weights='SqueezeNet1_1_Weights.IMAGENET1K_V1').eval()
    
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1').eval()
    
    elif name == "vit":
        model_path = "/model.onnx"
        onnx_model = onnx.load(model_path)
        input_names = [input.name for input in onnx_model.graph.input]
        print("Input names in ONNX model:", input_names)

        shape_dict = {'pixel_values': input_shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        return mod, params, input_shape, output_shape
        
    else:
        raise ValueError("Unsupported network: " + name)
    
    shape_list = [('data', input_shape)]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    return mod, params, input_shape, output_shape

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb_rank_binary_itervar",
    n_trial=1000,
    early_stopping=300,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def tune(tuning_opt, batch_size, model, target_device, db_path, target):
    if model_exists_in_db(target_device, model, batch_size, db_path):
        print(f"Model {model} with batch size {batch_size} already exists in the database.")
        
    else:
        print("Extract tasks...")
        mod, params, input_shape, output_shape = get_network(model, batch_size)
        tasks = autotvm.task.extract_from_program(
                mod["main"], target=target, params=params, ops=None
        )
        log_file = tuning_opt["log_filename"]
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Ensure the log file exists
        if not os.path.isfile(log_file):
            Path(log_file).touch()
        
        print(f"Start tuning {model} for {target_device} with batch size {batch_size}...")
        tune_tasks(tasks, **tuning_opt)
        with autotvm.apply_history_best(log_file):
            print("Compile...")
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)
        serealize_lib_to_database(target_device, model, batch_size, lib, db_path)
        return lib

def serealize_lib_to_database(target_device, model, batch_size, lib, db_path):
    """Serialize the compiled library to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create table if not exists
    cursor.execute('''CREATE TABLE IF NOT EXISTS models
                      (target_device TEXT, model TEXT, batch_size INTEGER, lib BLOB)''')
    # Check if the entry already exists
    cursor.execute('''SELECT * FROM models WHERE target_device=? AND model=? AND batch_size=?''',
                   (target_device, model, batch_size))
    if cursor.fetchone() is None:
        # Insert new entry
        cursor.execute('''INSERT INTO models (target_device, model, batch_size, lib)
                          VALUES (?, ?, ?, ?)''',
                       (target_device, model, batch_size, lib.export_library()))
        conn.commit()
    conn.close()

def model_exists_in_db(target_device, model, batch_size, db_path):
    """Check if the model already exists in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='device_models';")
    if cursor.fetchone() is None:
        print("Table 'device_models' doesn't exist in the database.")
        conn.close()
        return False

    # If the table exists, check if the specific model exists
    cursor.execute('''SELECT * FROM device_models WHERE target_device=? AND model=? AND batch_size=?''',
                   (target_device, model, batch_size))
    exists = cursor.fetchone() is not None
    if not exists:
        print(f"Model '{model}' with batch size {batch_size} doesn't exist in the database.")
    
    conn.close()
    return exists

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tune and compile models using TVM.')
    parser.add_argument('model', type=str, help='Name of the model to tune')
    parser.add_argument('target_device', type=str, help='Target device for tuning (e.g., cuda, llvm)')
    parser.add_argument('batch_size', type=int, help='Batch size for the model')

    args = parser.parse_args()
    main(args.model, args.target_device, args.batch_size)
