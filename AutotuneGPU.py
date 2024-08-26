import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_executor as runtime
import tvm.auto_scheduler as auto_scheduler
import sqlite3
import os
import random
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import timeit
import numpy as np
from pathlib import Path
import urllib.request
import onnx

import torch
from torchvision import models
from torchvision.io import read_image
from torchvision import transforms
from torch import Tensor
from torch import nn
import torch.backends.cudnn as cudnn

def main():
    model = "squeezenet_v1.1"
    target = tvm.target.Target("cuda")
    if str(target).split()[0] == "cuda":
        target_device = "cuda"
    else:
        target_device = "llvm"

    batch_size = 2
    dtype = "float32"
    db_path = '/home1/public/misampson/resnet-50/git/ITE-Forth-CARV/tvm_report/automate_tvm.db'

    tuning_option = {
        "log_filename": f"{model}.log",
        "tuner": "ga",
        "n_trial": 2000,
        "early_stopping": 2,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }
    lib =tune(tuning_option, batch_size, model, target_device, db_path, target)
    if ( lib ):
        print(lib)
    

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
    elif name == "squeezenet_v1.1":
        model = models.squeezenet1_1(weights='SqueezeNet1_1_Weights.IMAGENET1K_V1').eval()
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1').eval()
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
    tuner="ga",
    n_trial=1000,
    early_stopping=2,
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
                mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
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
        
def serealize_lib_to_database(device, network, batch_size, lib, db_path):
    lib_path = f'/home1/public/misampson/resnet-50/git/ITE-Forth-CARV/tvm_report/automated_database/{device}/{network}/{batch_size}'
    os.makedirs(lib_path, exist_ok=True)

    file_name = "deploy.so"
    path_lib = os.path.join(lib_path, file_name)
    lib.export_library(path_lib)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO device_models (device, model, batch_size)
    VALUES (?, ?, ?)
    ''', (device, network, batch_size))

    conn.commit()
    conn.close()

def model_exists_in_db(device, network, batch_size, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = '''
    SELECT COUNT(*) FROM device_models
    WHERE device = ? AND model = ? AND batch_size = ?
    '''
    cursor.execute(query, (device, network, batch_size))
    result = cursor.fetchone()[0]

    conn.close()
    return result > 0
    

if __name__ == "__main__":
    main()
