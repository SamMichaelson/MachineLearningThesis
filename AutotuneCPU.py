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
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner

import torch
from torchvision import models
from torchvision.io import read_image
from torchvision import transforms
from torch import Tensor
from torch import nn
import torch.backends.cudnn as cudnn
def main():
    model = "resnet-18"
    target = "llvm"
    if str(target).split()[0] == "cuda":
        target_device = "cuda"
    else:
        target_device = "llvm"
    num_threads = 1
    os.environ["TVM_NUM_THREADS"] = str(num_threads)
    log_file = "%s.log" % model
    graph_opt_sch_file = "%s_graph_opt.log" % model
    batch_size = 2
    dtype = "float32"
    db_path = '/home1/public/misampson/resnet-50/git/ITE-Forth-CARV/tvm_report/automate_tvm.db'

    tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "early_stopping": 5,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
        ),
    ),
}
    lib =tune(tuning_option, batch_size, model, target_device, db_path, target,graph_opt_sch_file)
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
def tune_kernels(
    tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(task, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(task, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(task, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(task, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(task, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(task, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(task, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(task, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )
def tune_graph(graph, dshape, records, opt_sch_file, target, use_DP=True):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {"data": dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

def tune(tuning_opt, batch_size, model, target_device, db_path, target,graph_opt_sch_file):
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
        tune_kernels(tasks, **tuning_opt)
        tune_graph(mod["main"], input_shape, log_file, graph_opt_sch_file,target)
       
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
