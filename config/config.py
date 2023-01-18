#!/usr/bin/env python3
import os, sys
import yaml
from collections import OrderedDict
from subprocess import check_output
from filelock import FileLock

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs

# https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

# https://ttl255.com/yaml-anchors-and-aliases-and-how-to-disable-them/
def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        def ignore_aliases(self, data):
            return True
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

def read_config_yaml(config_file):
    config_file_lock = config_file + ".lock"
    lock = FileLock(config_file_lock, timeout=30)

    if os.path.isfile(config_file):
        with lock:
            with open(config_file) as file:
                # config_dict = yaml.load(file, Loader=yaml.Loader)
                config_dict = ordered_load(file, yaml.SafeLoader)
                if config_dict==None:
                    config_dict = OrderedDict()
    else:
        config_dict = OrderedDict()
    return config_dict

def write_config_yaml(config_file, write_dict):
    config_file_lock = config_file + ".lock"
    lock = FileLock(config_file_lock, timeout=30)

    config_dict = read_config_yaml(config_file)
    config_dict.update(write_dict)

    with lock:
        with open(config_file, 'w') as file:
            ordered_dump(config_dict, file, Dumper=yaml.SafeDumper, default_flow_style=False)

def write_config_yaml_with_key_value(config_file, key, value):
    config_file_lock = config_file + ".lock"
    lock = FileLock(config_file_lock, timeout=30)

    config_dict = read_config_yaml(config_file)
    config_dict[key] = value

    with lock:
        with open(config_file, 'w') as file:
            # yaml.dump(config_dict, file, default_flow_style=False)
            ordered_dump(config_dict, file, Dumper=yaml.SafeDumper, default_flow_style=False)

def print_config_yaml(config_file):
    config_dict = read_config_yaml(config_file)
    print(dict(config_dict))

if __name__ == "__main__":

    # Read config_file
    config_file = os.path.join(aoi_dir, 'config/config.yaml')
    config_dict = read_config_yaml(config_file)


    config_dict['production']                        = False
    config_dict['gpu_list']                          = list(range(0,int(check_output("nvidia-smi -L | wc -l", shell=True))))
    config_dict['backup_time']                       = '00_00'
    config_dict['crop_w']                            = 512
    config_dict['crop_h']                            = 512
    config_dict['margin']                            = 100

    config_dict['image_dir']                         = os.path.join(aoi_dir, 'inference_result/factory_img') #input folder
    config_dict['image_backup_dir']                  = os.path.join(aoi_dir, 'inference_result/factory_img_backup')
    config_dict['image_wo_border_dir']               = os.path.join(aoi_dir, 'inference_result/factory_img_wo_border')
    config_dict['inference_result_image_dir']        = os.path.join(aoi_dir, 'inference_result/factory_result_img')
    config_dict['inference_result_image_backup_dir'] = os.path.join(aoi_dir, 'inference_result/factory_result_img_backup')
    config_dict['inference_result_label_dir']        = os.path.join(aoi_dir, 'inference_result/factory_result_json')
    config_dict['inference_result_txt_dir']          = os.path.join(aoi_dir, 'inference_result/factory_result_txt')
    config_dict['inference_log_file_path']           = os.path.join(aoi_dir, 'inference_result/inference_result.log')

    # CenterNet2
    config_dict['centernet2_yaml_file_path']         = os.path.join(aoi_dir, 'detectron2/projects/CenterNet2/configs/ctr2_retrain1_bs8_20211027.yaml')

    centernet2_model_file_dir = os.path.join(aoi_dir, 'models/update/CenterNet2')
    for model_file_name in os.listdir(centernet2_model_file_dir):
        ext = os.path.splitext(model_file_name)[1]
        if ext == '.pth':
            centernet2_model_file_path = os.path.join(centernet2_model_file_dir, model_file_name)
            break

    assert "centernet2_model_file_path" in locals(), "centernet2_model_file_path does not exist."

    config_dict['centernet2_model_file_path']        = centernet2_model_file_path
    config_dict['centernet2_label_dir']              = os.path.join(aoi_dir, 'inference_result/centernet2_json')

    # RetinaNet
    config_dict['retinanet_yaml_file_path']          = os.path.join(aoi_dir, 'detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x_PCB.yaml')

    retinanet_model_file_dir = os.path.join(aoi_dir, 'models/update/RetinaNet')
    for model_file_name in os.listdir(retinanet_model_file_dir):
        ext = os.path.splitext(model_file_name)[1]
        if ext == '.pth':
            retinanet_model_file_path = os.path.join(retinanet_model_file_dir, model_file_name)
            break

    assert "retinanet_model_file_path" in locals(), "retinanet_model_file_path does not exist."

    config_dict['retinanet_model_file_path']         = retinanet_model_file_path
    config_dict['retinanet_label_dir']               = os.path.join(aoi_dir, 'inference_result/retinanet_json')

    # YoloV4
    # setup yolov4 data file
    config_dict['yolov4_data_file_path']             = os.path.join(aoi_dir, 'YOLO_detection/cfg/pcb.data')
    fp = open(config_dict['yolov4_data_file_path'] , "w")
    fp.write("classes = %d\n" % (4))
    fp.write("train = %s\n" % (os.path.join(aoi_dir, 'YOLO_detection/cfg/train.txt')))
    fp.write("valid = %s\n" % (os.path.join(aoi_dir, 'YOLO_detection/cfg/valid.txt')))
    fp.write("names = %s\n" % (os.path.join(aoi_dir, 'YOLO_detection/cfg/pcb.names')))
    fp.write("backup = %s\n" % (os.path.join(aoi_dir, 'YOLO_detection/weights')))
    fp.write("eval = pcb-512\n")
    fp.close()

    yolov4_name_file_path = os.path.join(aoi_dir, 'YOLO_detection/cfg/pcb.names')

    config_dict['yolov4_cfg_file_path']              = os.path.join(aoi_dir, 'YOLO_detection/cfg/yolov4-pcb.cfg')

    yolov4_model_file_dir = os.path.join(aoi_dir, 'models/update/YoloV4')
    for model_file_name in os.listdir(yolov4_model_file_dir):
        ext = os.path.splitext(model_file_name)[1]
        if ext == '.weights':
            yolov4_model_file_path = os.path.join(yolov4_model_file_dir, model_file_name)
            break

    assert "yolov4_model_file_path" in locals(), "yolov4_model_file_path does not exist."

    config_dict['yolov4_model_file_path']            = yolov4_model_file_path
    config_dict['yolov4_label_dir']                  = os.path.join(aoi_dir, 'inference_result/yolov4_json')

    print(dict(config_dict))
    write_config_yaml(config_file, config_dict)