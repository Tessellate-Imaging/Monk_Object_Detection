from imports import *


def get_base_system_dict():
    system_dict = {};

    system_dict["cwd"] = False;
    system_dict["master_systems_dir"] = False

    system_dict["current_algo_type"] = False
    system_dict["current_algo"] = False
    system_dict["current_data"] = False
    system_dict["current_model"] = False
    system_dict["current_installation"] = False
    system_dict["current_venv"] = False
    system_dict["current_image_input_path"] = False;
    system_dict["current_image_output_path"] = False;
    system_dict["current_folder_input_path"] = False;
    system_dict["current_folder_output_path"] = False;
    system_dict["current_video_input_path"] = False;
    system_dict["current_video_output_path"] = False;
    system_dict["current_video_fps"] = False;
    system_dict["current_video_merge"] = False;
    system_dict["current_input_folder"] = False;
    system_dict["current_output_folder"] = False;
    system_dict["current_visualize"] = False;



    system_dict["algo_types"] = ["object_detection", "image_segmentation",
                                    "pose_estimation", "face_recognition",
                                    "facial_keypoint_detection", "activity_classification",
                                    "gaze_estimation", "object_tracking"]
    
    system_dict["object_detection"] = {};
    system_dict["image_segmentation"] = {};
    system_dict["pose_estimation"] = {};
    system_dict["face_recognition"] = {};
    system_dict["facial_keypoint_detection"] = {};
    system_dict["activity_classification"] = {};
    system_dict["gaze_estimation"] = {};
    system_dict["object_tracking"] = {};





    system_dict["object_detection"]["all_algo"] = ["gluoncv_finetune", "efficientdet_pytorch"];
    system_dict["object_detection"]["all_venv"] = ["gluoncv_finetune", "efficientdet_pytorch"];
    system_dict["object_detection"]["all_installation"] = [False, False];
    
    system_dict["object_detection"]["gluoncv_finetune"] = {};
    system_dict["object_detection"]["gluoncv_finetune"]["all_model"] = [
        ["coco", "ssd_300_vgg16_atrous", "25.1"],
        ["coco", "ssd_512_vgg16_atrous", "28.9"],
        ["coco", "ssd_512_resnet50_v1", "30.6"],
        ["coco", "ssd_512_mobilenet1.0", "21.7"],

        ["coco", "faster_rcnn_resnet50_v1b", "37.1"],
        ["coco", "faster_rcnn_resnet101_v1d", "40.1"],
        ["coco", "faster_rcnn_fpn_resnet50_v1b", "38.4"],
        ["coco", "faster_rcnn_fpn_resnet101_v1d", "40.8"],
        ["coco", "faster_rcnn_fpn_syncbn_resnest50", "42.7"],
        ["coco", "faster_rcnn_fpn_syncbn_resnest101", "44.9"],

        ["coco", "yolo3_darknet53", "37.0"],
        ["coco", "yolo3_mobilenet1.0", "28.0"],

        ["coco", "center_net_resnet18_v1b", "33.6"],
        ["coco", "center_net_resnet50_v1b", "37.0"],
        ["coco", "center_net_resnet101_v1b", "28.6"],

        ["voc", "ssd_300_vgg16_atrous", "77.6"],
        ["voc", "ssd_512_vgg16_atrous", "79.2"],
        ["voc", "ssd_512_resnet50_v1", "80.1"],
        ["voc", "ssd_512_mobilenet1.0", "78.54"],

        ["voc", "faster_rcnn_resnet50_v1b", "78.3"],

        ["voc", "yolo3_darknet53", "81.5"],
        ["voc", "yolo3_mobilenet1.0", "75.8"],

        ["voc", "center_net_resnet18_v1b", "66.8"],
        ["voc", "center_net_resnet50_v1b.0_320", "71.8"],
        ["voc", "center_net_resnet101_v1b", "75.5"]

    ]

    system_dict["object_detection"]["efficientdet_pytorch"] = {};
    system_dict["object_detection"]["efficientdet_pytorch"]["all_model"] = [
        ["coco", "efficientdet-d0", "33.1"],
        ["coco", "efficientdet-d1", "38.8"],
        ["coco", "efficientdet-d2", "42.1"],
        ["coco", "efficientdet-d3", "45.6"],
        ["coco", "efficientdet-d4", "48.8"],
        ["coco", "efficientdet-d5", "50.2"],
        ["coco", "efficientdet-d6", "50.7"],
        ["coco", "efficientdet-d7", "51.2"]
    ]
    
    system_dict["results"] = {};
    system_dict["results"]["object_detection_dir"] = False;
    system_dict["results"]["image_segmentation_dir"] = False;
    system_dict["results"]["pose_estimation_dir"] = False;
    system_dict["results"]["face_recognition_dir"] = False;
    system_dict["results"]["facial_keypoint_detection_dir"] = False;
    system_dict["results"]["activity_classification_dir"] = False;
    system_dict["results"]["gaze_estimation_dir"] = False;
    system_dict["results"]["object_tracking_dir"] = False;


    system_dict["result_dir_master"] = False;
    system_dict["result_dir_model"] = False;
    system_dict["result_dir_sub"] = False;


    system_dict["coco_classes"] = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove',
                                    'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl',
                                    'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair',
                                    'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant',
                                    'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse',
                                    'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle',
                                    'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant',
                                    'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard',
                                    'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase',
                                    'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 
                                    'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase',
                                    'wine glass', 'zebra']

    system_dict["voc_classes"] = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                                    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                                    'sheep', 'sofa', 'train', 'tvmonitor']


    return system_dict;