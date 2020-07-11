from imports import *
from base_class import system


class Infer(system):
    def __init__(self):
        super().__init__()

    def List_Algo_Types(self):
        self.print_available_algorithm_types();

    def List_Algos(self, algo_type="object_detection"):
        if(algo_type == "object_detection"):
            self.print_object_detection_algorithms();
        else:
            print("Algo type - {} has no algo added yet.".format(algo_type));
            print("Development in progress");

    def Install_Engine(self, algo_type="object_detection", algo="gluoncv_finetune", system="cuda-9.0"):
        if(algo_type == "object_detection"):
            if(algo == "gluoncv_finetune"):
                self.install_gluoncv_finetune(system);
            elif(algo == "efficientdet_pytorch"):
                self.install_efficientdet_pytorch(system);

    def List_Model_Names(self, algo_type="object_detection", algo="gluoncv_finetune"):
        if(algo_type == "object_detection"):
            if(algo == "gluoncv_finetune"):
                self.print_gluoncv_finetune_model_names();
            elif(algo == "efficientdet_pytorch"):
                self.print_efficientdet_pytorch_model_names();
        else:
            print("Algo type - {}, Algo - {} unimplemented".format(algo_type, algo));
            print("Development in progress");

    def List_Classes(self, data="coco"):
        self.print_class_list(data=data);

    def Infer_Image(self, algo_type="object_detection", 
                        algo="gluoncv_finetune",
                        data="coco",
                        model="faster_rcnn_fpn_syncbn_resnest269",
                        img_path="5.png",
                        classes="",
                        thresh=0.5,
                        visualize=True,
                        write_voc_format=False,
                        write_coco_format=False,
                        write_monk_format=True,
                        write_yolo_format=False,
                        save_output_img = True,
                        verbose=1):

        check_status = self.check_inputs_image(algo_type, algo, data, model, img_path, classes);


        if(check_status):
            if(not save_output_img):
                visualize=False;

            self.system_dict["result_dir_master"] = self.system_dict["results"]["object_detection_dir"] + "/" + algo + "/";
            self.system_dict["result_dir_model"] = self.system_dict["result_dir_master"] + "/" + model + "/";


            if not os.path.isdir(self.system_dict["result_dir_master"]):
                os.mkdir(self.system_dict["result_dir_master"]);


            if not os.path.isdir(self.system_dict["result_dir_model"]):
                os.mkdir(self.system_dict["result_dir_model"]); 

            self.system_dict["current_algo_type"] = algo_type;
            self.system_dict["current_algo"] = algo;
            self.system_dict["current_data"] = data;
            self.system_dict["current_model"] = model;
            self.system_dict["current_image_input_path"] = img_path;
            self.system_dict["current_image_output_path"] = self.system_dict["result_dir_model"] + "/" + img_path.split(".")[0].split("/")[-1] + ".jpg";
            self.system_dict["current_venv"] = algo;
            self.system_dict["current_visualize"] = visualize;
            self.system_dict["current_classes"] = classes;


            f = open("test_img.txt", 'w');
            f.write(self.system_dict["current_image_input_path"] + "\n");
            f.write(self.system_dict["current_data"] + "\n");
            f.write(self.system_dict["current_model"] + "\n");
            f.write(self.system_dict["current_image_output_path"] + "\n");
            f.write(str(thresh) + "\n");
            f.write(classes + "\n");
            if(write_voc_format):
                f.write("1\n");
            else:
                f.write("0\n");
            if(write_coco_format):
                f.write("1\n");
            else:
                f.write("0\n");
            if(write_monk_format):
                f.write("1\n");
            else:
                f.write("0\n");
            if(write_yolo_format):
                f.write("1\n");
            else:
                f.write("0\n");
            if(visualize):
                f.write("1");
            else:
                f.write("0");
            f.close();

            self.infer_image(algo_type, algo, model, verbose);
            if(not save_output_img):
                os.system("rm " + self.system_dict["current_image_output_path"])
            os.system("rm test_img.txt")
            print("Output stored at {}".format(self.system_dict["result_dir_model"].replace("//", "/")));



    def Infer_Folder_Images(self, algo_type="object_detection", 
                        algo="gluoncv_finetune",
                        data="coco",
                        model="faster_rcnn_fpn_syncbn_resnest269",
                        folder_path="data",
                        output_folder_name=None,
                        classes="",
                        thresh=0.5,
                        visualize=True,
                        write_voc_format=False,
                        write_coco_format=False,
                        write_monk_format=True,
                        write_yolo_format=False,
                        save_output_img = True,
                        verbose=1):

        check_status = self.check_inputs_folder(algo_type, algo, data, model, folder_path, classes);

        if(check_status):
            if(not save_output_img):
                visualize=False;

            self.system_dict["result_dir_master"] = self.system_dict["results"]["object_detection_dir"] + "/" + algo + "/";
            self.system_dict["result_dir_model"] = self.system_dict["result_dir_master"] + "/" + model + "/";

            if(output_folder_name):
                self.system_dict["result_dir_sub"] = self.system_dict["result_dir_model"] + "/" + output_folder_name + "/";
            else:
                output_folder_name = folder_path.split("/")[-1];
                self.system_dict["result_dir_sub"] = self.system_dict["result_dir_model"] + "/" + output_folder_name + "/";


            if not os.path.isdir(self.system_dict["result_dir_master"]):
                os.mkdir(self.system_dict["result_dir_master"]);



            if not os.path.isdir(self.system_dict["result_dir_model"]):
                os.mkdir(self.system_dict["result_dir_model"]); 

            if not os.path.isdir(self.system_dict["result_dir_sub"]):
                os.mkdir(self.system_dict["result_dir_sub"]); 


            self.system_dict["current_algo_type"] = algo_type;
            self.system_dict["current_algo"] = algo;
            self.system_dict["current_data"] = data;
            self.system_dict["current_model"] = model;
            self.system_dict["current_folder_input_path"] = folder_path;
            self.system_dict["current_folder_output_path"] = self.system_dict["result_dir_sub"] + "/";
            self.system_dict["current_venv"] = algo;
            self.system_dict["current_visualize"] = visualize;
            self.system_dict["current_classes"] = classes;
        
            f = open("test_folder.txt", 'w');
            f.write(self.system_dict["current_folder_input_path"] + "\n");
            f.write(self.system_dict["current_data"] + "\n");
            f.write(self.system_dict["current_model"] + "\n");
            f.write(self.system_dict["current_folder_output_path"] + "\n");
            f.write(str(thresh) + "\n");
            f.write(classes + "\n");
            if(write_voc_format):
                f.write("1\n");
            else:
                f.write("0\n");
            if(write_coco_format):
                f.write("1\n");
            else:
                f.write("0\n");
            if(write_monk_format):
                f.write("1\n");
            else:
                f.write("0\n");
            if(write_yolo_format):
                f.write("1\n");
            else:
                f.write("0\n");
            if(visualize):
                f.write("1\n");
            else:
                f.write("0\n");
            if(save_output_img):
                f.write("1");
            else:
                f.write("0");
            f.close();

            self.infer_folder_image(algo_type, algo, model, verbose);
            os.system("rm test_folder.txt");
            print("Output stored at {}".format(self.system_dict["result_dir_sub"].replace("//", "/")));



    def Infer_Video(self, algo_type="object_detection", 
                 algo="gluoncv_finetune",
                 data="coco",
                 model="ssd_512_resnet50_v1",
                 video_path="video.mp4",
                 fps=1,
                 merge_video=True,
                 output_folder_name="day_bicycle",
                 thresh=0.5,
                 classes = "",
                 write_voc_format=True,
                 write_coco_format=True,
                 write_monk_format=True,
                 write_yolo_format=True,
                 save_output_img = True,
                 verbose=1):

        check_status = self.check_inputs_video(algo_type, algo, data, model, video_path, classes);

        if(check_status):
            if(not save_output_img):
                visualize=False;

            self.system_dict["result_dir_master"] = self.system_dict["results"]["object_detection_dir"] + "/" + algo + "/";
            self.system_dict["result_dir_model"] = self.system_dict["result_dir_master"] + "/" + model + "/";

            if(output_folder_name):
                self.system_dict["result_dir_sub"] = self.system_dict["result_dir_model"] + "/" + output_folder_name + "/";
            else:
                output_folder_name = video_path.split("/")[-1].split(".")[0];
                self.system_dict["result_dir_sub"] = self.system_dict["result_dir_model"] + "/" + output_folder_name + "/";


            if not os.path.isdir(self.system_dict["result_dir_master"]):
                os.mkdir(self.system_dict["result_dir_master"]);



            if not os.path.isdir(self.system_dict["result_dir_model"]):
                os.mkdir(self.system_dict["result_dir_model"]); 

            if not os.path.isdir(self.system_dict["result_dir_sub"]):
                os.mkdir(self.system_dict["result_dir_sub"]); 

            self.system_dict["current_algo_type"] = algo_type;
            self.system_dict["current_algo"] = algo;
            self.system_dict["current_data"] = data;
            self.system_dict["current_model"] = model;
            self.system_dict["current_video_input_path"] = video_path;
            self.system_dict["current_video_output_path"] = self.system_dict["result_dir_sub"] + "/";
            self.system_dict["current_video_fps"] = fps;
            self.system_dict["current_video_merge"] = merge_video;
            self.system_dict["current_venv"] = algo;
            self.system_dict["current_visualize"] = False;
            self.system_dict["current_classes"] = classes;
        
            f = open("test_folder.txt", 'w');
            f.write("tmp_video\n");
            f.write(self.system_dict["current_data"] + "\n");
            f.write(self.system_dict["current_model"] + "\n");
            f.write(self.system_dict["current_video_output_path"] + "\n");
            f.write(str(thresh) + "\n");
            f.write(classes + "\n");
            if(write_voc_format):
                f.write("1\n");
            else:
                f.write("0\n");
            if(write_coco_format):
                f.write("1\n");
            else:
                f.write("0\n");
            if(write_monk_format):
                f.write("1\n");
            else:
                f.write("0\n");
            if(write_yolo_format):
                f.write("1\n");
            else:
                f.write("0\n");
            f.write("0\n");
            if(save_output_img):
                f.write("1");
            else:
                f.write("0");
            f.close();

            self.infer_video(algo_type, algo, model, verbose);
            os.system("rm test_folder.txt");
            os.system("rm -r tmp_video");
            print("Output stored at {}".format(self.system_dict["result_dir_sub"].replace("//", "/")));



    def Compare_On_Image(self, algo_type="object_detection", 
                            model_list="",
                            img_path="5.png",
                            thresh=0.5,
                            classes = "",
                            visualize=True,
                            write_voc_format=True,
                            write_coco_format=True,
                            write_monk_format=True,
                            write_yolo_format=True,
                            verbose=1):

        for i in range(len(model_list)):
            check_status = self.check_inputs_image(algo_type, model_list[i][0], 
                                                    model_list[i][1], model_list[i][2], 
                                                    img_path, classes);
            if(not check_status):
                break;

        if(check_status):
            for i in range(len(model_list)):
                self.Infer_Image(algo_type=algo_type, 
                                    algo=model_list[i][0],
                                    data=model_list[i][1],
                                    model=model_list[i][2],
                                    img_path=img_path,
                                    classes=classes,
                                    thresh=thresh,
                                    visualize=visualize,
                                    write_voc_format=write_voc_format,
                                    write_coco_format=write_coco_format,
                                    write_monk_format=write_monk_format,
                                    write_yolo_format=write_yolo_format,
                                    verbose=verbose);
