from imports import *
from base_system_dict import get_base_system_dict


class system():
    def __init__(self):
        self.system_dict = get_base_system_dict();
        self.system_dict["cwd"] = os.getcwd() + "/";
        self.system_dict["master_systems_dir"] = "workspace/";
        
        self.system_dict["results"]["object_detection_dir"] = self.system_dict["master_systems_dir"] + "/object_detection/"; 
        self.system_dict["results"]["image_segmentation_dir"] = self.system_dict["master_systems_dir"] + "/image_segmentation/"; 
        self.system_dict["results"]["pose_estimation_dir"] = self.system_dict["master_systems_dir"] + "/pose_estimation/"; 
        self.system_dict["results"]["face_recognition_dir"] = self.system_dict["master_systems_dir"] + "/face_recognition/"; 
        self.system_dict["results"]["facial_keypoint_detection_dir"] = self.system_dict["master_systems_dir"] + "/facial_keypoint_detection/"; 
        self.system_dict["results"]["activity_classification_dir"] = self.system_dict["master_systems_dir"] + "/activity_classification/"; 
        self.system_dict["results"]["gaze_estimation_dir"] = self.system_dict["master_systems_dir"] + "/gaze_estimation/"; 
        self.system_dict["results"]["object_tracking_dir"] = self.system_dict["master_systems_dir"] + "/object_tracking/"; 

        if not os.path.isdir(self.system_dict["master_systems_dir"]):
            os.mkdir(self.system_dict["master_systems_dir"]);

        if not os.path.isdir(self.system_dict["results"]["object_detection_dir"]):
            os.mkdir(self.system_dict["results"]["object_detection_dir"]);
        if not os.path.isdir(self.system_dict["results"]["image_segmentation_dir"]):
            os.mkdir(self.system_dict["results"]["image_segmentation_dir"]);
        if not os.path.isdir(self.system_dict["results"]["pose_estimation_dir"]):
            os.mkdir(self.system_dict["results"]["pose_estimation_dir"]);
        if not os.path.isdir(self.system_dict["results"]["face_recognition_dir"]):
            os.mkdir(self.system_dict["results"]["face_recognition_dir"]);
        if not os.path.isdir(self.system_dict["results"]["facial_keypoint_detection_dir"]):
            os.mkdir(self.system_dict["results"]["facial_keypoint_detection_dir"]);
        if not os.path.isdir(self.system_dict["results"]["activity_classification_dir"]):
            os.mkdir(self.system_dict["results"]["activity_classification_dir"]);
        if not os.path.isdir(self.system_dict["results"]["gaze_estimation_dir"]):
            os.mkdir(self.system_dict["results"]["gaze_estimation_dir"]);
        if not os.path.isdir(self.system_dict["results"]["object_tracking_dir"]):
            os.mkdir(self.system_dict["results"]["object_tracking_dir"]);



    def print_available_algorithm_types(self):
        print("Available Algorithm Types: ");
        for i in range(len(self.system_dict["algo_types"])):
            print("{}. {}".format(i+1, self.system_dict["algo_types"][i]));

    def print_object_detection_algorithms(self):
        print("Available Object Detection Algorithms: ");
        for i in range(len(self.system_dict["object_detection"]["all_algo"])):
            print("{}. {}".format(i+1, self.system_dict["object_detection"]["all_algo"][i]));

    def print_gluoncv_finetune_model_names(self):
        print("Available models: ");
        sorted_list = sorted(self.system_dict["object_detection"]["gluoncv_finetune"]["all_model"], key=lambda x: float(x[2]))
        for i in range(len(sorted_list)):
            print("{}. Data: {}, Model-Name: {}, mAP: {}".format(
                i+1,
                sorted_list[i][0],
                sorted_list[i][1],
                sorted_list[i][2]
                ))

    def print_efficientdet_pytorch_model_names(self):
        print("Available models: ");
        sorted_list = sorted(self.system_dict["object_detection"]["efficientdet_pytorch"]["all_model"], key=lambda x: float(x[2]))
        for i in range(len(sorted_list)):
            print("{}. Data: {}, Model-Name: {}, mAP: {}".format(
                i+1,
                sorted_list[i][0],
                sorted_list[i][1],
                sorted_list[i][2]
                ))

    def install_gluoncv_finetune(self, system):
        venv = self.system_dict["object_detection"]["all_venv"][0];

        if(system == "cuda-9.0"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_cuda90.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_cuda90.sh']
        elif(system == "cuda-9.2"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_cuda92.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_cuda92.sh']
        elif(system == "cuda-10.0"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_cuda100.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_cuda100.sh']
        elif(system == "cuda-10.1"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_cuda101.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_cuda101.sh']
        elif(system == "colab"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_colab.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_colab.sh']
        elif(system == "kaggle"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_kaggle.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_kaggle.sh']
        elif(system == "cpu"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_cpu.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/installation/install_cpu.sh']

        cmd_status = self.run_command(cmd);


        if(cmd_status):
            self.system_dict["object_detection"]["all_installation"][0] = True;


    def install_efficientdet_pytorch(self, system):
        venv = self.system_dict["object_detection"]["all_venv"][1];

        if(system == "cuda-9.0"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_cuda90.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_cuda90.sh']
        elif(system == "cuda-9.2"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_cuda92.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_cuda92.sh']
        elif(system == "cuda-10.0"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_cuda100.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_cuda100.sh']
        elif(system == "cuda-10.1"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_cuda101.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_cuda101.sh']
        elif(system == "colab"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_colab.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_colab.sh']
        elif(system == "kaggle"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_kaggle.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_kaggle.sh']
        elif(system == "cpu"):
            os.system("chmod +x Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_cpu.sh")
            cmd = ['bash', 'Monk_Object_Detection/inference_engine/efficientdet_pytorch/installation/install_cpu.sh']

        cmd_status = self.run_command(cmd);


        if(cmd_status):
            self.system_dict["object_detection"]["all_installation"][1] = True;


    def run_command(self, command, verbose=1):
        process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

        while True:
            out = process.stdout.readline()
            output = out.decode('utf-8')
            output = output[:len(output)-1]
            if out == '' and process.poll() != None:
                return True;
            if out and verbose:
                sys.stdout.write(out)
                sys.stdout.flush()
            if(output == "Completed"):
                return True;

            '''
            out2 = process.stderr.readline()
            output2 = out2.decode('utf-8')
            output2 = output2[:len(output2)-1]
            if out2 == '' and process.poll() != None:
                return False;
            if out2 and verbose:
                sys.stdout.write(out2)
                sys.stdout.flush()
            if("Error" in output2):
                return False;
            '''

        return True;


    def print_class_list(self, data="coco"):
        if(data == "coco"):
            sorted_list = self.system_dict["coco_classes"];
            for i in range(len(sorted_list)):
                print("{}. Class: {}".format(
                    i+1,
                    sorted_list[i]
                    ))
        elif(data == "voc"):
            sorted_list = self.system_dict["voc_classes"];
            for i in range(len(sorted_list)):
                print("{}. Class: {}".format(
                    i+1,
                    sorted_list[i]
                    ))

    def check_inputs_image(self, algo_type, algo, data, model, img_path, classes):
        if(not os.path.isfile(img_path)):
            print("Image {} not found".format(img_path));
            return False;

        if(algo_type not in self.system_dict["algo_types"]):
            print("Algo type {} not found.".format(algo_type));
            self.print_available_algorithm_types();
            return False;

        if(algo_type == "object_detection"):
            if(algo not in self.system_dict["object_detection"]["all_algo"]):
                print("Algo {} not found.".format(algo));
                self.print_object_detection_algorithms();
                return False;

            if(algo == "gluoncv_finetune"):
                if(data not in ["coco", "voc"]):
                    print("Data {} not found".format(data));
                    print("Available data types: {}".format(["coco", "voc"]));
                    return False;

                classes = classes.split(",");

                if(len(classes) > 0 and classes[0] != ""):
                    for i in range(len(classes)):
                        classes[i] = classes[i].replace(" ", "");

                    if(data == "coco"):
                        for i in range(len(classes)):
                            if(classes[i] not in self.system_dict["coco_classes"]):
                                print("Class {} not part of coco dataset".format(classes[i]));
                                self.print_class_list(data="coco");
                                return False;
                    elif(data == "voc"):
                        for i in range(len(classes)):
                            if(classes[i] not in self.system_dict["voc_classes"]):
                                print("Class {} not part of voc dataset".format(classes[i]));
                                self.print_class_list(data="voc");
                                return False;

                available_models = [row[1] for row in self.system_dict["object_detection"]["gluoncv_finetune"]["all_model"]]
                if(model not in available_models):
                    print("Model {} not found".format(model));
                    self.print_gluoncv_finetune_model_names();
                    return False;

        return True;


    def check_inputs_folder(self, algo_type, algo, data, model, folder_path, classes):
        if(not os.path.isdir(folder_path)):
            print("Folder {} not found".format(folder_path));
            return False;

        if(algo_type not in self.system_dict["algo_types"]):
            print("Algo type {} not found.".format(algo_type));
            self.print_available_algorithm_types();
            return False;

        if(algo_type == "object_detection"):
            if(algo not in self.system_dict["object_detection"]["all_algo"]):
                print("Algo {} not found.".format(algo));
                self.print_object_detection_algorithms();
                return False;

            if(algo == "gluoncv_finetune"):
                if(data not in ["coco", "voc"]):
                    print("Data {} not found".format(data));
                    print("Available data types: {}".format(["coco", "voc"]));
                    return False;

                classes = classes.split(",");

                if(len(classes) > 0 and classes[0] != ""):
                    for i in range(len(classes)):
                        classes[i] = classes[i].replace(" ", "");

                    if(data == "coco"):
                        for i in range(len(classes)):
                            if(classes[i] not in self.system_dict["coco_classes"]):
                                print("Class {} not part of coco dataset".format(classes[i]));
                                self.print_class_list(data="coco");
                                return False;
                    elif(data == "voc"):
                        for i in range(len(classes)):
                            if(classes[i] not in self.system_dict["voc_classes"]):
                                print("Class {} not part of voc dataset".format(classes[i]));
                                self.print_class_list(data="voc");
                                return False;

                available_models = [row[1] for row in self.system_dict["object_detection"]["gluoncv_finetune"]["all_model"]]
                if(model not in available_models):
                    print("Model {} not found".format(model));
                    self.print_gluoncv_finetune_model_names();
                    return False;

        return True;


    def check_inputs_video(self, algo_type, algo, data, model, video_path, classes):
        if(not os.path.isfile(video_path)):
            print("Video {} not found".format(video_path));
            return False;

        if(algo_type not in self.system_dict["algo_types"]):
            print("Algo type {} not found.".format(algo_type));
            self.print_available_algorithm_types();
            return False;

        if(algo_type == "object_detection"):
            if(algo not in self.system_dict["object_detection"]["all_algo"]):
                print("Algo {} not found.".format(algo));
                self.print_object_detection_algorithms();
                return False;

            if(algo == "gluoncv_finetune"):
                if(data not in ["coco", "voc"]):
                    print("Data {} not found".format(data));
                    print("Available data types: {}".format(["coco", "voc"]));
                    return False;

                classes = classes.split(",");

                if(len(classes) > 0 and classes[0] != ""):
                    for i in range(len(classes)):
                        classes[i] = classes[i].replace(" ", "");

                    if(data == "coco"):
                        for i in range(len(classes)):
                            if(classes[i] not in self.system_dict["coco_classes"]):
                                print("Class {} not part of coco dataset".format(classes[i]));
                                self.print_class_list(data="coco");
                                return False;
                    elif(data == "voc"):
                        for i in range(len(classes)):
                            if(classes[i] not in self.system_dict["voc_classes"]):
                                print("Class {} not part of voc dataset".format(classes[i]));
                                self.print_class_list(data="voc");
                                return False;

                available_models = [row[1] for row in self.system_dict["object_detection"]["gluoncv_finetune"]["all_model"]]
                if(model not in available_models):
                    print("Model {} not found".format(model));
                    self.print_gluoncv_finetune_model_names();
                    return False;

        return True;


    def infer_image(self, algo_type, algo, model, verbose):
        if(algo_type == "object_detection"):
            if(algo == "gluoncv_finetune"):
                if("ssd" in model):
                    os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_img_ssd.sh");
                    cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_img_ssd.sh'];
                if("rcnn" in model):
                    os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_img_rcnn.sh");
                    cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_img_rcnn.sh'];
                if("yolo" in model):
                    os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_img_yolo.sh");
                    cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_img_yolo.sh'];
                if("center_net" in model):
                    os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_img_center_net.sh");
                    cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_img_center_net.sh'];
            elif(algo == "efficientdet_pytorch"):
                os.system("chmod +x Monk_Object_Detection/inference_engine/efficientdet_pytorch/lib/infer_img_efficientdet.sh");
                cmd = ['bash', 'Monk_Object_Detection/inference_engine/efficientdet_pytorch/lib/infer_img_efficientdet.sh'];



        cmd_status = self.run_command(cmd, verbose=verbose);


        
        if(cmd_status):
            if(self.system_dict["current_visualize"]):
                img=mpimg.imread(self.system_dict["current_image_output_path"])
                plt.figure(figsize=(20,10))
                imgplot = plt.imshow(img)
                plt.show()
        


    def infer_folder_image(self, algo_type, algo, model, verbose):
        if(not os.path.isfile(self.system_dict["current_folder_output_path"] + "/status.csv")):
            combined = [];
            all_imgs = sorted(os.listdir(self.system_dict["current_folder_input_path"]));
            for i in range(len(all_imgs)):
                combined.append([all_imgs[i], 0]);
            df = pd.DataFrame(combined, columns = ['id', 'status']);
            df.to_csv(self.system_dict["current_folder_output_path"] + "/status.csv", index=False);


        if(algo_type == "object_detection"):
            if(algo == "gluoncv_finetune"):
                if("ssd" in model):
                    os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_folder_img_ssd.sh");
                    cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_folder_img_ssd.sh'];
                if("rcnn" in model):
                    os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_folder_img_rcnn.sh");
                    cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_folder_img_rcnn.sh'];
                if("yolo" in model):
                    os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_folder_img_yolo.sh");
                    cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_folder_img_yolo.sh'];
                if("center_net" in model):
                    os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_folder_img_center_net.sh");
                    cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_folder_img_center_net.sh'];
            elif(algo == "efficientdet_pytorch"):
                os.system("chmod +x Monk_Object_Detection/inference_engine/efficientdet_pytorch/lib/infer_folder_img_efficientdet.sh");
                cmd = ['bash', 'Monk_Object_Detection/inference_engine/efficientdet_pytorch/lib/infer_folder_img_efficientdet.sh'];


        cmd_status = self.run_command(cmd, verbose=verbose);


        if(cmd_status):
            print("");
            print("");
            if(self.system_dict["current_visualize"]):
                output_files = os.listdir(self.system_dict["current_folder_output_path"])
                img_files = [];
                for i in range(len(output_files)):
                    ext = output_files[i].split(".")[-1];
                    if(ext not in ["txt", "xml", "csv", "json"]):
                        img_files.append(output_files[i]);

                for i in range(len(img_files)):
                    print("Img - {}".format(self.system_dict["current_folder_output_path"] + "/" + img_files[i]))
                    img=mpimg.imread(self.system_dict["current_folder_output_path"] + "/" + img_files[i])
                    plt.figure(figsize=(20,10))
                    imgplot = plt.imshow(img)
                    plt.show()


    def infer_video(self, algo_type, algo, model, verbose):
        if(not os.path.isfile(self.system_dict["current_video_output_path"] + "/split_status.txt")):
            self.split_frames(self.system_dict["current_video_input_path"],
                                self.system_dict["current_video_fps"],
                                verbose)
            f = open(self.system_dict["current_video_output_path"] + "/split_status.txt", 'w');
            f.write("1");
            f.close();
        else:
            f = open(self.system_dict["current_video_output_path"] + "/split_status.txt", 'r');
            lines = f.readlines();
            f.close();
            value = int(lines[0]);
            if(not value):
                self.split_frames(self.system_dict["current_video_input_path"],
                                self.system_dict["current_video_fps"],
                                verbose)
                f = open(self.system_dict["current_video_output_path"] + "/split_status.txt", 'w');
                f.write("1");
                f.close();
            else:
                print("Skipping splitting of frames");


        if(not os.path.isfile(self.system_dict["current_video_output_path"] + "/status.csv")):
            combined = [];
            all_imgs = sorted(os.listdir("tmp_video"));
            for i in range(len(all_imgs)):
                combined.append([all_imgs[i], 0]);
            df = pd.DataFrame(combined, columns = ['id', 'status']);
            df.to_csv(self.system_dict["current_video_output_path"] + "/status.csv", index=False);


        if(algo_type == "object_detection"):
            if(algo == "gluoncv_finetune"):
                if("ssd" in model):
                    os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_video_ssd.sh");
                    cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_video_ssd.sh'];
                if("rcnn" in model):
                    os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_video_rcnn.sh");
                    cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_video_rcnn.sh'];
                if("yolo" in model):
                    os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_video_yolo.sh");
                    cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_video_yolo.sh'];
                if("center_net" in model):
                    os.system("chmod +x Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_video_center_net.sh");
                    cmd = ['bash', 'Monk_Object_Detection/inference_engine/gluoncv_finetune/lib/infer_video_center_net.sh'];
            elif(algo == "efficientdet_pytorch"):
                os.system("chmod +x Monk_Object_Detection/inference_engine/efficientdet_pytorch/lib/infer_video_efficientdet.sh");
                cmd = ['bash', 'Monk_Object_Detection/inference_engine/efficientdet_pytorch/lib/infer_video_efficientdet.sh'];



        cmd_status = self.run_command(cmd, verbose=verbose);


        if(cmd_status):
            if(self.system_dict["current_video_merge"]):
                self.merge_frames(self.system_dict["current_video_output_path"],
                                    self.system_dict["current_video_fps"],
                                    "png",
                                    verbose);

        '''
        print("");
        print("");
        if(self.system_dict["current_visualize"]):
            output_files = os.listdir(self.system_dict["current_video_output_path"])
            img_files = [];
            for i in range(len(output_files)):
                ext = output_files[i].split(".")[-1];
                if(ext not in ["txt", "xml", "csv", "json"]):
                    img_files.append(output_files[i]);

            for i in range(len(img_files)):
                print("Img - {}".format(self.system_dict["current_folder_output_path"] + "/" + img_files[i]))
                img=mpimg.imread(self.system_dict["current_folder_output_path"] + "/" + img_files[i])
                plt.figure(figsize=(20,10))
                imgplot = plt.imshow(img)
                plt.show()
        '''




    def split_frames(self, video_path, fps, verbose):
        if(os.path.isdir("tmp_video")):
            os.system("rm -r tmp_video");

        os.system("mkdir tmp_video");
        f = open("split_frames.sh", 'w');
        wr = "echo 'Splitting Frames ....'\n"; 
        wr += "ffmpeg" + " -v verbose -i " + video_path + " -r " + str(fps) + " tmp_video/%d.png\n";
        wr += "echo 'Completed'";
        f.write(wr);
        f.close();
        
        os.system("chmod +x split_frames.sh");
        cmd = ['bash', 'split_frames.sh'];

        cmd_status = self.run_command(cmd, verbose=verbose);
        os.system("rm split_frames.sh")


    def merge_frames(self, output_folder, fps, ext, verbose):
        f = open("merge_frames.sh", 'w');
        wr = "echo 'Merging Frames ....'\n"; 
        wr += "ffmpeg" + " -r " + str(fps) + " -i " \
                + self.system_dict["current_video_output_path"] + "/%d." + ext + " -c:v libx264 -pix_fmt yuv420p " \
                + self.system_dict["current_video_output_path"] + "/" + self.system_dict["current_video_input_path"].split("/")[-1] + "\n";
        wr += "echo 'Completed'";
        f.write(wr);
        f.close();

        os.system("chmod +x merge_frames.sh");
        cmd = ['bash', 'merge_frames.sh'];
        cmd_status = self.run_command(cmd, verbose=verbose);
        os.system("rm merge_frames.sh")


    def display_monk_format(self, img_path, csv_file):
        df = pd.read_csv(csv_file);
        img_name = img_path;
        img = cv2.imread(img_name);
        
        wr = df.iloc[0]["Label"];
        wr = wr.split(" ");
        
        for i in range(len(wr)//5):
            x1 = int(wr[i*5+0])
            y1 = int(wr[i*5+1])
            x2 = int(wr[i*5+2])
            y2 = int(wr[i*5+3])
            label = wr[i*5+4];
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
            
        cv2.imwrite("output_monk.jpg", img);

        img=mpimg.imread("output_monk.jpg")
        plt.figure(figsize=(20,10))
        imgplot = plt.imshow(img)
        plt.show()


    def display_coco_format(self, img_path, json_file):
        with open(json_file) as f:
            data = json.load(f); 
        
        
        img_name = img_path;
        img = cv2.imread(img_name);
        
        for i in range(len(data["annotations"])):
            x1 = data["annotations"][i]["bbox"][0];
            y1 = data["annotations"][i]["bbox"][1];
            w = data["annotations"][i]["bbox"][2];
            h = data["annotations"][i]["bbox"][3];
            label_id = data["annotations"][i]["category_id"]
            label = data["categories"][label_id]["name"]
            x2 = x1 + w;
            y2 = y1 + h;
            
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
            
        cv2.imwrite("output_coco.jpg", img);

        img=mpimg.imread("output_coco.jpg")
        plt.figure(figsize=(20,10))
        imgplot = plt.imshow(img)
        plt.show()



    def display_voc_format(self, img_path, xml_file):
        with open(xml_file) as f:
            data = xmltodict.parse(f.read())
        
        img_name = img_path;
        img = cv2.imread(img_name);
        
        objs = data["annotation"]["object"];
        for i in range(len(objs)):
            x1 = int(objs[i]["bndbox"]["xmin"]);
            y1 = int(objs[i]["bndbox"]["ymin"]);
            x2 = int(objs[i]["bndbox"]["xmax"]);
            y2 = int(objs[i]["bndbox"]["ymax"]);
            label = objs[i]["name"]
        
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
            
        cv2.imwrite("output_voc.jpg", img);

        img=mpimg.imread("output_voc.jpg")
        plt.figure(figsize=(20,10))
        imgplot = plt.imshow(img)
        plt.show()



    def display_yolo_format(self, img_name, txt_file, class_file):
        f = open(txt_file, 'r');
        lines = f.readlines();
        f.close();
        
        f = open(class_file, 'r');
        lines2 = f.readlines();
        f.close();
        
        classes = [];
        for i in range(len(lines2)):
            classes.append(lines2[i]);
        
        img = cv2.imread(img_name);
        h, w, c = img.shape;
        

        for i in range(len(lines)):
            tmp = lines[i].split(" ");
            class_id = int(tmp[0]);
            xC = float(tmp[1])*w;
            yC = float(tmp[2])*h;
            W = float(tmp[3])*w;
            H = float(tmp[4])*h;
            x1 = int(xC - W/2);
            y1 = int(yC - H/2);
            x2 = int(xC + W/2);
            y2 = int(yC + H/2);

            
            label = classes[class_id]
        
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
            
        cv2.imwrite("output_yolo.jpg", img);

        img=mpimg.imread("output_yolo.jpg")
        plt.figure(figsize=(20,10))
        imgplot = plt.imshow(img)
        plt.show()