import os
import argparse
import torch
from torchvision import transforms
from src.dataset import CocoDataset, Resizer, Normalizer
from src.config import COCO_CLASSES, colors
import cv2
import shutil


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
    parser.add_argument("--data_path", type=str, default="data/COCO", help="the root folder of dataset")
    parser.add_argument("--cls_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pretrained_model", type=str, default="trained_models/signatrix_efficientdet_coco.pth")
    parser.add_argument("--output", type=str, default="predictions")
    args = parser.parse_args()
    return args



def test(opt):
    model = torch.load(opt.pretrained_model).module
    model.cuda()
    dataset = CocoDataset(opt.data_path, set='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))

    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)

    for index in range(len(dataset)):
        data = dataset[index]
        scale = data['scale']
        with torch.no_grad():
            scores, labels, boxes = model(data['img'].cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
            boxes /= scale

        if boxes.shape[0] > 0:
            image_info = dataset.coco.loadImgs(dataset.image_ids[index])[0]
            path = os.path.join(dataset.root_dir, 'images', dataset.set_name, image_info['file_name'])
            output_image = cv2.imread(path)

            for box_id in range(boxes.shape[0]):
                pred_prob = float(scores[box_id])
                if pred_prob < opt.cls_threshold:
                    break
                pred_label = int(labels[box_id])
                xmin, ymin, xmax, ymax = boxes[box_id, :]
                color = colors[pred_label]
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(COCO_CLASSES[pred_label] + ' : %.2f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

                cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                cv2.putText(
                    output_image, COCO_CLASSES[pred_label] + ' : %.2f' % pred_prob,
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)

            cv2.imwrite("{}/{}_prediction.jpg".format(opt.output, image_info["file_name"][:-4]), output_image)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
