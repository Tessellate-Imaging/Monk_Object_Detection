import argparse
import torch
from src.config import COCO_CLASSES, colors
import cv2
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
    parser.add_argument("--cls_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pretrained_model", type=str, default="trained_models/signatrix_efficientdet_coco.pth")
    parser.add_argument("--input", type=str, default="test_videos/input.mp4")
    parser.add_argument("--output", type=str, default="test_videos/output.mp4")

    args = parser.parse_args()
    return args


def test(opt):
    model = torch.load(opt.pretrained_model).module
    if torch.cuda.is_available():
        model.cuda()

    cap = cv2.VideoCapture(opt.input)
    out = cv2.VideoWriter(opt.output,  cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        flag, image = cap.read()
        output_image = np.copy(image)
        if flag:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            break
        height, width = image.shape[:2]
        image = image.astype(np.float32) / 255
        image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        if height > width:
            scale = opt.image_size / height
            resized_height = opt.image_size
            resized_width = int(width * scale)
        else:
            scale = opt.image_size / width
            resized_height = int(height * scale)
            resized_width = opt.image_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((opt.image_size, opt.image_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        new_image = np.transpose(new_image, (2, 0, 1))
        new_image = new_image[None, :, :, :]
        new_image = torch.Tensor(new_image)
        if torch.cuda.is_available():
            new_image = new_image.cuda()
        with torch.no_grad():
            scores, labels, boxes = model(new_image)
            boxes /= scale
        if boxes.shape[0] == 0:
            continue

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
        out.write(output_image)

    cap.release()
    out.release()


if __name__ == "__main__":
    opt = get_args()
    test(opt)
