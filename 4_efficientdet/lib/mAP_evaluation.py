from src.dataset import CocoDataset, Resizer, Normalizer
from torchvision import transforms
from pycocotools.cocoeval import COCOeval
import json
import torch


def evaluate_coco(dataset, model, threshold=0.05):
    model.eval()
    with torch.no_grad():
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            scores, labels, boxes = model(data['img'].cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
            boxes /= scale

            if boxes.shape[0] > 0:

                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    if score < threshold:
                        break

                    image_result = {
                        'image_id': dataset.image_ids[index],
                        'category_id': dataset.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }

                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == '__main__':
    efficientdet = torch.load("trained_models/signatrix_efficientdet_coco.pth").module
    efficientdet.cuda()
    dataset_val = CocoDataset("data/COCO", set='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))
    evaluate_coco(dataset_val, efficientdet)
