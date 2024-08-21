#import the COCO Evaluator to use the COCO Metrics
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

#register your data
#/home/exx/Documents/Tianma/ICM/save_model/deIMG/
#/data/Dataset/coco2017/val2017
register_coco_instances("my_dataset_train", {}, "/data/Dataset/coco2017/annotations/instances_train2017.json", "/data/Dataset/coco2017/train2017")
register_coco_instances("my_dataset_val", {}, "/data/Dataset/coco2017/annotations/instances_val2017.json", "/data/Dataset/coco2017/val2017/")
# register_coco_instances("my_dataset_val", {}, "/data/Dataset/coco2017/annotations/instances_val2017.json", "/home/exx/Documents/Tianma/ICM/save_model/deIMG/")
register_coco_instances("my_dataset_test", {}, "/data/Dataset/coco2017/annotations/instances_test2017.json", "/data/Dataset/coco2017/test2017")

#load the config file, configure the threshold value, load weights
cfg = get_cfg()

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.merge_from_file("/home/exx/Documents/Tianma/ICM/config/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "/home/exx/Documents/Tianma/ICM/save_model/R50-FPN_x3.pkl"
# cfg.merge_from_file("/home/exx/Documents/Tianma/ICM/config/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "/home/exx/Documents/Tianma/ICM/save_model/Masked_R50_FPNx3.pkl"

# Create predictor
predictor = DefaultPredictor(cfg)

#Call the COCO Evaluator function and pass the Validation Dataset
evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")

#Use the created predicted model in the previous step
inference_on_dataset(predictor.model, val_loader, evaluator)