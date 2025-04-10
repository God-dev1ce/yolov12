import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('run/voc/yolov12/weights/best.pt') # 选择训练好的权重路径
    model.val(        
        data='ultralytics/cfg/datasets/VOC.yaml', #加载数据集，路径是数据集的yaml文件位置
              split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=1,
              # iou=0.7,
              # rect=False,
              save_json=True, 
              project='run_val/voc',
              name='yolov12',
              )