import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
     model = YOLO('ultralytics/cfg/models/v12/yolov12.yaml') #加载模型，路径是模型的yaml文件位置
     model.train(
        data='ultralytics/cfg/datasets/VOC.yaml',  #加载数据集，路径是数据集的yaml文件位置
                cache=False,
                # cos_lr=True,
                imgsz=640,
                epochs=100,
                batch=32,
                close_mosaic=0,
                workers=8, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                device='0',
                optimizer='SGD', # using SGD
                patience=100, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,例如YOLO('last.pt')
                # amp=False, # close amp
                # fraction=0.2,
                project='run/voc',
                name='yolov12',
                )