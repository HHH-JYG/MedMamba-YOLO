import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('MedMamba-YOLO.yaml')

    model.train(data= 'Br35H.yaml',
                # cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                # single_cls=False,  # 是否是单类别检测
                # batch=18,
                # close_mosaic=10,
                # workers=0,
                # device='0',
                # optimizer='SGD', # using SGD
                # # resume='runs/train/exp21/weights/last.pt', # 如过想续训就设置last.pt的地址
                # amp=False,  # 如果出现训练损失为Nan可以关闭amp
                # project='runs/train',
                # name='exp',
                )