from ultralytics import YOLO

yolo = YOLO('yolov8s.yaml')
yolo.train(
    data='./Dataset_hand_label/data.yaml',
    device=0,
    imgsz=960,
    epochs=300,
    batch=8,
    lr0=0.003, warmup_epochs=3,
    mosaic=1.0, scale=0.5, copy_paste=0.7,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=2, translate=0.1, 
    cache=True,
    patience=0, 
    name='fish_tiny_aug',
    pretrained=True,
)