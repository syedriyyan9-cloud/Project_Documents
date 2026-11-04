from ultralytics import YOLO

# model = YOLO('yolo11n.pt')   # nano model, ~5MB download

# results = model.train(
#     data='D:/FYP_Documents/Project/Modals/Yolo/yolo_test/data.yaml',
#     epochs=3,                 # just 3 — this is a pipeline test
#     imgsz=320,                # smaller image size = faster
#     batch=4,
#     name='local_test'
# )

model = YOLO('D:/FYP_Documents/runs/detect/local_test-2/weights/best.pt')
results = model.predict('D:/FYP_Documents/Project/Modals/Yolo/yolo_test/images/val/C33P1thinF_IMG_20150619_120645a_cell_216.png', save=True)