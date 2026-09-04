from ultralytics import YOLO

model = YOLO("yolo11n-tuesday.pt")
output = model.export(format="hailo", name="hailo8")
print(output)  # yolo11n_hailo_model/