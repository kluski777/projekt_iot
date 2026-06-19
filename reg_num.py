from ultralytics import YOLO


model = YOLO("./train_run12/weights/best.pt")
print(model.names)


results = model("/bmw_5_series_EL6A519.jpg", conf=0.2)


for r in results:
    r.show()
    r.save("out_test.jpg")
    print(r.boxes.xyxy)
    print(r.boxes.conf)
    print(r.boxes.cls)


