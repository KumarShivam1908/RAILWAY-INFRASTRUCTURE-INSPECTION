from ModelWrappers.TrackDefects.Yolov11.yolov11 import YOLOv11Trainer
trainer = YOLOv11Trainer(
    data_yaml_path=r"", # Add path to data.yaml
    batch_size=4, 
    img_size=640  # Reduced image size
)

# Train and get results
training_results = trainer.train()


