import os
import torch
from ultralytics import YOLO
import gc
from typing import Dict, Optional

class YOLOv11Trainer:
    def __init__(self, 
                 data_yaml_path: str,
                 model_type: str = "yolo11m.pt",
                 epochs: int = 30,
                 img_size: int = 640,
                 batch_size: int = 8):  # Reduced batch size
        self.data_yaml = data_yaml_path
        self.model_type = model_type
        self.epochs = epochs
        self.img_size = img_size
        self.batch_size = batch_size
        self.setup_device()
        
    def setup_device(self):
        """Configure GPU and memory settings"""
        if torch.cuda.is_available():
            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Set memory allocation settings
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
            torch.backends.cudnn.benchmark = True
            self.device = "cuda"
        else:
            self.device = "cpu"
            
    def train(self, output_dir: str = "runs/detect/train") -> Dict:
        """Train with memory optimization"""
        try:
            # Initialize model with memory settings
            self.model = YOLO(self.model_type)
            
            # Configure training parameters
            train_args = {
                "data": self.data_yaml,
                "epochs": self.epochs,
                "imgsz": self.img_size,
                "batch": self.batch_size,
                "device": self.device,
                "cache": False,  # Disable caching
                "workers": 1,    # Reduce worker threads
                "project": output_dir,
                "plots": True
            }
            
            return self.model.train(**train_args)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU OOM error. Try reducing batch size or image size")
                torch.cuda.empty_cache()
            raise e
            
if __name__ == "__main__":
    trainer = YOLOv11Trainer(
        data_yaml_path=r"C:\Users\shiva\Desktop\EXCEED\Dataset\Railway-Track-Yolov11\data.yaml",
        batch_size=4,  
        img_size=640
    )
    trainer.train()