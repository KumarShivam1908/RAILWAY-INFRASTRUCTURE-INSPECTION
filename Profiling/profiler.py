import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import Dinov2ForImageClassification
import onnxruntime
import tensorrt as trt
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
import pandas as pd

class ModelProfiler:
    def __init__(self, test_data_path: str, batch_size: int = 1):
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_data()
        
    def setup_data(self):
        """Setup data transformations and loader"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_dataset = datasets.ImageFolder(self.test_data_path, transform)
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False
        )
        
    def profile_pytorch_model(self, model_path):
        """Profile PyTorch DINOV2 model"""
        model = Dinov2ForImageClassification.from_pretrained(
            "facebook/dinov2-base",
            num_labels=5
        )
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        model.eval()

        print("Started profiling PyTorch model...")
        
        latencies = []
        accuracies = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                start_time = time.time()
                outputs = model(images).logits
                end_time = time.time()
                
                latencies.append((end_time - start_time) * 1000)  # ms
                preds = torch.argmax(outputs, dim=1)
                accuracies.append((preds == labels).float().mean().item())

            print(f"Accuracy is {np.mean(accuracies) * 100}%")
        
        return {
            "framework": "PyTorch",
            "avg_latency": np.mean(latencies),
            "accuracy": np.mean(accuracies) * 100
        }
        
    def profile_onnx_model(self, model_path):
        """Profile ONNX model"""
        session = onnxruntime.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        latencies = []
        accuracies = []

        print("Started profiling ONNX model...")
        
        for images, labels in self.test_loader:
            images = images.numpy()
            
            start_time = time.time()
            outputs = session.run(None, {input_name: images})
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000)
            preds = np.argmax(outputs[0], axis=1)
            accuracies.append((preds == labels.numpy()).mean())
        print(f"Accuracy is {np.mean(accuracies) * 100}%")
            
        return {
            "framework": "ONNX",
            "avg_latency": np.mean(latencies),
            "accuracy": np.mean(accuracies) * 100
        }
        
    def profile_tensorrt_model(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            
        context = engine.create_execution_context()
        
        # Get input and output binding names
        input_name = engine.get_tensor_name(0)  # First input tensor
        output_name = engine.get_tensor_name(1)  # First output tensor
        
        latencies = []
        accuracies = []
        
        for images, labels in self.test_loader:
            input_batch = images.numpy()
            
            # Allocate device memory for input and output
            d_input = cuda.mem_alloc(input_batch.nbytes)
            output = np.empty((self.batch_size, 5), dtype=np.float32)  # Adjust output shape as needed
            d_output = cuda.mem_alloc(output.nbytes)
            
            # Create stream
            stream = cuda.Stream()
            
            # Set input and output tensor addresses
            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))
            
            start_time = time.time()
            cuda.memcpy_htod_async(d_input, input_batch, stream)
            
            # Execute the model
            context.execute_async_v3(stream_handle=stream.handle)
            
            cuda.memcpy_dtoh_async(output, d_output, stream)
            stream.synchronize()
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000)
            preds = np.argmax(output, axis=1)
            accuracies.append((preds == labels.numpy()).mean())
            
            # Free memory
            d_input.free()
            d_output.free()
        
        print(f"Accuracy is {np.mean(accuracies) * 100}%")
            
        return {
            "framework": "TensorRT",
            "avg_latency": np.mean(latencies),
            "accuracy": np.mean(accuracies) * 100
        }

    def run_complete_profile(self, pytorch_path, onnx_path, tensorrt_path):
        """Run complete profiling"""
        results = []
        
        # Profile each model
        results.append(self.profile_pytorch_model(pytorch_path))
        results.append(self.profile_onnx_model(onnx_path))
        results.append(self.profile_tensorrt_model(tensorrt_path))
        
        # Create DataFrame
        df = pd.DataFrame(results)
        return df