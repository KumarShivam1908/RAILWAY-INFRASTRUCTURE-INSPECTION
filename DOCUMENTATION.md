# Railway Infrastructure Inspection Documentation

## 🧾 Project Overview

This project aims to improve railway infrastructure inspection by using deep learning models to detect defects in tracks, bridges, and obstacles on the tracks. The project leverages various state-of-the-art models like DINO v2 and YOLOv11, and provides a framework for training, converting, and profiling these models for optimal performance. The included `doc-keeper.py` script automates documentation generation using the Gemini API.

## ⚙️ Setup & Installation Instructions

1. **Clone the repository:** `git clone <repository_url>`
2. **Create a virtual environment:** `python -m venv .venv`
3. **Activate the virtual environment:**
    - Windows: `.venv\\Scripts\\activate`
    - Linux/macOS: `source .venv/bin/activate`
4. **Install dependencies:** `pip install -r requirements.txt`
5. **Set up Gemini API Key:**  Create a `.env` file in the root directory and add your Gemini API key: `GEMINI_API_KEY=<your_key>`.  Alternatively, set the `GEMINI_API_KEY` environment variable directly in your system.

### 🧩 Explanation of Key Modules, Classes, and Functions

#### **Profiling Module:**

- **`ModelProfiler` Class:**
    - `__init__(self, test_data_path: str, batch_size: int = 1)`: Initializes the profiler with the test data path and batch size. Sets up data transformations and the data loader.
    - `setup_data(self)`: Sets up the data transformations (resize, to tensor, normalize) and creates the data loader.
    - `profile_pytorch_model(self, model_path: str) -> dict`: Profiles a PyTorch model given its path. Loads the model, runs inference on the test data, and returns metrics like average latency and accuracy.
    - `profile_onnx_model(self, model_path: str) -> dict`: Profiles an ONNX model. Loads the model using ONNX Runtime, runs inference, and returns performance metrics.
    - `profile_tensorrt_model(self, engine_path: str) -> dict`: Profiles a TensorRT engine. Loads the engine, performs inference, and returns metrics.
    - `run_complete_profile(self, pytorch_path: str = None, onnx_path: str = None, tensorrt_path: str = None) -> pd.DataFrame`: Runs a complete profile of PyTorch, ONNX, and TensorRT models if their paths are provided and returns the results in a Pandas DataFrame.


- **`Classifier` Class (Pytorch_Wrapper.py):**
    - `__init__(self, model_name: str, num_classes: int = 1, batch_size: int = 32, lr: float = 0.0001, num_epochs: int = 10, train_data_path: str = None, test_data_path: str = None)`: Initializes the classifier. Loads a pre-trained model from `timm`, freezes all layers except the classifier, and prepares data loaders if data paths are provided.
    - `train(self)`: Trains the model using the provided training data.
    - `save_model(self, path: str)`: Saves the trained model to the given path.
    - `load_model(self, path: str)` : Loads a saved model from the given path.
    - `_prepare_dataloaders(self)`: Prepares data loaders for training and testing data using torchvision transforms.

- **`OnnxWrapper` Class (Onnx_Wrapper.py):**
    - `__init__(self, model_path: str = None)`: Initializes the ONNX wrapper. Can optionally load a model.
    - `Torch2Onnx(self, input_size: tuple = (1, 3, 224, 224), output_onnx_path: str = None, pt_model_path: str = None)`: Converts a PyTorch model to ONNX format and saves it. Uses dynamic axes for batch size.
    - `loadOnnx(self, model_path: str)`: Loads an ONNX model from a given path.


- **`TensorRTWrapper` Class (TensorRT_Wrapper.py):**
    - `__init__(self, model_path: str = None, engine_path: str = None, quantize: str = "fp16", workspace_size: int = 1 << 30, log_file: str = "tensorrt.log")`: Initializes the TensorRT wrapper. Builds or loads a TensorRT engine.
    - `build_engine(self)`: Builds a TensorRT engine from an ONNX model and returns the serialized engine. Configures quantization based on `self.quantize`.
    - `load_engine(self, engine_path: str)`: Loads a serialized TensorRT engine from a file.
    - `infer(self, image_list: list)`: Performs inference using TensorRT. Takes a list of PIL images as input, performs pre-processing, inference, and returns predictions and latency.



#### **Track Defects Module (ModelWrappers/TrackDefects):**

- **`YOLOv11Trainer` Class (yolov11.py):**
    - `__init__(self, data_yaml_path: str, model_type: str = "yolo11m.pt", epochs: int = 30, img_size: int = 640, batch_size: int = 8)`: Initializes the YOLOv11 trainer with dataset path, model type, training parameters, and sets up device.
    - `setup_device(self)`: Configures the device for GPU acceleration if available.
    - `train(self, output_dir: str = "runs/detect/train") -> dict`: Trains a YOLOv11 model. Handles GPU out-of-memory errors.
- **`YOLODatasetReporter` Class (DataAnalytics.py):**
    - `__init__(self, dataset_path: str, output_dir: str = None)`: Analyzes YOLOv11 datasets and generates reports with visualizations.


#### **doc-keeper.py:**

- `read_repo_files(base_path: str) -> dict`: Reads all relevant files in the repository and returns a dictionary mapping file paths to content. Skips specified files and directories.
- `generate_documentation(repo_files: dict) -> str`: Generates Markdown documentation using the Gemini API, based on the provided codebase files.
- `write_documentation(doc_text: str, output_file: str = "DOCUMENTATION.md")`: Writes generated documentation to a file.


## 🗂 Folder & File Structure with Descriptions

```
.
├── .github
│   └── workflows
│       └── main.yml    # GitHub Actions workflow for automated documentation generation
├── ModelWrappers
│   ├── BridgeDefects   # Models and scripts for bridge defect detection
│   │   ├── Dinov2      # DinoV2 related notebooks and assets
│   │   └── YOLOV11    # YOLOv11 related files
│   ├── DistanceTracking # Files for distance tracking
│   └── TrackDefects    # Models and scripts for track defect detection
├── Profiling         # Module containing code for model profiling and wrappers
│   ├── Pytorch_Wrapper.py
│   ├── Onnx_Wrapper.py
│   ├── TensorRT_Wrapper.py
│   └── profiler.py
├── CODE_OF_CONDUCT.md    # Contributor code of conduct
├── LICENSE                 # Project license (GNU GPLv3)
├── README.md              # Project overview and details
├── doc-keeper.py          # Script for automated documentation generation
├── main.py                # Main application script
├── requirements.txt        # Project dependencies
└── .gitignore             # Files and directories to ignore for Git
```

## 🔧 How to Use

The primary usage is for profiling different model implementations (PyTorch, ONNX, TensorRT):

```python
import asyncio
from Profiling.profiler import ModelProfiler

async def main():
    profiler = ModelProfiler(
        test_data_path=r"<path_to_test_data>",  # Update with the correct path
        batch_size=1
    )
    
    results_df = profiler.run_complete_profile(
        pytorch_path=r"<path_to_pytorch_model>",  # Optional: path to .pth file
        onnx_path=r"<path_to_onnx_model>",      # Optional: path to .onnx file
        tensorrt_path=r"<path_to_tensorrt_engine>" # Optional: path to .trt file
    )
    print(results_df)
    results_df.to_csv('model_profiling_results.csv', index=False)

if __name__ == '__main__':
    asyncio.run(main())

```


## 🤝 Contribution Guidelines

This project uses the Contributor Covenant Code of Conduct (see `CODE_OF_CONDUCT.md`). Contributions are welcome via pull requests. Please ensure your code is well-documented and tested.

## 🧪 Testing & Debugging Instructions
The project currently contains Jupyter Notebooks for model training and evaluation.  Rigorous testing would involve implementing unit tests for the python scripts in the 'Profiling' directory, focusing on edge cases and data validation.  Debugging should start by enabling logging using the logzero library as demonstrated in `TensorRT_Wrapper.py`.

##  Generated Docstrings (Within code comments above).
