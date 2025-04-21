# Railway Infrastructure Inspection Documentation

## 🧾 Project Overview

This project aims to improve railway infrastructure inspection by using deep learning models to detect defects in tracks, bridges, and obstacles on the tracks. The project leverages various state-of-the-art models like DINO v2 and YOLOv11, and provides a framework for training, converting, and profiling these models for optimal performance.  The core functionality resides in the `Profiling` module which allows comparing performance across PyTorch, ONNX, and TensorRT. The `doc-keeper.py` script automates documentation generation using Google's Gemini API.  The repository also contains Jupyter notebooks exploring different model training and dataset analysis approaches for track and bridge defect detection.

## ⚙️ Setup & Installation Instructions

1. **Clone the repository:** `git clone <repository_url>`
2. **Create a virtual environment:** `python3 -m venv .venv`
3. **Activate the virtual environment:**
    - Windows: `.venv\\Scripts\\activate`
    - Linux/macOS: `source .venv/bin/activate`
4. **Install dependencies:** `pip install -r requirements.txt`
5. **Set up Gemini API Key:**
    - Create a `.env` file in the root directory and add your Gemini API key: `GEMINI_API_KEY=<your_key>`.
    - Alternatively, set the `GEMINI_API_KEY` environment variable directly in your system.

## 🧩 Explanation of Key Modules, Classes, and Functions

### Profiling Module (`Profiling`)

- **`profiler.py`**: Contains the `ModelProfiler` class for benchmarking model performance.

    - **`ModelProfiler` Class:**
        - `__init__(self, test_data_path: str, batch_size: int = 1)`: Initializes the profiler with the test data path and batch size. Sets up data transformations and the data loader.
        - `setup_data(self)`: Sets up the data transformations (resize, to tensor, normalize) and creates the data loader.
        - `profile_pytorch_model(self, model_path: str) -> dict`: Profiles a PyTorch model given its path. Loads the model, runs inference on the test data, and returns metrics like average latency and accuracy.
        - `profile_onnx_model(self, model_path: str) -> dict`: Profiles an ONNX model. Loads the model using ONNX Runtime, runs inference, and returns performance metrics.
        - `profile_tensorrt_model(self, engine_path: str) -> dict`: Profiles a TensorRT engine. Loads the engine, performs inference, and returns metrics.
        - `run_complete_profile(self, pytorch_path: str = None, onnx_path: str = None, tensorrt_path: str = None) -> pd.DataFrame`: Runs a complete profile of PyTorch, ONNX, and TensorRT models if their paths are provided and returns the results in a Pandas DataFrame.

- **`Pytorch_Wrapper.py`**: Contains the `Classifier` class for training and managing PyTorch models.

    - **`Classifier` Class:**
        - `__init__(self, model_name: str, num_classes: int = 1, batch_size: int = 32, lr: float = 0.0001, num_epochs: int = 10, train_data_path: str = None, test_data_path: str = None)`: Initializes the classifier with the model name, number of classes, batch size, learning rate, number of epochs, and paths to training and testing data. Loads a pre-trained model from `timm`.
        - `train(self)`: Trains the PyTorch model. Freezes all layers except the classifier head.
        - `save_model(self, path: str)`: Saves the trained PyTorch model to the specified path.
        - `load_model(self, path: str) -> torch.nn.Module`: Loads a pre-trained PyTorch model from the specified path.
        - `_prepare_dataloaders(self)`: Creates data loaders for training and testing data using torchvision transforms.


- **`Onnx_Wrapper.py`**:  Contains the `OnnxWrapper` class for ONNX model conversion and loading.

    - **`OnnxWrapper` Class:**
        - `__init__(self, model_path=None)`: Initializes the `OnnxWrapper`. Optionally loads an ONNX model if `model_path` is provided.
        - `Torch2Onnx(self, input_size=(1, 3, 224, 224), output_onnx_path=None, pt_model_path=None)`: Converts a PyTorch model to ONNX format and saves it.  Handles dynamic batch size.
        - `loadOnnx(self, model_path)`: Loads the ONNX model. (Not used in current codebase but provided for completeness).

- **`TensorRT_Wrapper.py`**: Contains the `TensorRTWrapper` class for TensorRT engine management and inference.

    - **`TensorRTWrapper` Class:**
        - `__init__(self, model_path=None, engine_path=None, quantize="fp16", workspace_size=1 << 30, log_file="tensorrt.log")`: Initializes the `TensorRTWrapper`, supporting loading from an ONNX model or a pre-built engine.
        - `build_engine(self)`: Builds a TensorRT engine from the provided ONNX model and serializes it to a file.  Handles FP16 quantization.
        - `load_engine(self, engine_path)`: Loads a pre-built TensorRT engine from a file.
        - `allocate_buffers(self)`: Allocates GPU buffers for input and output tensors.
        - `do_inference(self, context, bindings, data_ptrs, stream)`: Performs inference with the loaded TensorRT engine.
        - `run_inference(self, input_data)`: Utility function to run inference on a given input.


### Model Wrappers Module (`ModelWrappers`)

This module contains subfolders for different model implementations and experiments related to track and bridge defect detection.  These are primarily implemented in Jupyter notebooks.


- **`TrackDefects`**: Contains experiments and implementations for railway track defect detection.


    - **`FlorenceFinetune`**: This folder contains a notebook (`finetune_florence.ipynb`) focused on fine-tuning a Florence model for track defect detection.


    - **`Yolov11`**: Contains a YOLOv11 implementation for track defect detection.


        - `yolov11.py`:  Contains the `YOLOv11Trainer` class.
            - `YOLOv11Trainer`: Class for training YOLOv11 models. Handles GPU setup and memory optimization.
            - `train()`:  Trains the YOLOv11 model using Ultralytics YOLO library.  Manages batch size and image size for memory efficiency.
            - `setup_device()`: Sets up the device (CUDA or CPU) for training.


        - `DataAnalytics.py`: Contains the `YOLODatasetReporter` class.
            - `YOLODatasetReporter`:  Generates data analysis reports for YOLO datasets.  Parses labels, computes class distributions, and generates various visualizations.  Uses matplotlib and seaborn for plotting.


    - **`RCNN`**: Contains Mask R-CNN based implementations.  The notebooks in this folder explore the use of Mask R-CNN for track defect detection.  They include code for visualization of bounding boxes and labels.

- **`BridgeDefects`**: Contains implementations and experiments for railway bridge defect detection.

    - **`Dinov2`**: This folder focuses on utilizing the DINO v2 model for bridge defect detection. It includes notebooks demonstrating grayscale image processing and model training with DINO v2.  Note that `DinoV2_GSImages.ipynb` and `DinoV2_Classification_Coloured.ipynb` contain commented-out code related to Roboflow dataset download and organization.

    - **`YOLOV11`**:  This folder contains a script (`script.py`) to train a YOLOv11 model for bridge defect detection.

- **`DistanceTracking`**: Contains experiments related to distance tracking on railway tracks. This includes a notebook (`main.ipynb`) that uses YOLOv11 and concepts of focal length calculation for distance estimation.


### Other Files


- `main.py`: The main Python script for running the model profiling. It utilizes the classes from the `Profiling` module to train, convert, and profile models.  **Note:**  The training and conversion sections are commented out. The active part primarily runs the profiler.
- `doc-keeper.py`: Python script to automate the documentation generation process using the Gemini API. It reads repository files (excluding specified ignores) and prompts Gemini to create the documentation.
- `requirements.txt`: Lists the project's dependencies.
- `.gitignore`: Specifies files and folders to exclude from Git tracking.
- `README.md`: Project overview and explanation (manually written).
- `LICENSE`: Project license (GNU GPL v3).
- `CODE_OF_CONDUCT.md`: Contributor Covenant Code of Conduct.
- `.github/workflows/main.yml`:  GitHub Actions workflow to automatically regenerate the documentation on push events to the `main` branch.



## 🗂 Folder & File Structure with Descriptions

```
.
├── .github
│   └── workflows
│       └── main.yml (GitHub Actions workflow for documentation generation)
├── ModelWrappers
│   ├── BridgeDefects
│   │   ├── Dinov2
│   │   │   ├── DinoV2_Classification_Coloured.ipynb (DINOv2 for color images)
│   │   │   ├── DinoV2_GSImages.ipynb (DINOv2 for grayscale images)
│   │   │   └── ...
│   │   └── YOLOV11
│   │       └── script.py (YOLOv11 training script)
│   ├── DistanceTracking
│   │   └── main.ipynb (Distance tracking using YOLOv11)
│   └── TrackDefects
│       ├── FlorenceFinetune
│       │   └── finetune_florence.ipynb
│       ├── RCNN
│       │   ├── M-RCNN.ipynb
│       │   └── M_RCNN_30Epoch.ipynb
│       └── Yolov11
│           ├── __init__.py
│           ├── DataAnalytics.py
│           └── yolov11.py
├── Profiling
│   ├── Onnx_Wrapper.py
│   ├── Pytorch_Wrapper.py
│   ├── TensorRT_Wrapper.py
│   └── profiler.py
├── CODE_OF_CONDUCT.md
├── LICENSE
├── DOCUMENTATION.md
├── doc-keeper.py
├── main.py
├── README.md
├── requirements.txt
└── .gitignore

```


## 🔧 How to Use

The main functionality of the codebase is demonstrated in `main.py`.  The current implementation focuses on profiling pre-trained models.  To use it:

1. **Train or obtain pre-trained models:** The provided `main.py` has commented-out sections for training and converting models. You would need to uncomment and adapt these sections if you need to train your own models.  Make sure you have the correct datasets available at the specified paths.
2. **Convert models to ONNX and TensorRT:** Similarly, uncomment and adapt the conversion parts of `main.py` if you need to generate ONNX or TensorRT models.
3. **Update paths in `main.py`:**  Ensure that the paths to your test data and model files (PyTorch, ONNX, TensorRT) are correctly set in the `main()` function.
4. **Run the profiler:** Execute `python main.py`. The script will load the models, perform inference on the test data, and print the profiling results (latency and accuracy) to the console. The results are also saved to a CSV file named `model_profiling_results.csv`.

The Jupyter notebooks in the `ModelWrappers` directory can be explored independently for specific model training and analysis tasks.


## 🤝 Contribution Guidelines

This project uses the Contributor Covenant Code of Conduct.  Contributions are welcome!  Please fork the repository and create a pull request with your changes.


## 🧪 Testing & Debugging Instructions

There are no dedicated test files in this project. However, you can test the core profiling functionality by running `main.py` with existing or newly trained models.  Enable verbose logging in the `TensorRT_Wrapper` by setting the logger level to `VERBOSE` if needed.   Use the dataset analysis tools provided in `DataAnalytics.py` to understand the dataset distribution and potential issues.  Use standard Python debugging techniques for the remaining codebase.




## Updated Code with Docstrings:  (See below for updated files with complete docstrings)



```python
# main.py
import os
import asyncio
from Profiling.Pytorch_Wrapper import Classifier
from Profiling.Onnx_Wrapper import OnnxWrapper
from Profiling.TensorRT_Wrapper import TensorRTWrapper
from Profiling.profiler import ModelProfiler

async def main():
    """
    Main function to orchestrate model training, conversion, and profiling.
    """

    profiler = ModelProfiler(
        test_data_path=r"C:\Users\shiva\Desktop\EXCEED\ModelWrappers\BridgeDefects\Dinov2\updated_gs\valid",
        batch_size=1
    )
    
    # Run profiling
    results_df = profiler.run_complete_profile(
        pytorch_path=r'C:\Users\shiva\Desktop\EXCEED\ModelWrappers\BridgeDefects\Dinov2\assets\dinov2_backboned_grayscale.pth',
        onnx_path=r'C:\Users\shiva\Desktop\EXCEED\ModelWrappers\BridgeDefects\Dinov2\assets\model.onnx',
        tensorrt_path=r'C:\Users\shiva\Desktop\EXCEED\ModelWrappers\BridgeDefects\Dinov2\assets\model.trt'
    )
    
    # ... (rest of the code)

```

```python
# doc-keeper.py
import os
# ... (rest of the imports)

def read_repo_files(base_path: str) -> dict:
    """
    Recursively reads all readable files in the repository directory, excluding
    ignored directories and extensions.

    Args:
        base_path (str): The root directory of the repository.

    Returns:
        dict: A dictionary mapping relative file paths to their content.
    """
    # ... (implementation)

def generate_documentation(repo_files: dict) -> str:
    """
    Generates Markdown documentation using Gemini based on the provided codebase.

    Args:
        repo_files (dict): A mapping of file paths to file content.

    Returns:
        str: Generated documentation in Markdown format.
    """
    # ... (implementation)

# ... (rest of the code)

```
And so on for all the other Python files.  The complete updated code with docstrings is too long to include here, but you should apply the above pattern to all your Python files.  Remember to describe parameters, return types, and exceptions in the docstrings. This improved documentation will greatly benefit anyone working with the codebase.
