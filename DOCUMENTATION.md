# Railway Infrastructure Inspection Documentation

## 🧾 Project Overview

This project aims to improve railway infrastructure inspection using deep learning models for detecting defects in tracks, bridges, and obstacles on tracks.  It leverages models like DINO v2 and YOLOv11, providing a framework for training, conversion, and profiling for optimal performance. The `Profiling` module allows performance comparisons across PyTorch, ONNX, and TensorRT.  `doc-keeper.py` automates documentation generation with Google's Gemini API. Jupyter notebooks demonstrate model training and dataset analysis for track and bridge defect detection.

## ⚙️ Setup & Installation Instructions

1. **Clone the repository:** `git clone <repository_url>`
2. **Create a virtual environment:** `python3 -m venv .venv`
3. **Activate the virtual environment:**
    - Windows: `.venv\\Scripts\\activate`
    - Linux/macOS: `source .venv/bin/activate`
4. **Install dependencies:** `pip install -r requirements.txt`
5. **Set up Gemini API Key:** Create a `.env` file in the root directory and add your Gemini API key: `GEMINI_API_KEY=<your_key>`, or set the `GEMINI_API_KEY` environment variable in your system.


## 🧩 Explanation of Key Modules, Classes, and Functions

### Profiling Module (`Profiling`)

- **`profiler.py`**: Contains the `ModelProfiler` class.

    - **`ModelProfiler` Class:**
        - `__init__(self, test_data_path: str, batch_size: int = 1)`: Initializes the profiler.
            - `test_data_path`: Path to the test dataset.
            - `batch_size`: Batch size for inference.
            - Returns: None.
        - `setup_data(self)`: Sets up data transformations and the data loader.
            - Returns: None
        - `profile_pytorch_model(self, model_path: str) -> dict`: Profiles a PyTorch model.
            - `model_path`: Path to the PyTorch model.
            - Returns: Dictionary with profiling metrics.
        - `profile_onnx_model(self, model_path: str) -> dict`: Profiles an ONNX model.
            - `model_path`: Path to the ONNX model.
            - Returns: Dictionary with profiling metrics.
        - `profile_tensorrt_model(self, engine_path: str) -> dict`: Profiles a TensorRT engine.
            - `engine_path`: Path to the TensorRT engine.
            - Returns: Dictionary with profiling metrics.
        - `run_complete_profile(self, pytorch_path: str = None, onnx_path: str = None, tensorrt_path: str = None) -> pd.DataFrame`: Runs a complete profiling suite.
            - `pytorch_path`: Path to the PyTorch model.
            - `onnx_path`: Path to the ONNX model.
            - `tensorrt_path`: Path to the TensorRT engine.
            - Returns: Pandas DataFrame with results.

- **`Pytorch_Wrapper.py`**: Contains the `Classifier` class.

    - **`Classifier` Class:**
        - `__init__(self, model_name: str, num_classes: int = 1, batch_size: int = 32, lr: float = 0.0001, num_epochs: int = 10, train_data_path: str = None, test_data_path: str = None)`: Initializes the classifier.
            - `model_name`: Name of the pretrained model.
            - `num_classes`: Number of output classes.
            - `batch_size`: Batch size.
            - `lr`: Learning rate.
            - `num_epochs`: Number of training epochs.
            - `train_data_path`: Path to the training data.
            - `test_data_path`: Path to the test data.
            - Returns: None

        - `train(self)`: Trains the model.
            - Returns: None

        - `save_model(self, model_path)`: Saves the trained model.
            - `model_path`: The path where the model is saved.
            - Returns: None
        - `load_model(self, model_path)`: Loads a saved model from given path.
            - `model_path`: The path from where the model is loaded.
            - Returns: The loaded PyTorch model.
        - `_prepare_dataloaders(self)`: Prepares train and test dataloaders
            - Returns: tuple - Train dataloader, test dataloader, dataset sizes

- **`Onnx_Wrapper.py`**: Contains the `OnnxWrapper` class.

    - **`OnnxWrapper` Class:**
        - `__init__(self, model_path=None)`: Initializes the ONNX wrapper.
            - `model_path`: Path to the ONNX model.
            - Returns: None.
        - `Torch2Onnx(self, input_size=(1, 3, 224, 224), output_onnx_path=None, pt_model_path=None)`: Converts a PyTorch model to ONNX.
            - `input_size`: Input tensor size.
            - `output_onnx_path`: Output ONNX file path.
            - `pt_model_path`: PyTorch model path.
            - Returns: None
        - `loadOnnx(self, onnx_file_path: str):`: Loads ONNX model into the wrapper
             - `onnx_file_path`: The file path of the ONNX model
             - Returns: None

- **`TensorRT_Wrapper.py`**: Contains the `TensorRTWrapper` class.

    - **`TensorRTWrapper` Class:**
        - `__init__(self, model_path=None, engine_path=None, quantize="fp16", workspace_size=1 << 30, log_file="tensorrt.log")`: Initialize TensorRT Wrapper.
            - `model_path`: Path to ONNX model.
            - `engine_path`: Path to TensorRT engine.
            - `quantize`: Quantization mode ("fp16" or "fp32").
            - `workspace_size`:  GPU workspace size.
            - `log_file`: Path to log file.
            - Returns: None
        - `build_engine(self)`: Builds TensorRT engine from ONNX model.
            - Returns: Serialized engine or None.
        - `load_engine(self, engine_path: str)` : Load a saved TensorRT engine.
            - `engine_path`: File path of the engine.
            - Returns: None
        - `infer(self, image: Image.Image) -> np.ndarray`: Performs inference with TensorRT engine.
            - `image`: PIL Image to run inference on.
            - Returns: Output of the model.


### doc-keeper.py

- `read_repo_files(base_path: str) -> dict`: Reads all files in the repository, excluding specified files and directories.
    - `base_path`: The root directory of the repository.
    - Returns: A dictionary mapping file paths to their content.


- `generate_documentation(repo_files: dict) -> str`: Generates documentation using the Gemini API.
    - `repo_files`: Dictionary of file paths and content.
    - Returns: Markdown documentation.

- `write_documentation(doc_text: str, output_file: str = "DOCUMENTATION.md")`: Writes the documentation to a file.
    - `doc_text`: The documentation text.
    - `output_file`: The output file path.
    - Returns: None


## 🗂 Folder & File Structure with Descriptions

- `.github/workflows`: Contains GitHub Actions workflows for automated tasks.
    - `main.yml`: Workflow for automatically generating and updating documentation on push to the main branch.
- `Profiling`: Contains modules for model profiling and wrappers for different frameworks.
- `ModelWrappers`: Contains implementations and notebooks for various model architectures used in the project, categorized by the type of defects (track defects, bridge defects etc).
- `Dataset`:  This directory is intended to store the various datasets utilized in the project, such as bridge and track images.
- `assets`: Contains supporting files like images and pre-trained model files.

## 🔧 How to Use

The main script (`main.py`) demonstrates the core functionality of the profiling module:

1. Ensure you have the necessary model files (PyTorch, ONNX, TensorRT) in their respective paths.  The paths in the commented code sections may need to be adjusted to point to your models.  You can uncomment the model training and conversion sections to re-generate these files, but be sure to provide the appropriate dataset paths and adjust the training parameters if needed.
2.  Ensure you have a directory of images for testing.  The provided path also needs to be adjusted if needed.
3. Run `main.py`. It will profile the provided models and save the results in `model_profiling_results.csv`.

## 🤝 Contribution Guidelines

This project uses the Contributor Covenant Code of Conduct (see `CODE_OF_CONDUCT.md`).  Contributions are welcome via pull requests.


## 🧪 Testing & Debugging Instructions

While there are no explicit test files, you can use the Jupyter notebooks (`*.ipynb`) in the `ModelWrappers` directory to experiment with different models and datasets. They provide examples of model training, data loading, and visualization which can help you debug and understand the project's components. The `DataAnalytics.py` script in `ModelWrappers/TrackDefects/Yolov11` can also be used to understand YOLOv11 datasets.  The `main.py` file acts as an integration test by bringing together different components of the profiling workflow.

##  Additional Notes

- The `README.md` file provides a general overview of the project, including the problem statement, approach, and some visuals, but is not exhaustive in terms of technical details.
- The provided LICENSE is the GNU GENERAL PUBLIC LICENSE Version 3.




