import torch
import onnx
from .Pytorch_Wrapper import Classifier

class OnnxWrapper:
    def __init__(self, model_path=None):
        """
        Initializes the Onnx_Wrapper instance.

        Args:
            model_path (str, optional): The file path to the ONNX model. If provided, the model will be loaded during initialization.

        Attributes:
            model: Placeholder for the loaded model.
            onnx_model: Placeholder for the ONNX model.
        """
        self.model = None
        self.onnx_model = None
        if model_path:
            self.loadOnnx(model_path)

    def Torch2Onnx(self, input_size=(1, 3, 224, 224), output_onnx_path=None, pt_model_path=None):
        """
        Converts a PyTorch model to ONNX format.
        Args:
            input_size (tuple, optional): The size of the input tensor. Default is (1, 3, 224, 224).
            output_onnx_path (str, optional): The path where the ONNX model will be saved. 
                                              If None, defaults to 'assets/model.onnx'.
            pt_model_path (str, optional): The path to the saved PyTorch model weights. 
                                           Must be provided.
        Raises:
            ValueError: If pt_model_path is not provided.
        Returns:
            None
        """
        if pt_model_path is None:
            raise ValueError("pt_model_path must be provided")
            
        try:
            # Create a new Classifier instance without data paths
            classifier = Classifier(model_name='resnet18', num_classes=1)
            
            # Load the saved weights
            model = classifier.load_model(pt_model_path)
            
            if output_onnx_path is None:
                output_onnx_path = 'assets/model.onnx'
            
            # Ensure model is in eval mode
            model.eval()
            
            # First try GPU if available
            try:
                if torch.cuda.is_available():
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    device = torch.device("cuda")
                    print("Using CUDA device for conversion")
                else:
                    device = torch.device("cpu")
                    print("CUDA not available, using CPU")
                
                model = model.to(device)
                input_tensor = torch.randn(*input_size).to(device)
                
                # Export with dynamic axes for better compatibility
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
                
                torch.onnx.export(
                    model,
                    input_tensor,
                    output_onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )
                
            except RuntimeError as cuda_error:
                print(f"CUDA error encountered: {cuda_error}")
                print("Falling back to CPU conversion...")
                
                # Clear memory and try again with CPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                model = model.cpu()
                input_tensor = torch.randn(*input_size)
                
                torch.onnx.export(
                    model,
                    input_tensor,
                    output_onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    verbose=False
                )
            
            # Verify the exported model
            onnx_model = onnx.load(output_onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"Model successfully exported and verified at {output_onnx_path}")
            
        except Exception as e:
            raise Exception(f"Error during ONNX conversion: {str(e)}")

    def loadOnnx(self, model_path):
        """
        Loads an ONNX model from the specified file path.

        Args:
            model_path (str): The file path to the ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.

        Raises:
            onnx.onnx_cpp2py_export.checker.ValidationError: If the model is invalid.
            FileNotFoundError: If the specified file does not exist.
        """
        try:
            self.onnx_model = onnx.load(model_path)
            onnx.checker.check_model(self.onnx_model)
            print(f"ONNX model successfully loaded and verified from {model_path}")
            return self.onnx_model
        except Exception as e:
            raise Exception(f"Error loading ONNX model: {str(e)}")

    def getModelSummary(self):
        """
        Prints a summary of the ONNX model.

        This method checks if an ONNX model is loaded. If no model is loaded, it prints a message indicating that.
        If a model is loaded, it prints a human-readable representation of the model's graph.

        Returns:
            None
        """
        if self.onnx_model is None:
            print("No ONNX model loaded. Load an ONNX model first.")
        else:
            print(onnx.helper.printable_graph(self.onnx_model.graph))