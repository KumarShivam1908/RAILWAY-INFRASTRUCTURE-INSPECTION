import os
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from logzero import logger, logfile
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
class TensorRTWrapper:
    """
    Wrapper class for TensorRT model inference.
    Handles model loading, engine creation, and inference.
    """
    def __init__(self, model_path=None, engine_path=None, quantize="fp16", 
                 workspace_size=1 << 30, log_file="tensorrt.log"):
        """
        Initialize TensorRT wrapper with model configuration.
        
        Args:
            model_path (str, optional): Path to the ONNX model file
            engine_path (str, optional): Path to the TensorRT engine file
            quantize (str): Quantization type ("fp16" or "fp32")
            workspace_size (int): GPU workspace size in bytes
            log_file (str): Path to log file
        """
        # Setup logging
        logfile(log_file)
        logger.info("Initializing TensorRT Wrapper")
        
        # Initialize attributes
        self.model_path = model_path
        self.engine_path = engine_path
        self.quantize = quantize
        self.workspace_size = workspace_size
        
        # TensorRT initialization
        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        self.EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        
        # Engine and context initialization
        self.engine = None
        self.context = None
        
        # Initialize engine based on provided paths
        if self.engine_path:
            self.load_engine(self.engine_path)
        elif self.model_path:
            self.build_engine()

    def build_engine(self):
        """
        Build TensorRT engine from ONNX model.
        
        Returns:
            bytes: Serialized engine if successful, None otherwise
        """
        logger.info("Building TensorRT engine from ONNX")
        try:
            with trt.Builder(self.TRT_LOGGER) as builder, \
                 builder.create_network(self.EXPLICIT_BATCH) as network, \
                 builder.create_builder_config() as config, \
                 trt.OnnxParser(network, self.TRT_LOGGER) as parser:

                # Configure builder
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)
                if self.quantize == "fp16":
                    if builder.platform_has_fast_fp16:
                        config.set_flag(trt.BuilderFlag.FP16)
                        logger.info("Enabled FP16 precision")
                    else:
                        logger.warning("Platform doesn't support fast FP16, using FP32")

                # Additional configurations
                config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
                config.default_device_type = trt.DeviceType.GPU

                # Parse ONNX model
                if not self._parse_model(parser):
                    return None

                # Create optimization profile
                profile = self._create_optimization_profile(builder, config)

                # Build engine
                logger.info("Building serialized engine")
                self.engine = builder.build_serialized_network(network, config)
                
                if self.engine:
                    logger.info("Engine built successfully")
                    self._save_engine("assets/model.trt")
                else:
                    logger.error("Failed to build engine")
                
                return self.engine

        except Exception as e:
            logger.error(f"Error building engine: {str(e)}")
            return None

    def load_engine(self, engine_path):
        """
        Load serialized TensorRT engine from file.
        
        Args:
            engine_path (str): Path to serialized engine file
        """
        logger.info(f"Loading TensorRT engine from {engine_path}")
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(self.TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine:
                logger.info("Engine loaded successfully")
                self.context = self.engine.create_execution_context()
            else:
                logger.error("Failed to load engine")
                
        except Exception as e:
            logger.error(f"Error loading engine: {str(e)}")
    def _parse_model(self, parser):
        """
        Parse ONNX model file.
        
        Args:
            parser: TensorRT ONNX parser
            
        Returns:
            bool: True if parsing succeeds, False otherwise
        """
        try:
            with open(self.model_path, "rb") as f:
                model_data = f.read()
                if not parser.parse(model_data):
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return False
            logger.info("ONNX model parsed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error parsing model: {str(e)}")
            return False

    def _create_optimization_profile(self, builder, config):
        """
        Create optimization profile for dynamic shapes.
        
        Args:
            builder: TensorRT builder
            config: TensorRT builder configuration
            
        Returns:
            IOptimizationProfile: Created optimization profile
        """
        try:
            profile = builder.create_optimization_profile()
            input_shape = (1, 3, 224, 224)  # Adjust based on model requirements
            profile.set_shape("input", input_shape, input_shape, input_shape)
            config.add_optimization_profile(profile)
            logger.info("Optimization profile created successfully")
            return profile
            
        except Exception as e:
            logger.error(f"Error creating optimization profile: {str(e)}")
            return None

    def _save_engine(self, file_path):
        """
        Save serialized engine to file.
        
        Args:
            file_path (str): Path to save the engine
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(self.engine)
            logger.info(f"Engine saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving engine: {str(e)}")

