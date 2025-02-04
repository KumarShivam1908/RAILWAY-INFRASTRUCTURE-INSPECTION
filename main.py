import os
import asyncio
from Profiling.Pytorch_Wrapper import Classifier
from Profiling.Onnx_Wrapper import OnnxWrapper
from Profiling.TensorRT_Wrapper import TensorRTWrapper
from Profiling.profiler import ModelProfiler

async def main():
    
    '''
    # Ensure the assets directory exists
    os.makedirs('assets', exist_ok=True)

    # If you need to train the model first, uncomment these lines
    
    try:
        pt_model = Classifier(
            train_data_path=r'Dataset/melanoma_cancer_dataset/train', 
            test_data_path=r'Dataset/melanoma_cancer_dataset/test', 
            model_name='resnet18', 
            num_classes=1, 
            batch_size=32, 
            lr=0.0001, 
            num_epochs=20
        )
        pt_model.train()  
        pt_model.save_model(r'pt_model.pth')  
        print("PyTorch model trained and saved.")
    except Exception as e:
        print(f"Error during model training: {e}")
    

    # Convert to ONNX
    
    try:
        onnx_model = OnnxWrapper()
        onnx_model.Torch2Onnx(
            pt_model_path=r'assets/pt_model.pth',
            input_size=(1, 3, 224, 224),
            output_onnx_path='assets/model.onnx'
        )
        print("ONNX model successfully converted.")
    except Exception as e:
        print(f"Error during ONNX conversion: {e}")
    

    # Convert to TensorRT
    
    try:
        trt_model = TensorRTWrapper(model_path=r'assets/model.onnx', quantize="fp16")
        trt_model.build_engine()
        print("TensorRT model successfully built and saved.")
    except Exception as e:
        print(f"Error during TensorRT conversion: {e}")
    

    '''
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
    
    # Display results
    print("\nProfiling Results:")
    print(results_df.to_string(index=False))
    
    # Save results to CSV
    results_df.to_csv('model_profiling_results.csv', index=False)
    print("\nResults saved to 'model_profiling_results.csv'")


if __name__ == '__main__':
    asyncio.run(main())