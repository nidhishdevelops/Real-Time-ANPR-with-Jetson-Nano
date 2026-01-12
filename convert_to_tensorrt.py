# convert_trt.py - RUN THIS ON JETSON
import tensorrt as trt
import os

def convert_onnx_to_tensorrt(onnx_path, engine_path, fp16_mode=True):
    """Convert ONNX model to TensorRT engine"""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    print(f"[1/5] Loading ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"  Error: {parser.get_error(error)}")
            raise ValueError("ONNX parsing failed")
    
    print(f"[2/5] Network created: {network.num_layers} layers")
    
    # Config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # Set precision
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[3/5] Using FP16 precision")
    else:
        print("[3/5] Using FP32 precision")
    
    # Optimization profiles (for dynamic shapes)
    profile = builder.create_optimization_profile()
    profile.set_shape("images", min=(1, 3, 640, 640), 
                                 opt=(1, 3, 640, 640), 
                                 max=(1, 3, 640, 640))
    config.add_optimization_profile(profile)
    
    # Build engine
    print("[4/5] Building TensorRT engine... (This may take 10-30 minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")
    
    # Save engine
    print(f"[5/5] Saving engine to: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"✅ Conversion complete! Engine saved: {engine_path}")
    print(f"   File size: {os.path.getsize(engine_path) / 1e6:.2f} MB")

if __name__ == "__main__":
    # Convert your model
    convert_onnx_to_tensorrt(
        onnx_path="models/best.onnx",
        engine_path="models/best.engine",
        fp16_mode=True  # Use FP16 for faster inference on Jetson
    )
    
    # Test the engine
    print("\n🔧 Testing the TensorRT engine...")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open("models/best.engine", "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        print(f"Engine loaded: {engine.num_bindings} bindings")
        print(f"Max batch size: {engine.max_batch_size}")