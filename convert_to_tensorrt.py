#!/usr/bin/env python3
"""
TensorRT 10.x Converter for JetPack 6
Converts ONNX model to TensorRT engine with FP16 optimization
"""
import tensorrt as trt
import os
import sys
import argparse
from datetime import datetime

class TRTConverterJP6:
    """TensorRT 10.x converter for JetPack 6"""
    
    def __init__(self, verbose=False):
        self.logger = trt.Logger(trt.Logger.WARNING)
        if verbose:
            self.logger = trt.Logger(trt.Logger.INFO)
        
        self.builder = trt.Builder(self.logger)
        self.runtime = trt.Runtime(self.logger)
        
    def convert(self, onnx_path, engine_path, precision='fp16', 
                max_workspace=2, dynamic_batch=True):
        """
        Convert ONNX to TensorRT engine
        
        Args:
            onnx_path: Path to input ONNX file
            engine_path: Path to output TensorRT engine
            precision: 'fp16', 'fp32', or 'int8'
            max_workspace: Maximum workspace size in GB
            dynamic_batch: Enable dynamic batch sizes
        """
        print(f"🚀 TensorRT 10.x Conversion for JetPack 6")
        print(f"   ONNX Input: {onnx_path}")
        print(f"   Engine Output: {engine_path}")
        print(f"   Precision: {precision.upper()}")
        
        # Step 1: Parse ONNX model
        print("\n[1/5] Parsing ONNX model...")
        network = self.builder.create_network()
        parser = trt.OnnxParser(network, self.logger)
        
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("❌ Failed to parse ONNX file:")
                for i in range(parser.num_errors):
                    print(f"   Error {i}: {parser.get_error(i)}")
                return False
        
        print(f"   ✓ Network layers: {network.num_layers}")
        print(f"   ✓ Inputs: {network.num_inputs}")
        print(f"   ✓ Outputs: {network.num_outputs}")
        
        # Display input/output details
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(f"   Input {i}: {tensor.name}, Shape: {tensor.shape}")
        
        # Step 2: Configure builder
        print("\n[2/5] Configuring TensorRT builder...")
        config = self.builder.create_builder_config()
        
        # Set workspace size
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 
            max_workspace * (1 << 30)  # Convert GB to bytes
        )
        
        # Set precision
        if precision == 'fp16' and self.builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("   ✓ FP16 precision enabled")
        elif precision == 'int8' and self.builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("   ✓ INT8 precision enabled")
        else:
            print("   ✓ FP32 precision (default)")
        
        # Step 3: Set optimization profiles (dynamic shapes)
        print("\n[3/5] Setting optimization profiles...")
        profile = self.builder.create_optimization_profile()
        
        # Get first input tensor (assuming batch, channel, height, width format)
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        
        # Handle dynamic shapes
        if len(input_shape) == 4:  # NCHW format
            min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
            opt_shape = (1, input_shape[1], input_shape[2], input_shape[3])
            max_shape = (4, input_shape[1], input_shape[2], input_shape[3])
            
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            print(f"   ✓ Dynamic shapes: min{min_shape}, opt{opt_shape}, max{max_shape}")
        
        config.add_optimization_profile(profile)
        
        # Step 4: Build engine
        print("\n[4/5] Building TensorRT engine...")
        print("   This may take 5-15 minutes on Jetson...")
        
        start_time = datetime.now()
        serialized_engine = self.builder.build_serialized_network(network, config)
        build_time = (datetime.now() - start_time).total_seconds()
        
        if serialized_engine is None:
            print("❌ Failed to build engine")
            return False
        
        print(f"   ✓ Build completed in {build_time:.1f} seconds")
        
        # Step 5: Save engine
        print("\n[5/5] Saving TensorRT engine...")
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        
        engine_size = os.path.getsize(engine_path) / (1024 * 1024)  # MB
        print(f"   ✓ Engine saved: {engine_path}")
        print(f"   ✓ Engine size: {engine_size:.2f} MB")
        
        # Verify engine
        print("\n🔧 Verifying engine...")
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
            
        print(f"   ✓ Bindings: {engine.num_io_tensors}")
        print(f"   ✓ Layers: {engine.num_layers}")
        
        # List all bindings
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_mode = "Input" if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT else "Output"
            tensor_shape = engine.get_tensor_shape(tensor_name)
            print(f"   {tensor_mode}: {tensor_name}, Shape: {tensor_shape}")
        
        print(f"\n🎉 Conversion successful!")
        print(f"   You can now use: {engine_path} in your detect.py")
        
        return True
    
    def benchmark(self, engine_path, warmup=100, iterations=1000):
        """Benchmark the TensorRT engine"""
        print("\n📊 Benchmarking TensorRT engine...")
        
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Allocate buffers
        import pycuda.driver as cuda
        import pycuda.autoinit
        import numpy as np
        
        # Get input/output sizes
        input_name = engine.get_tensor_name(0)
        input_shape = engine.get_tensor_shape(input_name)
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Allocate GPU memory
        d_input = cuda.mem_alloc(dummy_input.nbytes)
        
        # Warmup
        print(f"   Warmup: {warmup} iterations")
        for _ in range(warmup):
            cuda.memcpy_htod(d_input, dummy_input)
            context.execute_v2([int(d_input)])
        
        # Benchmark
        import time
        times = []
        
        print(f"   Benchmark: {iterations} iterations")
        for _ in range(iterations):
            start = time.time()
            context.execute_v2([int(d_input)])
            cuda.Context.synchronize()
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time
        
        print(f"\n📈 Benchmark Results:")
        print(f"   Average inference: {avg_time:.2f} ms")
        print(f"   Standard deviation: {std_time:.2f} ms")
        print(f"   Throughput: {fps:.1f} FPS")
        print(f"   95th percentile: {np.percentile(times, 95):.2f} ms")
        
        return fps, avg_time

def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT for JetPack 6')
    parser.add_argument('--onnx', default='models/best.onnx', help='Input ONNX file')
    parser.add_argument('--engine', default='models/best.engine', help='Output engine file')
    parser.add_argument('--precision', default='fp16', choices=['fp32', 'fp16', 'int8'], 
                       help='Precision mode')
    parser.add_argument('--workspace', type=int, default=2, help='Workspace size in GB')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark after conversion')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create converter
    converter = TRTConverterJP6(verbose=args.verbose)
    
    # Convert
    success = converter.convert(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        max_workspace=args.workspace
    )
    
    # Benchmark if requested
    if success and args.benchmark:
        converter.benchmark(args.engine)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())