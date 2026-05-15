import os
import time
import yaml
import torch
import numpy as np
from tqdm import tqdm
from main_base import Processor
from parser import get_parser

def benchmark_fps():
    # 1. Get Arguments and Config
    parser = get_parser()
    p = parser.parse_args()
    
    # load arg form config file
    if p.config is not None:
        with open(p.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)
        parser.set_defaults(**default_arg)

    args = parser.parse_args()
    
    # We must have a config to know the model architecture
    if not args.config:
        print("Error: Please provide a config file using --config")
        return

    # Force test phase to avoid training-specific logic
    args.phase = 'test'

    # Ensure train_feeder_args has window_size to avoid KeyError in Processor.__init__
    # The Processor class uses this to format the work_dir path even in test phase.
    if 'window_size' not in args.train_feeder_args:
        ws = args.test_feeder_args.get('window_size', 150)
        args.train_feeder_args['window_size'] = ws
    
    # 2. Initialize Processor and Model
    print(f"Loading model from config: {args.config}")
    processor = Processor(args)
    processor.mk_dir() # Ensure work_dir exists before loading model
    processor.load_model()
    model = processor.model
    model.eval()
    device = processor.device
    print(f"Model loaded on device: {device}")

    # 3. Determine Input Shape
    # Standard GCN input: [Batch, Channels, Frames, Joints, Persons]
    # Default values based on common HA-SLR-GCN settings
    batch_size = getattr(args, 'test_batch_size', 1) 
    channels = 3
    window_size = args.test_feeder_args.get('window_size', 150)
    num_joints = 27 # Usually 27 for this project's reduced keypoints
    num_persons = 1

    # Try to refine window_size if present in model_args
    if 'window_size' in args.model_args:
        window_size = args.model_args['window_size']
    
    print(f"Input shape: [{batch_size}, {channels}, {window_size}, {num_joints}, {num_persons}]")
    
    # 4. Prepare Dummy Data
    dummy_input = torch.randn(batch_size, channels, window_size, num_joints, num_persons).to(device)

    # 5. Warm-up (Crucial for CUDA)
    print("Warming up for 20 iterations...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 6. Benchmark Loop
    num_iters = 200
    print(f"Benchmarking over {num_iters} iterations (Total samples: {num_iters * batch_size})...")
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in tqdm(range(num_iters)):
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
    end_time = time.perf_counter()
    
    # 7. Calculate Results
    total_time = end_time - start_time
    total_samples = num_iters * batch_size
    avg_latency_batch = total_time / num_iters
    avg_latency_sample = total_time / total_samples
    fps = 1.0 / avg_latency_sample
    
    print("\n" + "="*30)
    print("BENCHMARK RESULTS")
    print("="*30)
    print(f"Device:         {device}")
    print(f"Batch Size:     {batch_size}")
    print(f"Iterations:     {num_iters}")
    print(f"Total Samples:  {total_samples}")
    print(f"Total Time:     {total_time:.4f} s")
    print(f"Avg Latency/Batch:  {avg_latency_batch * 1000:.2f} ms")
    print(f"Avg Latency/Sample: {avg_latency_sample * 1000:.2f} ms")
    print(f"Throughput (FPS):   {fps:.2f}")
    print("="*30)

if __name__ == '__main__':
    benchmark_fps()
