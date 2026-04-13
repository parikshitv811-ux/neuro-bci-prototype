"""Benchmark inference latency"""
import sys, time
sys.path.insert(0, '/home/claude/bci_prototype')
import numpy as np, torch
from bci_core import EEGSimulator, Preprocessor, EEGNet

np.random.seed(1); torch.manual_seed(1)
sim  = EEGSimulator()
prep = Preprocessor()
model = EEGNet().eval()

# warm-up
for _ in range(5):
    ep = prep.process(sim.generate_epoch(0))
    t  = torch.tensor(ep[None, None]).float()
    _  = model(t)

latencies = []
for _ in range(200):
    ep = sim.generate_epoch(np.random.randint(0, 5))
    t0 = time.perf_counter()
    ep = prep.process(ep)
    t  = torch.tensor(ep[None, None]).float()
    _  = model(t)
    latencies.append((time.perf_counter() - t0) * 1000)

arr = np.array(latencies)
print(f"Inference latency over 200 runs:")
print(f"  mean  : {arr.mean():.2f} ms")
print(f"  median: {np.median(arr):.2f} ms")
print(f"  p95   : {np.percentile(arr, 95):.2f} ms")
print(f"  p99   : {np.percentile(arr, 99):.2f} ms")
print(f"  max   : {arr.max():.2f} ms")
