import onnxruntime as ort

# 打印可用的执行提供者
print("Available Execution Providers:", ort.get_available_providers())

# 检查是否有CUDA可用
cuda_available = 'CUDAExecutionProvider' in ort.get_available_providers()
print("CUDA Available:", cuda_available)

# 如果有CUDA，打印CUDA相关信息
if cuda_available:
    cuda_providers = [p for p in ort.get_available_providers() if 'CUDA' in p]
    print("CUDA Providers:", cuda_providers)