import onnxruntime as ort
print("Available providers:", ort.get_available_providers())
print("Default provider:", ort.get_default_provider())

