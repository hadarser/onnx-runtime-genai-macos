# ONNX Runtime GenAI with Apple Silicon Optimization

This repository is dedicated to running ONNX Runtime with GenAI using Apple Silicon hardware optimization. It aims to provide a learning platform for working with ONNX Runtime.

### What is ONNX Runtime?
ONNX Runtime is a high-performance inference engine designed to execute machine learning models in the ONNX (Open Neural Network Exchange) format. It provides a flexible and efficient way to run models on various hardware accelerators.

### What is ONNX Runtime GenAI?
ONNX Runtime GenAI extends ONNX Runtime to support Generative AI workloads. It simplifies running large language models and other generative models by leveraging ONNX Runtime’s optimized infrastructure.

### Purpose

The purpose of this repository is to help users learn how to work with ONNX Runtime, specifically optimized for Apple Silicon hardware. It provides examples and instructions to get started with ONNX Runtime and ONNX Runtime GenAI.

### Resources

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [ONNX Runtime GitHub Repository](https://github.com/microsoft/onnxruntime)
- [ONNX Runtime GenAI GitHub Repository](https://github.com/microsoft/onnxruntime-genai)
- [Vadim Bakhrenkov's blog post on how to run LLMs with ONNX Runtime GenAI](https://medium.com/@vadikus/running-phi-3-mistral-7b-llms-on-raspberry-pi-5-a-step-by-step-guide-185e8102e35b)

# Installation 

### Can I use Docker?

Currently (as of January 2025), using a Docker image on a Mac does not provide access to the underlying Apple Silicon GPU (MPS). This means that only the CPU version of ONNX Runtime will work within a Docker container, which will be significantly slower compared to utilizing the GPU. For optimal performance, it is recommended to run ONNX Runtime natively on your Mac to take full advantage of the Apple Silicon hardware.

### Can I just use `pip install onnxruntime-genai`?

Yes, `onnxruntime-genai` supports macOS, but it must be built from source, as stated in the repository's [README](https://github.com/microsoft/onnxruntime-genai).

## Clone this repository and create a virtual environment
```bash
git clone https://github.com/hadarser/onnx-runtime-genai-macos.git
cd onnx-runtime-genai-macos
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## Clone the ONNX Runtime GenAI repository
> **Note:** The following commands will install version 0.3.0 of ONNX Runtime GenAI. From my experience, this is the only version that works with the Apple Silicon optimization. I have tried other versions (v0.4.0, v0.5.0, v0.5.1, v0.5.2), but they worked with the CPU only. 
```bash
git clone https://github.com/microsoft/onnxruntime-genai.git
cd onnxruntime-genai
git checkout tags/v0.3.0
```

## Install the ONNX Runtime GenAI package
> **Note:** This will use `onnxruntime` version 1.20.1. Other versions can be used as well.
> **Note:** You might need to install `curl` and `cmake` if you don't have them installed. You can install them using Homebrew: `brew install curl cmake`.

```bash
pip install --pre onnxruntime-genai
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-osx-arm64-1.20.1.tgz -o onnxruntime-osx-arm64-1.20.1.tgz
tar -xvzf onnxruntime-osx-arm64-1.20.1.tgz
mv onnxruntime-osx-arm64-1.20.1 ort
python build.py --config Release
pip install build/macOS/Release/wheel/onnxruntime_genai-0.3.0-cp312-cp312-macosx_15_0_arm64.wh
```