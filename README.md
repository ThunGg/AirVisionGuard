# Face Recognition Framework

A powerful and flexible face recognition training framework built with PyTorch. This repository provides a streamlined pipeline for training state-of-the-art face recognition models, performing extensive benchmarking, and extracting high-dimensional facial features.

## 🚀 Key Features

- **Multi-Task Learning**: Efficiently handle multiple datasets simultaneously by treating them as separate tasks.
- **Modern Training Techniques**:
  - **Distributed Data Parallel (DDP)** for efficient multi-GPU scaling.
  - **Automatic Mixed Precision (AMP)** for faster training and reduced memory footprint.
  - **Flexible Schedulers**: Linear warmup followed by Cosine Annealing.
- **Diverse Model Support**: Supports popular backbones like ResNet, DenseNet, Inception, and more.
- **Comprehensive Benchmarking**: Built-in support for LFW, CFP-FF, CFP-FP, AgeDB-30, CALFW, CPLFW, and MegaFace.
- **Dynamic Configuration**: YAML-based configuration with easy command-line overrides.
- **Fast Prototyping**: Includes a `--demo` mode for rapid verification of your setup.

---

## 🛠 Setup Instructions

### 1. Environment Preparation

Ensure you have Python 3.8+ installed. We recommend using a virtual environment or Conda.

```bash
# Clone the repository
git clone https://github.com/someonelearn/face_rec_framework.git
cd face_rec_framework

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> This framework utilizes `torch.amp` and `torch.distributed`. Ensure your CUDA environment is correctly configured for GPU acceleration.

### 2. Data Preparation

The framework expects datasets in a specific format (images + a list file). You can convert InsightFace MXNet records using the provided tools.

#### Converting MXNet Records (e.g., CASIA-Webface)
1. Download the dataset (e.g., `faces_CASIA_112x112.zip`) and unzip it.
2. Run the conversion script:

```bash
python tools/convert_data.py --rec_path /path/to/mxnet_rec --output_path ./data/webface
```

This will create:
- `data/webface/images/`: Organized by identity.
- `data/webface/list.txt`: Mapping of image paths to labels.

#### Setting up Benchmarks
For evaluation, place `.bin` benchmark files (like `lfw.bin`) in your test root directory (e.g., `data/webface/`).

---

## 🏋️ Training

### Basic Training
Launch training using a configuration file:

```bash
python main.py --config experiments/webface/res50-bs64-sz224-ep35/config.yaml
```

### Multi-GPU Training (DDP)
For high-performance training across multiple GPUs:

```bash
torchrun --nproc_per_node=2 main.py --config experiments/webface/res50-bs64-sz224-ep35/config.yaml
```

### Fast Verification (Demo Mode)
Use the `--demo` flag to run a quick end-to-end test (limited iterations and epochs):

```bash
python main.py --config experiments/webface/res50-bs64-sz224-ep35/config.yaml --demo
```

### Command-Line Overrides
You can override any YAML configuration parameter directly from the CLI:

```bash
python main.py --config experiments/webface/res50-bs64-sz224-ep35/config.yaml \
    train.base_lr 0.1 \
    model.backbone resnet101
```

---

## 📊 Evaluation & Benchmarking

### Offline Evaluation
Evaluate a specific checkpoint against configured benchmarks:

```bash
python main.py --config experiments/webface/res50-bs64-sz224-ep35/config.yaml \
    --evaluate --load-path experiments/webface/res50-bs64-sz224-ep35/checkpoints/ckpt_epoch_35.pth
```

### Monitoring
Visualize training progress, losses, and benchmark accuracy using TensorBoard:

```bash
tensorboard --logdir experiments/webface/res50-bs64-sz224-ep35/events
```

---

## 🔍 Feature Extraction

To extract features for a custom dataset:
1. Define `extract_info` in your `config.yaml` (set `data_root` and `data_list`).
2. Run the extraction command:

```bash
python main.py --config experiments/webface/res50-bs64-sz224-ep35/config.yaml \
    --extract --load-path path/to/model.pth
```

The extracted features will be saved as a `.bin` file in the specified location.

## 📝 Citation

If you use this framework in your research, please cite the underlying work. See [CITATION.cff](CITATION.cff) for details.

---

## 🤝 Acknowledgments

This framework is built upon the research and codebase presented in "Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition" (ECCV 2018). Special thanks to Xiaohang Zhan and the community for providing the original foundations, datasets, and benchmarks.
