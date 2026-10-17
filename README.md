# AirVisionGuard — Edge AI Biometric Security and Face Recognition Suite

AirVisionGuard builds practical, low-power computer vision and biometric security tools designed to run locally on everyday laptops and edge devices. We focus on real-time responsiveness, minimal CPU/battery overhead and complete on-device privacy without cloud dependencies.

Our current focus is **[AirVisionGuard Face Lock](apps/face_lock/README.md)**, a real-time Windows screen-locking application that monitors webcam presence and automatically protects a workstation when the authorized user steps away. Our foundation engine, the **Multi-Task Face Recognition & Compression Framework**, powers training, knowledge distillation, width scaling and INT8/INT4 quantization across 10+ neural backbones.

This is our progress and reporting page for SEP under **Innovation Lab @ CCDS**. It covers product development, model compression, system benchmarking and our next steps.

**Last updated: 17 October 2026**

**[Watch our latest progress update](#progress-videos)**

---

## Team

| Name | Role | Responsibilities |
|---|---|---|
| Nguyen Tuan Hung | Technical Lead | Model architecture, knowledge distillation, ONNX/TorchAO quantization |
| Ta Hau Viet Long | Team Member | Face Lock application and benchmarking infrastructure |

---

## Our Journey

- **Multi-task training foundation:** AirVisionGuard began as a distributed PyTorch face recognition framework supporting multi-dataset learning, ArcFace margin heads, Focal Loss and standard verification benchmarks.
- **Lightweight edge architectures:** Profiling on resource-constrained devices shifted our focus toward compact backbones (`MobileFaceNet`, `MobileFaceNetV3`, `FaceLiVTv2-S`) and configurable channel width multipliers (`0.75x`).
- **Knowledge distillation and quantization:** We built a teacher-student embedding distillation pipeline (`KDLoss`) and post-training quantization toolchains across ONNX Runtime (`QUInt8`) and `torchao` (`int8_dynamic`, `int8_weight_only`, `int4_weight_only`).
- **AirVisionGuard Face Lock:** Our main focus is now delivering a real-time, power-throttled Windows screen security application powered by pruned YuNet detection and quantized ONNX recognition.

---

## Current Product: AirVisionGuard Face Lock

**[Face Lock Documentation](apps/face_lock/README.md)** · **[Application Entry Point](apps/face_lock/main.py)**

AirVisionGuard Face Lock is an **always-on biometric desktop guard**, supported by multi-distance face registration, 5-point ArcFace landmark alignment and a fullscreen blurred Tkinter lock overlay (`ScreenLocker`). It continuously verifies the user at the workstation, locks the screen after consecutive absent frames and unlocks immediately when the authorized user returns.

The product combines three things:

1. **Low-power face and landmark detection:** `YuNetDetector` (default CPU backend supporting both the standard 12-output OpenCV DNN graph and a custom stage-2 pruned 8-output ONNX Runtime graph) with `MTCNNDetector` (`facenet-pytorch`) as a GPU-accelerated fallback.
2. **Multi-distance identity matching:** Guided 3-step registration (`Close up`, `Medium distance`, `Far distance`) aligned via similarity transform to a canonical 112×112 template and matched using normalized 512-D cosine similarity (default threshold `0.6`).
3. **Adaptive runtime power management:** Dynamic frame-rate throttling across operating states (`8 FPS` active, `3 FPS` locked, `2 FPS` authorized), every-3rd-frame detection reuse (`DETECT_EVERY_N = 3`), 15-frame post-authorization recognition caching (`SKIP_AFTER_AUTH = 15`) and 2-thread OpenCV/ORT execution caps.

All camera frames and 512-D biometric embeddings are processed **100% locally in volatile memory** and are never written to disk or transmitted over a network. The Tkinter lock screen intercepts `Esc`, `Alt+F4`, `Alt+Tab` and focus-loss events, while OS kernel-level `Ctrl+Alt+Del` remains governed by Windows security policies.

### Deployment Scenarios Supported

The system is designed for privacy-sensitive, shared or unattended workspace environments:

| Category | Target Environments |
|---|---|
| Enterprise workstations | Shared office desks, hot-desking terminals, engineering labs and compliance-monitored endpoints |
| Healthcare and clinical | Nurse stations, EMR/EHR terminals and consultation rooms requiring rapid walk-away locking |
| Financial and legal | Advisory desks, trading laptops and client-facing workstations handling confidential records |
| Education and campus labs | University computer clusters, research terminals and student innovation workspaces |

These are supported deployment environments for pilot evaluation, not a claim of commercial installations in every sector.

---

## Recent Engineering Progress

### Product and Framework Development

| Period | Progress |
|---|---|
| 29 July 2026 | Initialized the AirVisionGuard repository with 10+ backbone families, multi-benchmark evaluation, ONNX/TorchAO quantization tools and the Face Lock desktop application |
| 29 July 2026 | Added CPU fallback support in `main.py` and introduced a pure-Python `MXIndexedRecordIOFallback` converter in `tools/convert_data.py` to unpack MXNet `.rec` datasets on Windows without requiring `mxnet` |
| 2 August 2026 | Added `MobileFaceNet` training experiment configurations (`mbf-bs128-sz112-ep30`, `ep40`), Cosine Annealing + Linear Warmup schedulers and automated shell workflows |
| October 2026 | Implemented `MobileFaceNet` width multiplier scaling (`width: 0.75`) and teacher-student Knowledge Distillation (`KDLoss` with Cosine/MSE objectives and temperature scaling) in `models/multitask.py` |

The recorded verification scripts pass **end-to-end numerical equivalence checks** for stage-3/4 pruned YuNet outputs (`rtol=1e-4, atol=1e-5` in `test_yunet_stage34_pruned.py`), exported ONNX Runtime graphs (`opset 18` in `tools/export_onnx.py`) and `0.75x` MobileFaceNet Knowledge Distillation forward/backward passes (`scratch/verify_kd_width.py`).

The repository includes an end-to-end `--demo` mode in `main.py` and synthetic no-face frame fallback in `benchmark_model_variants.py` for rapid verification. **Full multi-dataset training and hardware power telemetry (`pynvml` / `pyRAPL`) require corresponding benchmark `.bin` datasets and GPU/Linux sensor support.**

### Optimization and Benchmark Metrics

Our model compression and runtime optimization pipeline tracks the following system metrics across the repository's shipped artifacts:

| Metric / Artifact | Result |
|---|---:|
| Exported ONNX recognition graph + weights (`pretrained_models/model.onnx` + `.data`) | 8.18 MB |
| Standard YuNet detector (`yunet_n_640_640.onnx` / `2023mar.onnx`) | 310.0 KB / 227.1 KB |
| Stage-2 pruned YuNet detector (`face_detection_yunet_stage2_dropped.onnx`, 8 outputs) | 197.3 KB (~13% smaller) |
| Face Lock adaptive frame rates (`Active` / `Locked` / `Authorized Cached`) | 8 FPS / 3 FPS / 2 FPS |
| Detection & recognition duty-cycle reduction (`DETECT_EVERY_N` / `SKIP_AFTER_AUTH`) | 1 in 3 frames / 15-frame skip |
| Standard verification benchmarks supported | 7 (`LFW`, `CFP-FF`, `CFP-FP`, `AgeDB-30`, `CALFW`, `CPLFW`, `MegaFace`) |

*Source: repository model artifacts and `apps/face_lock/benchmark_model_variants.py`, which replays shared frame buckets across four application stages (`Registering`, `Detecting`, `Authorized Cached` and `Locked`) to measure disk size, Win32 RSS RAM delta, compute time, loop latency, throughput and CPU/GPU utilization.*

---

## Deployment Tiers and Model Variants

AirVisionGuard supports five model and runtime configurations tailored to different hardware budgets:

| Variant | Runtime Backend | Precision / Scaling | Main Scope |
|---|---|---|---|
| **Edge Ultra-Light** | ONNX Runtime (`CPUExecutionProvider`) | Dynamic `QUInt8` + Pruned YuNet | Lowest RAM and CPU footprint for battery-powered Windows laptops |
| **Edge Standard** | ONNX Runtime (CPU or CUDA `IOBinding`) | FP32 ONNX (`opset 18`) | Default Face Lock recognition pipeline with 512-D embeddings |
| **Distilled Compact** | PyTorch / ONNX | `0.75x` MobileFaceNet + KD | Student model distilled from a `1.0x` teacher via Cosine/MSE `KDLoss` |
| **TorchAO Quantized** | PyTorch Native (`torchao`) | `int8_dynamic`, `int8_weight_only`, `int4_weight_only` | Native PyTorch low-bit inference (`Int4CPULayout` supported on CPU) |
| **Server / Teacher** | PyTorch DDP + `torch.amp` | FP16 / FP32 (`1.0x`+) | High-capacity training (`ResNet`, `DenseNet`, `IR-SE`, `FaceLiVTv2`, `NASNet`) |

---

## Next Phase: Edge Validation and Pilot Rollout

Our next phase is to complete cross-architecture accuracy-vs-power trade-off studies and validate Face Lock with pilot users in real workstation workflows.

| Stage | Planned Work |
|---|---|
| **Distillation & quantization ablation** | Benchmark `0.75x` KD-trained MobileFaceNet + `QUInt8`/`int4` variants across `LFW`, `CFP-FP` and `AgeDB-30` |
| **Stage-level power profiling** | Record multi-hour CPU utilization and battery discharge metrics across the 4-stage Face Lock benchmark suite |
| **Desktop UX & anti-spoofing hardening** | Add liveness/anti-spoofing heuristics, persistent encrypted profile enrollment and system-tray controls |
| **Pilot workstation trials** | Run supervised usability and false-lock rate evaluations with student and lab workstation users |

### Support from Our Incubation Centre

We are seeking **further technical and pilot validation support** for edge hardware testing, user trials in shared lab environments and packaging the application as a standalone Windows installer. Introductions to enterprise IT and campus security stakeholders would also help.

We plan to report regularly on:

- Verification accuracy retention after `0.75x` width scaling, KD and INT8/INT4 quantization
- Mean compute latency (ms) and CPU utilization (%) in the `Authorized Cached` state
- False lock rate and recovery latency across lighting and distance variations
- Active pilot workstation installations and user retention feedback

---

## Foundation Engine: Multi-Task Face Recognition Framework

[`main.py`](main.py) · [`models/multitask.py`](models/multitask.py) · [`tools/`](tools/)

Before building Face Lock, we developed the **Multi-Task Face Recognition Training Framework** in PyTorch. It trains shared feature extractors across multiple labeled datasets simultaneously using per-task classification heads (`ArcFullyConnected` or `Linear`) and configurable loss weights.

Key capabilities include:

- **10+ Backbone Families:** `mobilefacenet` (`0.75x`–`1.0x`+ width scaling), `mobilefacenetv3` (Hard-Swish + Squeeze-and-Excitation + GDC), `facelivtv2` (`s`/`m`/`l` Structural Reparameterization + Next-ViT hybrid), `efficientnet` (`b0`–`b7`), `resnet` (`18`–`152`), `ir` / `ir_se`, `densenet`, `inception`, `nasnet` and `vgg`.
- **Knowledge Distillation (`KDLoss`):** Freezes a pretrained teacher checkpoint and aligns student embeddings via Cosine or MSE loss with configurable `alpha` (task weight), `beta` (KD weight), `temperature` and automatic dimension projection.
- **Modern Distributed Training:** Supports `torchrun` Distributed Data Parallel (DDP), `torch.amp` Automatic Mixed Precision with gradient clipping, `SequentialLR` (Linear Warmup + Cosine Annealing), CPU fallback and CLI dot-notation config overrides.

### Repository Structure and Quick Commands

| Module / Path | Responsibility |
|---|---|
| [`main.py`](main.py) | Unified CLI for multi-task training, DDP, `--demo` verification, offline `--evaluate` and `--extract` |
| [`models/`](models/multitask.py) | `MultiTaskWithLoss`, `KDLoss`, `FocalLoss`, `ArcFullyConnected` and 10+ backbone implementations |
| [`datasets.py`](datasets.py) & [`evaluation/`](evaluation/verify.py) | `FileListLabeledDataset`, `BinDataset`, `GivenSizeSampler`, ROC/verification metrics and MegaFace evaluator |
| [`tools/`](tools/export_onnx.py) | MXNet record converters (`convert_data.py`), ONNX exporter (`export_onnx.py`), `torchao` quantizer (`load_quantized.py`) and profilers (`evaluate_onnx.py`, `eval_quantized.py`) |
| [`apps/face_lock/`](apps/face_lock/README.md) | Real-time Windows Face Lock application (`main.py`), `YuNet`/`MTCNN` detectors and 4-stage pipeline benchmark (`benchmark_model_variants.py`) |
| [`experiments/`](experiments/) | Ready-to-run YAML configs and shell scripts for `webface` (`mbf`, `mbfv3`, `facelivtv2_s`, `effb0`, `res50`) and `ms1m` (`densenet121`) |

#### Common Workflows

```bash
# 1. Install dependencies (root framework & Face Lock app)
pip install -r requirements.txt
pip install -r apps/face_lock/requirements.txt

# 2. Convert InsightFace / MXNet .rec datasets (includes pure-Python fallback on Windows)
python tools/convert_data.py --rec_path /path/to/mxnet_rec --output_path ./data/webface

# 3. Train a model (single GPU / CPU fallback, multi-GPU DDP, or fast --demo mode)
python main.py --config experiments/webface/mbf-bs128-sz112-ep40/config.yaml
torchrun --nproc_per_node=2 main.py --config experiments/webface/res50-bs64-sz224-ep35/config.yaml
python main.py --config experiments/webface/res50-bs64-sz224-ep35/config.yaml --demo

# 4. Train a compressed 0.75x MobileFaceNet student with Knowledge Distillation
python main.py --config experiments/webface/mbf-bs128-sz112-ep40/config.yaml \
    model.width 0.75 \
    knowledge_distillation.enabled true \
    knowledge_distillation.teacher_backbone mobilefacenet \
    knowledge_distillation.teacher_width 1.0 \
    knowledge_distillation.teacher_checkpoint pretrained_models/teacher.pth.tar \
    knowledge_distillation.alpha 1.0 \
    knowledge_distillation.beta 0.5

# 5. Export to ONNX (opset 18) with dynamic QUInt8 quantization & benchmark
python tools/export_onnx.py --config experiments/webface/mbf-bs128-sz112-ep40/config.yaml \
    --load-path pretrained_models/ckpt_epoch_40.pth.tar --output-path pretrained_models/model.onnx --quantize
python tools/evaluate_onnx.py --config experiments/webface/mbf-bs128-sz112-ep40/config.yaml \
    --pth-path pretrained_models/ckpt_epoch_40.pth.tar --onnx-path pretrained_models/model.onnx --eval-acc

# 6. Launch AirVisionGuard Face Lock & run the 4-stage pipeline benchmark
cd apps/face_lock
python main.py --model ../../pretrained_models/model.onnx --detector yunet
python benchmark_model_variants.py
```

---

## Progress Videos

| Period | Update | Video |
|---|---|---|
| May 2026 | Project progress update | [Watch](https://m.youtube.com/watch?v=PNNr2ZSZ5gs&list=PLFT8WOZ7V9uA&index=2&pp=iAQBsAgC&ra=m) |
| July 2026 | Initial framework architecture, ONNX quantization and Face Lock prototype | [Watch](https://m.youtube.com/watch?v=6AgOkSSd_nw&list=PLFT8WOZ7V9uA&index=1&pp=iAQBsAgC&ra=m) |
| September 2026 | TBD | TBD |

---

## Materials and Activities

| Material | Link |
|---|---|
| Face Lock application & benchmarking guide | [View documentation](apps/face_lock/README.md) |
| Model compression & KD verification script | [View script](scratch/verify_kd_width.py) |
| Pruned YuNet stage-3/4 equivalence test | [View script](apps/face_lock/test_yunet_stage34_pruned.py) |
| Project pitch deck | [Watch pitch deck](https://m.youtube.com/watch?v=Kpf-9zVx5Qg&list=PLFT8WOZ7V9uA&index=3&pp=iAQBsAgC&ra=m) |
| Hugging Face live demo | [Try AirVisionGuard online](https://huggingface.co/spaces/ThunGg/AirVisionGuard_HF_Space) |
| Citation metadata | [CITATION.cff](CITATION.cff) |

---

## Longer-Term Direction

Our priority is to deliver a self-contained, ultra-low-power biometric security agent for everyday consumer hardware, quantify accuracy-power-latency trade-offs across distilled and quantized backbones, and expand AirVisionGuard to support multi-factor presence and liveness verification.

The **Multi-Task Face Recognition Framework** and **AirVisionGuard Face Lock** form the core training-to-deployment pipeline of our project. The underlying training foundation builds upon *"Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition"* (ECCV 2018); see [`CITATION.cff`](CITATION.cff) for full citation details.

---

## Contact

**Team AirVisionGuard — Nguyen Tuan Hung & Ta Hau Viet Long**

[Email Tuan Hung](mailto:tuanhung001@e.ntu.edu.sg) · [GitHub Repository](https://github.com/someoneheh/AirVisionGuard) · [Face Lock Docs](apps/face_lock/README.md)

_We update this page with progress videos, benchmark milestones and deployment validation at least every two months._
