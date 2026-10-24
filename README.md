# TCW-OS: Living AI on Microcontrollers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23615632.svg)](https://doi.org/10.5281/zenodo.23615632)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32--C3-blue)](https://www.seeedstudio.com/XIAO-ESP32C3-p-5431.html)

**Reference implementation** for the preprint:

> **TCW-OS: A Living-AI Operating System for Microcontrollers via Genome-Phenotype Weight Decomposition**
> Preprint v1.0, October 2026 — [DOI: 10.5281/zenodo.23615632](https://doi.org/10.5281/zenodo.23615632)

---

## What is TCW-OS?

An RTOS-based runtime that manages the **lifecycle** of neural network weights on microcontrollers using **genome-phenotype decomposition**:
W = (C, Phi, b)
C = frozen core model (genome, never changes)
Phi = learnable adapter (phenotype, 25 weights = 100 bytes)
b = gating (expression regulator, [0.5, 1.5])

text

**All 8 operations preserve C and modify only Phi.**

## 8 Operations

| Op | Formula | Purpose |
|---|---|---|
| E_topo | `Phi' = Phi - eta * grad(L)` | Metabolism (SGD, C frozen) |
| R | `Phi <- backup` | Regeneration |
| A | `w_i = 0` (min abs w) | Aging |
| M | `Phi ~ Uniform[-0.5, 0.5]` | Mutation |
| P | `Phi_i = dropout(Phi)` | Proliferation |
| N | `Phi' = alpha * Phi` | Normalization |
| compose | `Phi_h = 0.5*Phi_A + 0.5*Phi_B` | Composition |
| EWC | `Phi' = Phi - eta*grad(L + lambda*F*(Phi - theta*)^2)` | Anti-forgetting |

## Key Results (measured, reproducible)

Six findings, validated on **both x86 (native_sim) and RISC-V (ESP32-C3)**:

1. **Homeostasis PROVEN**: `max|Delta y_model| = 0.00000000` after all 8 operations.
2. **Genome-phenotype PROVEN**: shared `y_model = 1.025116`, different `y_final` (0.969 / 0.668 / 0.777).
3. **Fisher init-critical**: zero-init -> 1/25 (dead); random-init -> 25/25.
4. **Multi-anchor EWC HARMFUL**: at lambda=1000, multi +62.4% vs single +26.4% (-36 pp).
5. **Non-monotonic lambda curve**: +33.7 / +48.5 / +63.7 / +26.4% — "dead zone" at lambda in [50,500].
6. **Production-ready**: 284 KB Flash (3.4%), 104 KB RAM (26%) on $2 hardware.

## Hardware

| Platform | CPU | RAM | Flash | Speed |
|---|---|---|---|---|
| native_sim | x86 @ 3.5 GHz | ~10 GB | -- | ~50 ms/inference |
| **ESP32-C3** | RISC-V @ 160 MHz | 400 KB | 4 MB | ~200-500 ms/inference |

Tested on **Seeed XIAO ESP32-C3** (~$2).

## Quick Start

### Native simulation (fastest)

```bash
west build -b native_sim -d build_sim .
./build_sim/zephyr/zephyr.exe
ESP32-C3 (real hardware)
bash
# One-time
west blobs fetch hal_espressif
pip install "esptool>=5.0.2"

# Build & flash
west build -b xiao_esp32c3 -d build_esp_c3 .
west flash -d build_esp_c3

# Monitor
picocom -b 115200 /dev/ttyACM0
Repository Structure
text
tcw_demo/
  src/
    main.c          -- 10 threads, lambda sweep, lifecycle demo
    tcw_runtime.c   -- 8 operations + MLP + Fisher
    tcw_runtime.h
    tcw_ai.cpp      -- TFLite Micro wrapper (C++)
    tcw_ai.h        -- C-API for main.c
    model.cpp       -- sine-wave flatbuffer
    model.hpp
    constants.c/h   -- kXrange, kInferencesPerCycle
    assert.cpp      -- TFLite Micro assertion handler
  CMakeLists.txt
  prj.conf          -- Kconfig for native_sim / ESP32
  README.md
Limitations (honest)
Sine-wave is synthetic -- needs SKAB / industrial validation.

One task (T1-T3), not 10+.

ESP32-C3 is single-core -- SMP unused (planned for ESP32-S3).

No OTA for Phi updates.

Citation
If you use TCW-OS in your research, please cite:

bibtex
@misc{tcw-os-2026,
  title  = {TCW-OS: A Living-AI Operating System for Microcontrollers
            via Genome-Phenotype Weight Decomposition},
  author = {Nazar Pavel},
  year   = {2026},
  month  = {oct},
  note   = {Preprint v1.0},
  doi    = {10.5281/zenodo.23615632},
  url    = {https://doi.org/10.5281/zenodo.23615632}
}
License
MIT -- see LICENSE.

"We do not create life. We create a language for describing life."
