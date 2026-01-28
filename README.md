# APK Similarity Detection - Supplementary Materials

This repository contains the supplementary materials and code implementation for research on Android APK similarity detection.

## Project Structure

- **Feature/**: Feature extraction modules including icon, Smali opcode, native library (SO), and API call graph extraction
- **traintest/**: Training and testing modules for similarity detection, including CNN models, optimal transport utilities, and multi-feature fusion
- **images/**: Figures and diagrams

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages: `numpy`, `scikit-learn`, `torch`, `networkx`, `POT` (Python Optimal Transport)
- APK analysis tools: `apktool`, `dex2jar`

### Installation

```bash
# Clone the repository
git clone https://github.com/anonymous-project-2026/anonymous-project-2026.git
cd anonymous-project-2026

# Install dependencies
pip install -r requirements.txt
```

### Feature Extraction

```bash
# Extract features from APK files
python Feature/main.py --input /path/to/apk --output features.json
```

### Similarity Detection

```bash
# Run multi-feature detection
python traintest/multi_feature_main.py --apk1 app1.apk --apk2 app2.apk
```

## Features

The framework extracts and analyzes multiple types of features:

### 1. Visual Features
- Icon Content: Deep CNN embeddings capturing visual content
- Icon Style: Intermediate layer features for style similarity

### 2. Code Features
- Smali Opcode: Dalvik bytecode patterns and transition matrices
- Native Code (SO): ARM/x86 instruction transition matrices
- API Call Graph: Semantic function call graph with API embeddings

### 3. Similarity Metrics
- Cosine similarity for embeddings
- Optimal Transport distance for graphs
- Euclidean distance for statistical features

## Documentation

Detailed implementation information can be found in [Appendix.md](Appendix.md).

## Experimental Results

The approach achieves:
- High accuracy in detecting repackaged applications
- Robustness against code obfuscation techniques
- Scalability to large-scale APK datasets

Detailed experimental results and comparisons with baseline methods are available in the paper.
