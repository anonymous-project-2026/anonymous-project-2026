# APK Similarity Detection - Supplementary Materials

This repository contains the supplementary materials and code implementation for our research on Android APK similarity detection.

## 📁 Repository Structure

```
.
├── Feature/              # Feature extraction modules
│   ├── icon_extractor.py        # Icon content and style feature extraction
│   ├── omm_extractor.py         # Opcode & Method Move transition extraction
│   ├── so_extractor.py          # Native library (SO) feature extraction
│   ├── so_preprocessor.py       # SO file preprocessing utilities
│   ├── sfcg_extractor.py        # Semantic Function Call Graph extraction
│   ├── sfcg_graph_builder.py    # SFCG construction utilities
│   ├── sfcg_enhance.py          # API embedding enhancement
│   ├── feature_config.py        # Configuration for TPL filtering
│   ├── generate_config.py       # Auto-generate filter configurations
│   ├── main.py                  # Main feature extraction pipeline
│   └── res/                     # Resource files (opcodes, embeddings, etc.)
│
├── traintest/           # Training and testing modules
│   ├── icon_detection.py        # CNN-based icon similarity detection
│   ├── omm_detection.py         # OMM transition-based detection
│   ├── so_detection.py          # Native code similarity detection
│   ├── so_trainer.py            # SO feature model training
│   ├── so_tester.py             # SO feature model testing
│   ├── sfcg_detection.py        # SFCG-based detection
│   ├── sfcg_ot_utils.py         # Optimal Transport utilities for graphs
│   ├── multi_feature_main.py    # Multi-feature fusion pipeline
│   ├── optimize_thresholds.py   # Threshold optimization
│   ├── feature_cnn_models.py    # CNN model architectures
│   └── androzoo_detection.py    # AndroZoo dataset detection
│
├── images/              # Figures and diagrams
│   ├── apk.png                  # APK structure diagram
│   └── model_vs_ot_scatter.svg  # OT vs GNN comparison
│
├── Appendix.md          # Detailed implementation documentation
├── .gitignore           # Git ignore configuration
└── README.md            # This file
```

## 🚀 Quick Start

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

## 📊 Features

Our framework extracts and analyzes multiple types of features:

### 1. **Visual Features**
- **Icon Content**: Deep CNN embeddings capturing visual content
- **Icon Style**: Intermediate layer features for style similarity

### 2. **Code Features**
- **OMM (Opcode & Method Move)**: Dalvik bytecode transition patterns
- **Native Code (SO)**: ARM/x86 instruction transition matrices
- **SFCG (Semantic Function Call Graph)**: API call graph with semantic embeddings

### 3. **Similarity Metrics**
- Cosine similarity for embeddings
- Optimal Transport distance for graphs
- Euclidean distance for statistical features

## 📖 Documentation

For detailed implementation information, please refer to [Appendix.md](Appendix.md), which includes:

- Feature extraction methodology
- TPL (Third-Party Library) filtering strategy
- API embedding construction
- Baseline implementation details
- Code obfuscation handling

## 🔬 Experimental Results

Our approach achieves:
- High accuracy in detecting repackaged applications
- Robustness against code obfuscation techniques
- Scalability to large-scale APK datasets

Detailed experimental results and comparisons with baseline methods are available in the paper.

## 📝 Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex
@inproceedings{anonymous2026apk,
  title={APK Similarity Detection via Multi-Feature Fusion},
  author={Anonymous Authors},
  booktitle={Proceedings of [Conference Name]},
  year={2026}
}
```

## 🔒 License

This project is released under the MIT License for research purposes only.

## 📧 Contact

For questions or issues, please open an issue in this repository.

---

**Note**: This repository is maintained for double-blind review purposes. Author information will be disclosed upon acceptance.
