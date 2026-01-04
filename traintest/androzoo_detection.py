import os
import sys
import argparse
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torchvision.models import vgg19, VGG19_Weights
import torchvision.transforms as transforms

# Add current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import models
try:
    from feature_cnn_models import SmaliOpcodeDetailCaptureCNN, SOOpcodeDetailCaptureCNN
except ImportError:
    print("Error: feature_cnn_models.py not found.")
    sys.exit(1)

# Import SFCG utils
try:
    from sfcg_ot_utils import SFCG_OT_ThresholdAnalyzer
except ImportError:
    print("Warning: sfcg_ot_utils.py not found. SFCG detection might fail.")

# -----------------------------------------------------------------------------
# 1. Data Structures & Path Finding
# -----------------------------------------------------------------------------

class ProjectSample:
    def __init__(self, group_id, project_path, is_original):
        self.group_id = group_id
        self.path = project_path
        self.is_original = is_original
        self.name = os.path.basename(project_path)
        
        # Feature paths
        self.so_npy = self._find_so_npy()
        self.omm_npy = self._find_omm_npy()
        self.gexf = self._find_gexf()
        self.icon = self._find_icon()
        
        # Cache for features
        self.so_vector = None
        self.omm_vector = None
        self.icon_vector = None
        # SFCG graph is loaded on demand usually, or cached if memory permits

    def _find_so_npy(self):
        p1 = os.path.join(self.path, 'transition_probabilities.npy')
        if os.path.exists(p1): return p1
        p2 = os.path.join(self.path, 'so_transition_probabilities.npy')
        if os.path.exists(p2): return p2
        return None

    def _find_omm_npy(self):
        p1 = os.path.join(self.path, 'dalvik.npy')
        if os.path.exists(p1): return p1
        p2 = os.path.join(self.path, 'omm', 'dalvik.npy')
        if os.path.exists(p2): return p2
        return None

    def _find_gexf(self):
        p = os.path.join(self.path, 'community_processed_graph.gexf')
        return p if os.path.exists(p) else None

    def _find_icon(self):
        img_dir = os.path.join(self.path, 'images')
        if not os.path.isdir(img_dir): return None
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                return os.path.join(img_dir, f)
        return None

    def has_all_features(self):
        # Depending on requirement, maybe we don't need ALL features?
        # User said "if not in, feature generation failed".
        # Let's assume we proceed with whatever is available, but record missing ones.
        return True

def scan_androzoo(root_dir, limit=None):
    """
    Scans the AndroZoo directory structure.
    Returns a dict: { group_id: {'original': sample, 'repacks': [sample, ...]} }
    """
    groups = {}
    
    # List all numbered folders
    if not os.path.exists(root_dir):
        return groups
        
    group_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]
    group_ids.sort(key=int)
    
    if limit:
        group_ids = group_ids[:limit]
    
    print(f"Scanning {len(group_ids)} groups...")
    
    for gid in tqdm(group_ids, desc="Scanning Groups"):
        group_path = os.path.join(root_dir, gid)
        orig_dir = os.path.join(group_path, 'original_apk')
        repack_dir = os.path.join(group_path, 'repack_apk')
        
        # Find original
        original_sample = None
        if os.path.isdir(orig_dir):
            # Assume only one project dir inside (or use the first one)
            # The folder inside original_apk is the decompiled project
            # It might be named same as apk
            subdirs = [os.path.join(orig_dir, d) for d in os.listdir(orig_dir) if os.path.isdir(os.path.join(orig_dir, d))]
            if subdirs:
                original_sample = ProjectSample(gid, subdirs[0], True)
        
        if not original_sample:
            continue
            
        # Find repacks
        repack_samples = []
        if os.path.isdir(repack_dir):
            subdirs = [os.path.join(repack_dir, d) for d in os.listdir(repack_dir) if os.path.isdir(os.path.join(repack_dir, d))]
            for sd in subdirs:
                repack_samples.append(ProjectSample(gid, sd, False))
        
        if repack_samples:
            groups[gid] = {
                'original': original_sample,
                'repacks': repack_samples
            }
            
    return groups

# -----------------------------------------------------------------------------
# 2. Datasets for Batch Extraction
# -----------------------------------------------------------------------------

class MatrixDataset(Dataset):
    def __init__(self, samples, feature_type='so'):
        self.samples = samples
        self.feature_type = feature_type # 'so' or 'omm'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample.so_npy if self.feature_type == 'so' else sample.omm_npy
        
        if not path:
            # Return zero tensor if missing
            if self.feature_type == 'so':
                return torch.zeros((1, 94, 94), dtype=torch.float32), idx
            else:
                return torch.zeros((1, 256, 256), dtype=torch.float32), idx # Assuming OMM size, need check
                # Check OMM input size. SmaliOpcodeDetailCaptureCNN uses (1, 32, 32) or similar?
                # The CNN def has Conv2d(1, 32, ...). 
                # Let's check feature_cnn_models.py again.
                # SmaliOpcodeDetailCaptureCNN:
                # conv1: 1->32. 
                # It doesn't strictly enforce input size in init, but FC layer does: 
                # self.projection_head = nn.Sequential(nn.Linear(512 * 8 * 8, 1024)...)
                # 512 channels, 8x8 spatial.
                # 5 poolings (stride 2). Input / 32 = 8 => Input = 256.
                # So OMM input should be 256x256.

    def load_matrix(self, path):
        try:
            mat = np.load(path, allow_pickle=True)
            t = torch.tensor(mat, dtype=torch.float32)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            return t
        except:
            return None

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample.so_npy if self.feature_type == 'so' else sample.omm_npy
        
        default_size = 94 if self.feature_type == 'so' else 256
        zeros = torch.zeros((1, default_size, default_size), dtype=torch.float32)
        
        if not path:
            return zeros, idx
            
        try:
            mat = np.load(path, allow_pickle=True)
            t = torch.tensor(mat, dtype=torch.float32)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            
            # Simple resize if needed? No, user implies standard matrix.
            # But let's safe guard OMM size.
            if self.feature_type == 'omm':
                if t.shape[1] != 256 or t.shape[2] != 256:
                    # Resize or pad? For now assume correct.
                    pass
            
            return t, idx
        except Exception:
            return zeros, idx

class ImageDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample.icon
        
        if not path:
            return torch.zeros((3, 224, 224)), idx, False
            
        try:
            image = Image.open(path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_t = self.transform(image)
            return img_t, idx, True
        except:
            return torch.zeros((3, 224, 224)), idx, False

# -----------------------------------------------------------------------------
# 3. Feature Extraction
# -----------------------------------------------------------------------------

def extract_features(samples, model, feature_type, batch_size, device):
    """
    Extract features for a list of samples using the given model.
    Updates the samples in-place with vectors.
    """
    if not samples:
        return

    model.eval()
    model.to(device)
    
    if feature_type == 'icon':
        dataset = ImageDataset(samples)
    else:
        dataset = MatrixDataset(samples, feature_type)
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Extracting {feature_type} features...")
    
    with torch.no_grad():
        for data in tqdm(loader):
            if feature_type == 'icon':
                imgs, idxs, valids = data
                imgs = imgs.to(device)
                # VGG extraction
                # Assume model is VGG feature extractor
                # We need the FC output or just features?
                # Usually penultimate layer.
                # Let's assume the passed model returns the vector.
                outputs = model(imgs) # Expecting (B, Dim)
            else:
                mats, idxs = data
                mats = mats.to(device)
                outputs = model(mats) # Expecting (B, Dim)
                
            outputs = outputs.cpu().numpy()
            
            for i, idx in enumerate(idxs):
                sample = samples[idx]
                vec = outputs[i]
                
                # Check if it was a dummy zero input (missing file)
                # For icon, we have valid flag. For matrix, we check file path existence in Sample object
                is_valid = False
                if feature_type == 'so':
                    if sample.so_npy: is_valid = True
                    sample.so_vector = vec if is_valid else None
                elif feature_type == 'omm':
                    if sample.omm_npy: is_valid = True
                    sample.omm_vector = vec if is_valid else None
                elif feature_type == 'icon':
                    if valids[i]:
                        sample.icon_vector = vec
                    else:
                        sample.icon_vector = None

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        weights = VGG19_Weights.DEFAULT
        base_model = vgg19(weights=weights)
        # Use features + avgpool + part of classifier to get 4096 vector
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(*list(base_model.classifier.children())[:-1]) # remove last fc (classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -----------------------------------------------------------------------------
# 4. Main Detection Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AndroZoo Fast Detection")
    parser.add_argument('--root', type=str, required=True, help='AndroZoo root directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--omm_model', type=str, required=True, help='Path to OMM CNN model')
    parser.add_argument('--so_model', type=str, required=True, help='Path to SO CNN model')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of groups')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sfcg_threshold', type=float, default=0.85, help='SFCG threshold for acceleration')
    parser.add_argument('--sfcg_prefilter_margin', type=float, default=0.05, help='SFCG prefilter margin (set 0 to disable)')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Scan Data
    groups = scan_androzoo(args.root, args.limit)
    if not groups:
        print("No valid groups found.")
        return

    # Flatten samples for feature extraction
    all_samples = []
    for gid in groups:
        all_samples.append(groups[gid]['original'])
        all_samples.extend(groups[gid]['repacks'])
    
    print(f"Total samples found: {len(all_samples)}")
    
    # 2. Load Models & Extract Features
    
    # SO Model
    print("Loading SO Model...")
    so_net = SOOpcodeDetailCaptureCNN()
    try:
        so_net.load_state_dict(torch.load(args.so_model, map_location=device))
    except Exception as e:
        print(f"Failed to load SO model: {e}")
        # Continue? Or Exit?
        # User implies models are required.
        sys.exit(1)
    extract_features(all_samples, so_net, 'so', args.batch_size, device)
    del so_net # Free GPU memory
    
    # OMM Model
    print("Loading OMM Model...")
    omm_net = SmaliOpcodeDetailCaptureCNN()
    try:
        omm_net.load_state_dict(torch.load(args.omm_model, map_location=device))
    except Exception as e:
        print(f"Failed to load OMM model: {e}")
        sys.exit(1)
    extract_features(all_samples, omm_net, 'omm', args.batch_size, device)
    del omm_net
    
    # Icon Model (VGG)
    print("Loading Icon Model (VGG19)...")
    icon_net = VGGFeatureExtractor()
    extract_features(all_samples, icon_net, 'icon', args.batch_size, device)
    del icon_net
    
    # 3. Construct Pairs
    print("Constructing Pairs...")
    pairs = []
    
    group_ids = list(groups.keys())
    
    for gid in group_ids:
        orig = groups[gid]['original']
        repacks = groups[gid]['repacks']
        
        # Positive Pairs: Orig vs Repacks (same group)
        for rp in repacks:
            pairs.append({
                'p1': orig,
                'p2': rp,
                'label': 1,
                'type': 'positive'
            })
            
        # Negative Pairs: Orig vs Random Orig/Repack from OTHER group
        # Strategy: For each positive pair, generate 1 negative pair?
        # Or just 1 negative per group?
        # Let's generate 1 negative for each repack to balance roughly.
        for _ in repacks:
            other_gid = random.choice(group_ids)
            while other_gid == gid and len(group_ids) > 1:
                other_gid = random.choice(group_ids)
            
            if other_gid != gid:
                # Pick original from other group
                other_sample = groups[other_gid]['original']
                pairs.append({
                    'p1': orig,
                    'p2': other_sample,
                    'label': 0,
                    'type': 'negative'
                })

    print(f"Total pairs: {len(pairs)}")
    
    # 4. Compare & Detect
    results = []
    
    # Initialize SFCG Analyzer (cached)
    sfcg_analyzer = None
    try:
        # Pass root dir just for init, it doesn't scan unless we ask
        sfcg_analyzer = SFCG_OT_ThresholdAnalyzer(args.root, device=device.type)
        
        # Preload SFCG features
        print("Preloading SFCG Graph Features (this may take time)...")
        # Filter samples that have GEXF
        valid_sfcg_samples = [s for s in all_samples if s.gexf]
        valid_sfcg_paths = [s.path for s in valid_sfcg_samples]
        
        # preload_apk_features expects list of directories containing the gexf file
        sfcg_analyzer.preload_apk_features(valid_sfcg_paths, max_nodes=2000) 
        print(f"Loaded {len(sfcg_analyzer.features_cache)} graphs.")
        
    except Exception as e:
        print(f"SFCG Init Warning: {e}")
        sfcg_analyzer = None

    for p in tqdm(pairs, desc="Detecting"):
        s1 = p['p1']
        s2 = p['p2']
        
        row = {
            'p1_path': s1.path,
            'p2_path': s2.path,
            'label': p['label'],
            'group_id': s1.group_id
        }
        
        # --- SO Similarity ---
        if s1.so_vector is not None and s2.so_vector is not None:
            sim = F.cosine_similarity(
                torch.tensor(s1.so_vector).unsqueeze(0), 
                torch.tensor(s2.so_vector).unsqueeze(0)
            ).item()
            row['so_sim'] = sim
        else:
            row['so_sim'] = -1.0 # Missing
            
        # --- OMM Similarity ---
        if s1.omm_vector is not None and s2.omm_vector is not None:
            sim = F.cosine_similarity(
                torch.tensor(s1.omm_vector).unsqueeze(0), 
                torch.tensor(s2.omm_vector).unsqueeze(0)
            ).item()
            row['omm_sim'] = sim
        else:
            row['omm_sim'] = -1.0
            
        # --- Icon Similarity ---
        if s1.icon_vector is not None and s2.icon_vector is not None:
            sim = F.cosine_similarity(
                torch.tensor(s1.icon_vector).unsqueeze(0), 
                torch.tensor(s2.icon_vector).unsqueeze(0)
            ).item()
            row['icon_sim'] = sim
        else:
            row['icon_sim'] = -1.0
            
        # --- SFCG Similarity (OT Distance) ---
        if sfcg_analyzer and s1.gexf and s2.gexf:
            try:
                # Acceleration: Two-stage prefiltering
                f1 = sfcg_analyzer.features_cache.get(s1.path)
                f2 = sfcg_analyzer.features_cache.get(s2.path)
                
                sim = None
                if args.sfcg_prefilter_margin > 0 and f1 is not None and f2 is not None:
                    approx_sim = sfcg_analyzer.quick_prefilter_similarity(f1, f2)
                    # If definitely similar or definitely dissimilar, skip OT
                    if approx_sim >= (args.sfcg_threshold + args.sfcg_prefilter_margin) or \
                       approx_sim <= (args.sfcg_threshold - args.sfcg_prefilter_margin):
                        sim = approx_sim
                
                if sim is None:
                    # Use sinkhorn method as in original script
                    sim = sfcg_analyzer.calculate_ot_similarity(s1.path, s2.path, method='sinkhorn', sinkhorn_reg=0.1)
                
                row['sfcg_sim'] = sim
            except Exception as e:
                # print(f"SFCG Err: {e}")
                row['sfcg_sim'] = -1.0
        else:
            row['sfcg_sim'] = -1.0

        results.append(row)
        
    # 5. Save Results
    df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, 'androzoo_detection_results.csv')
    df.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")

if __name__ == '__main__':
    main()
