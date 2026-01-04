import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm


class VectorDataset(Dataset):
    def __init__(self, andro_dir):
        self.pair_paths = []

        # 遍历 andro_dir 下的子文件夹
        for first_level in os.listdir(andro_dir):
            first_level_path = os.path.join(andro_dir, first_level)
            if os.path.isdir(first_level_path):
                # 初始化 original_apk 的路径
                original_apk_freq_path = None

                # 遍历 original_apk 文件夹
                original_apk_path = os.path.join(first_level_path, "original_apk")
                if os.path.exists(original_apk_path) and os.path.isdir(original_apk_path):
                    for file in os.listdir(original_apk_path):
                        if file.endswith('.apk'):
                            apk_name = os.path.splitext(file)[0]
                            freq_path = os.path.join(original_apk_path, apk_name, 'data_read_frequency.npy')
                            if os.path.exists(freq_path) and not np.all(np.load(freq_path) == 0):
                                original_apk_freq_path = freq_path
                                break  # 只取一个有效的 original_apk

                # 如果找到有效的 original_apk，遍历 repack_apk 中的所有 APK
                if original_apk_freq_path:
                    repack_apk_path = os.path.join(first_level_path, "repack_apk")
                    if os.path.exists(repack_apk_path) and os.path.isdir(repack_apk_path):
                        for file in os.listdir(repack_apk_path):
                            if file.endswith('.apk'):
                                apk_name = os.path.splitext(file)[0]
                                freq_path = os.path.join(repack_apk_path, apk_name, 'data_read_frequency.npy')
                                if os.path.exists(freq_path) and not np.all(np.load(freq_path) == 0):
                                    # 将 original_apk 与每个有效的 repack_apk 配对
                                    self.pair_paths.append((original_apk_freq_path, freq_path))

        print(f"Total number of valid pairs: {len(self.pair_paths)}")

    def __len__(self):
        return len(self.pair_paths)

    def __getitem__(self, idx):
        origin_npy_path, repack_npy_path = self.pair_paths[idx]
        origin_npy = np.load(origin_npy_path)  # 形状为 (78,)
        repack_npy = np.load(repack_npy_path)  # 形状为 (78,)
        return torch.tensor(origin_npy, dtype=torch.float32), torch.tensor(repack_npy, dtype=torch.float32)

def contrastive_loss(features, temperature=0.07):
    batch_size = features.shape[0]
    labels = torch.cat([torch.arange(batch_size // 2) for _ in range(2)], dim=0).to(features.device)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(labels.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature
    return F.cross_entropy(logits, labels)

def visualize_tsne(features, epoch, save_dir='tsne_images'):
    # Reduce dimensions with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c='blue', label='Features')
    plt.title(f't-SNE Visualization (Epoch {epoch})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'tsne_epoch_{epoch}.png'))
    plt.close()

class SmaliOpcodeDetailCaptureCNN(nn.Module):
    """
    Renamed from DetailCaptureCNN.
    Used for Smali Opcode feature extraction.
    """
    def __init__(self):
        super(SmaliOpcodeDetailCaptureCNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.projection_head(x)
        return F.normalize(x, dim=1)

class SOOpcodeDetailCaptureCNN(nn.Module):
    """
    Renamed from ModifiedDetailCaptureCNN.
    Used for SO Opcode feature extraction.
    """
    def __init__(self):
        super(SOOpcodeDetailCaptureCNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 94 -> 47

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 47 -> 23
        )

        # 64 * 23 * 23 = 33856
        self.projection_head = nn.Sequential(
            nn.Linear(64 * 23 * 23, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

    def forward_once(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.projection_head(x)
        # 关键：输出已经归一化，后续只需做点积
        return F.normalize(x, dim=1)

    def forward(self, input1, input2=None):
        # 支持单个输入（用于特征提取）或成对输入（用于训练）
        if input2 is None:
            return self.forward_once(input1)
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

def calculate_metrics(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return recall, accuracy

if __name__ == "__main__":
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = r'/newdisk/liuzhuowu/lzw/apks_wdj/'
    # Note: NPZPairDataset was referenced but not defined in the original file, using VectorDataset instead which is defined
    dataset = VectorDataset(data_dir)

    # Split dataset into train and test sets
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    detail_cnn = SmaliOpcodeDetailCaptureCNN().to(device)
    optimizer = optim.Adam(detail_cnn.parameters(), lr=0.001)

    for epoch in tqdm(range(100)):
        detail_cnn.train()
        all_features = []
        for orig_batch, repark_batch in train_loader:
            orig_batch, repark_batch = orig_batch.to(device), repark_batch.to(device)
            optimizer.zero_grad()
            
            features1 = detail_cnn(orig_batch)
            features2 = detail_cnn(repark_batch)
            
            features = torch.cat([features1, features2], dim=0)
            loss = contrastive_loss(features)
            
            loss.backward()
            optimizer.step()
            all_features.append(features.detach().cpu().numpy())

        # After each epoch, visualize the embeddings using t-SNE
        all_features = np.concatenate(all_features, axis=0)
        visualize_tsne(all_features, epoch + 1)

        # Calculate recall and accuracy on test set
        detail_cnn.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for orig_batch, repark_batch in test_loader:
                orig_batch, repark_batch = orig_batch.to(device), repark_batch.to(device)
                features1 = detail_cnn(orig_batch)
                features2 = detail_cnn(repark_batch)
                
                similarities = F.cosine_similarity(features1, features2)
                test_preds.extend(similarities.cpu().numpy())
                test_labels.extend([1] * len(similarities))  # All repark samples are positives

        test_preds = torch.tensor(test_preds)
        test_labels = torch.tensor(test_labels)
        threshold = 0.7  # Adjust the threshold as necessary
        binary_preds = (test_preds >= threshold).int()
        recall, accuracy = calculate_metrics(binary_preds, test_labels)

        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

    model_save_path = 'detail_capture_cnn.pth'
    torch.save(detail_cnn.state_dict(), model_save_path)
    print("Model saved.")
    print("Training completed.")
