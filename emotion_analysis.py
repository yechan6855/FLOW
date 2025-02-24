import os
import glob
import pickle
import numpy as np
import librosa
import librosa.effects
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
# 0. 재현성을 위한 시드 설정
# =============================================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# =============================================================================
# 1. 오디오 전처리 및 증강 함수 (RAVDESS용)
# =============================================================================
def augment_audio(y, sr):
    """원본, 시간 스트레칭, 피치 쉬프트, 노이즈 추가, 랜덤 시프트"""
    augmented = []
    augmented.append(y)  # 원본
    try:
        augmented.append(librosa.effects.time_stretch(y, rate=0.9))
        augmented.append(librosa.effects.time_stretch(y, rate=1.1))
    except Exception:
        pass
    try:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
    except Exception:
        pass
    noise = np.random.normal(0, 0.005, y.shape)
    augmented.append(y + noise)
    shift = np.random.randint(int(sr * 0.5))
    augmented.append(np.roll(y, shift))
    return augmented

def spec_augment(mfcc, time_masking=2, freq_masking=2, max_time_mask=15, max_freq_mask=8):
    """MFCC 스펙트로그램에 대해 시간 및 주파수 마스킹 적용"""
    augmented = mfcc.copy()
    num_frames = augmented.shape[1]
    num_mfcc = augmented.shape[0]
    for _ in range(time_masking):
        t = np.random.randint(0, max_time_mask)
        if num_frames - t > 0:
            t0 = np.random.randint(0, num_frames - t)
            augmented[:, t0:t0+t] = 0
    for _ in range(freq_masking):
        f = np.random.randint(0, max_freq_mask)
        if num_mfcc - f > 0:
            f0 = np.random.randint(0, num_mfcc - f)
            augmented[f0:f0+f, :] = 0
    return augmented

def extract_features(audio_path, duration=4.0, sr=22050, n_mfcc=40, target_frames=173):
    """
    오디오 파일을 로드한 후 증강 및 MFCC 추출 (출력: list of (40, target_frames))
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return []
    aug_signals = augment_audio(y, sr)
    features = []
    for sig in aug_signals:
        mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < target_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, target_frames - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :target_frames]
        if np.random.rand() < 0.5:
            mfcc = spec_augment(mfcc)
        features.append(mfcc)
    return features

# =============================================================================
# 2. WESAD 데이터 로드 및 세그먼트화 (.pkl 파일 활용)
# =============================================================================
def load_wesad_sample_from_pkl(pkl_path, window_size=300):
    """
    .pkl 파일을 로드하여, chest와 wrist 센서 데이터를 결합한 후
    윈도우(=window_size) 단위로 segmentation하고, 다수결로 라벨을 결정합니다.
    반환: list of tuples (phys_seg, mapped_label)
    - phys_seg: (9, window_size) numpy array
    - mapped_label: WESAD 원래 라벨 {0,1,2}를 {0:Neutral, 1:Angry, 2:Happy}로 매핑
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    chest = data['signal']['chest']
    wrist = data['signal']['wrist']
    ECG = chest.get('ECG')
    EDA = chest.get('EDA')
    EMG = chest.get('EMG')
    Resp = chest.get('Resp') if 'Resp' in chest else chest.get('Respiration')
    Temp = chest.get('Temp') if 'Temp' in chest else chest.get('Temperature')
    BVP = wrist.get('BVP')
    ACC = wrist.get('ACC')
    base_length = len(ECG) if ECG is not None else 0
    if EDA is None and base_length > 0: EDA = np.zeros(base_length)
    if EMG is None and base_length > 0: EMG = np.zeros(base_length)
    if Resp is None and base_length > 0: Resp = np.zeros(base_length)
    if Temp is None and base_length > 0: Temp = np.zeros(base_length)
    if BVP is None and base_length > 0: BVP = np.zeros(base_length)
    if ACC is None and base_length > 0: ACC = np.zeros((base_length, 3))
    lengths = [len(x) for x in [ECG, EDA, EMG, Resp, Temp, BVP] if x is not None]
    if ACC is not None:
        lengths.append(ACC.shape[0])
    min_length = min(lengths) if lengths else 0
    segments = []
    label_array = np.array(data['label'])
    for start in range(0, min_length - window_size + 1, window_size):
        end = start + window_size
        # 각 센서 신호를 (1, window_size) 형태로 맞춤 (만약 이미 2차원 배열라면 필요 없음)
        seg_ECG = np.array(ECG[start:end]).reshape(1, -1)
        seg_EDA = np.array(EDA[start:end]).reshape(1, -1)
        seg_EMG = np.array(EMG[start:end]).reshape(1, -1)
        seg_Resp = np.array(Resp[start:end]).reshape(1, -1)
        seg_Temp = np.array(Temp[start:end]).reshape(1, -1)
        seg_BVP = np.array(BVP[start:end]).reshape(1, -1)
        # 가속도는 2차원 데이터이므로, 슬라이싱 후 전치하여 (3, window_size)로 만듦
        seg_ACC = ACC[start:end, :]  # (window_size, 3)
        seg_ACC = seg_ACC.T          # (3, window_size)
        phys_seg = np.vstack([seg_ECG, seg_EDA, seg_EMG, seg_Resp, seg_Temp, seg_BVP, seg_ACC])
        seg_labels = label_array[start:end]
        if len(seg_labels) > 0:
            majority_label = int(np.argmax(np.bincount(seg_labels)))
        else:
            majority_label = 0
        # 매핑: WESAD 원래 라벨 {0: Baseline, 1: Stress, 2: Amusement}
        # → {0: Neutral (0), 1: Stress → Angry (4), 2: Amusement → Happy (2)}
        mapping = {0: 0, 1: 4, 2: 2}
        mapped_label = mapping.get(majority_label, 0)
        segments.append((phys_seg, mapped_label))
    return segments

def load_all_wesad_samples(wesad_root, window_size=300):
    """
    wesad_root 내의 모든 Subject 폴더(S2, S3, …)에 대해 .pkl 파일을 로드하고
    세그먼트별 샘플을 반환합니다.
    """
    samples = []
    subject_dirs = [d for d in os.listdir(wesad_root) if d.startswith("S")]
    for subj in subject_dirs:
        subj_path = os.path.join(wesad_root, subj)
        pkl_files = glob.glob(os.path.join(subj_path, "*.pkl"))
        for pkl_file in pkl_files:
            segs = load_wesad_sample_from_pkl(pkl_file, window_size=window_size)
            if segs:
                for (phys_seg, seg_label) in segs:
                    sample = {
                        "audio": get_zero_audio(),  # 음성 없음 → 0 배열 (40,173)
                        "phys": phys_seg,           # (9, window_size), window_size=300
                        "label": seg_label
                    }
                    samples.append(sample)
    return samples

# =============================================================================
# 3. 누락 모달리티 대비 제로 배열 함수
# =============================================================================
def get_zero_audio():
    """음성 데이터가 없을 때 (40×173) 0 배열 반환"""
    return np.zeros((40, 173), dtype=np.float32)

def get_zero_phys():
    """생리 데이터가 없을 때 (9×300) 0 배열 반환"""
    return np.zeros((9, 300), dtype=np.float32)

# =============================================================================
# 4. 데이터셋 구성: RAVDESS (Audio)와 WESAD (Physiology)
# =============================================================================
# ----- 4-1. RAVDESS 데이터셋 로딩 -----
ravdess_dir = r"C:\Users\김진녕\RAVDESS"  # RAVDESS wav 파일 폴더
# RAVDESS 파일명 형식: "03-01-06-01-02-01-12.wav"
# 조건: Modality == "03", Vocal Channel == "01", Repetition == "01"
# 감정 매핑 (파일명의 3번째 필드): "01": Neutral, "02": Calm, "03": Happy, "04": Sad, "05": Angry, "06": Fearful, "07": Disgust, "08": Surprised
ravdess_samples = []
rav_files = glob.glob(os.path.join(ravdess_dir, "*.wav"))
print(f"RAVDESS 파일 수: {len(rav_files)}")
emotion_map = {"01": 0, "02": 1, "03": 2, "04": 3, "05": 4, "06": 5, "07": 6, "08": 7}
for file_path in rav_files:
    filename = os.path.basename(file_path)
    parts = filename.split("-")
    if len(parts) != 7:
        continue
    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor_part = parts
    repetition = repetition  # ex: "01"
    if modality != "03" or vocal_channel != "01" or repetition != "01":
        continue
    if emotion_code not in emotion_map:
        continue
    feats = extract_features(file_path)
    for feat in feats:
        sample = {
            "audio": feat,             # (40, 173)
            "phys": get_zero_phys(),   # 생리 없음 → 0 배열 (9,300)
            "label": emotion_map[emotion_code]
        }
        ravdess_samples.append(sample)
print(f"RAVDESS 최종 샘플 수: {len(ravdess_samples)}")

# ----- 4-2. WESAD 데이터셋 로딩 -----
wesad_root = r"C:\Users\김진녕\WESAD"  # WESAD 폴더 (내부에 S2, S3, …)
wesad_samples = load_all_wesad_samples(wesad_root, window_size=300)
print(f"WESAD 최종 샘플 수: {len(wesad_samples)}")

# ----- 4-3. 두 데이터셋 결합 -----
combined_samples = ravdess_samples + wesad_samples
print(f"전체 결합 샘플 수: {len(combined_samples)}")

# =============================================================================
# 5. PyTorch Dataset 및 DataLoader (멀티모달)
# =============================================================================
class CombinedEmotionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio = torch.tensor(sample["audio"], dtype=torch.float32).unsqueeze(0)  # (1,40,173)
        phys  = torch.tensor(sample["phys"], dtype=torch.float32)                # (9,300)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return audio, phys, label

all_labels = [s["label"] for s in combined_samples]
train_samples, test_samples = train_test_split(combined_samples, test_size=0.2, stratify=all_labels, random_state=seed)
batch_size = 16
train_dataset = CombinedEmotionDataset(train_samples)
test_dataset  = CombinedEmotionDataset(test_samples)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# =============================================================================
# 6. 멀티모달 모델 정의: Audio Branch + Physiology Branch + Fusion
# =============================================================================
# ----- Audio Branch (CNN + 2층 LSTM) -----
class AudioBranch(nn.Module):
    def __init__(self, lstm_hidden=256):
        super(AudioBranch, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1344, 640)
        self.bn_fc1 = nn.BatchNorm1d(640)
        self.lstm = nn.LSTM(640, lstm_hidden, num_layers=2, batch_first=True, dropout=0.5)
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   # (B,16,20,86)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))     # (B,32,10,43)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))     # (B,64,5,21)
        x = x.permute(0, 2, 1, 3)                               # (B,5,64,21)
        x = x.reshape(x.size(0), x.size(1), -1)                # (B,5,1344)
        x = self.dropout(x)
        B, T, D = x.size()
        x = x.reshape(B * T, D)
        x = torch.relu(self.bn_fc1(self.fc1(x)))               # (B*T,640)
        x = x.reshape(B, T, -1)                                # (B, T, 640)
        x, _ = self.lstm(x)                                   # (B, T, lstm_hidden)
        x = x[:, -1, :]                                       # 마지막 time step: (B, lstm_hidden)
        return x  # (B, 256)

# ----- Physiology Branch (1D CNN) -----
class PhysBranch(nn.Module):
    def __init__(self, out_dim=128):
        super(PhysBranch, self).__init__()
        self.conv1 = nn.Conv1d(9, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        # 입력: (B, 9, 300) → 연산 후 예상 출력: (B, 128, 37) → flatten하면 128 * 37 = 4736
        self.fc = nn.Linear(4736, out_dim)  # 수정된 부분: 4736로 변경
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # (B,32,150)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))    # (B,64,75)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))    # (B,128,37) (대략)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten → (B, 4736)
        x = torch.relu(self.fc(x)) # (B, out_dim) → (B,128)
        return x

# ----- Fusion: Audio + Physiology -----
class MultiModalModel(nn.Module):
    def __init__(self, num_classes=8):
        super(MultiModalModel, self).__init__()
        self.audio_branch = AudioBranch(lstm_hidden=256)
        self.phys_branch  = PhysBranch(out_dim=128)
        self.fc1 = nn.Linear(256 + 128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, audio, phys):
        audio_feat = self.audio_branch(audio)  # (B,256)
        phys_feat  = self.phys_branch(phys)      # (B,128)
        fused = torch.cat((audio_feat, phys_feat), dim=1)  # (B,384)
        x = torch.relu(self.fc1(fused))
        x = self.dropout(x)
        out = self.fc2(x)
        return out

# =============================================================================
# 7. 학습 및 평가
# =============================================================================
learning_rate = 0.0005
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalModel(num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for audios, phys, labels in train_loader:
        audios, phys, labels = audios.to(device), phys.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(audios, phys)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * audios.size(0)
    train_loss = running_loss / len(train_dataset)
    
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for audios, phys, labels in test_loader:
            audios, phys, labels = audios.to(device), phys.to(device), labels.to(device)
            outputs = model(audios, phys)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * audios.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss /= len(test_dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

model.eval()
final_preds = []
final_labels = []
with torch.no_grad():
    for audios, phys, labels in test_loader:
        audios, phys, labels = audios.to(device), phys.to(device), labels.to(device)
        outputs = model(audios, phys)
        preds = outputs.argmax(dim=1)
        final_preds.extend(preds.cpu().numpy())
        final_labels.extend(labels.cpu().numpy())
final_acc = accuracy_score(final_labels, final_preds)
print("\nFinal Test Accuracy:", final_acc)
print("\nClassification Report:")
print(classification_report(final_labels, final_preds, target_names=["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"], zero_division=1))
