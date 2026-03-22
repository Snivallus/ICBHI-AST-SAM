import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import ASTFeatureExtractor
from tqdm import tqdm
import librosa
import argparse

from src.model import CustomAST
from preprocess import cyclic_padding, TARGET_SR, TARGET_SAMPLES


THRESHOLD = 3


class CustomDataset(Dataset):
    def __init__(self, X, processor):
        self.X = X
        self.processor = processor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        wav = self.X[idx]

        inputs = self.processor(
            wav,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_values = inputs.input_values.squeeze(0)

        return input_values


def inference(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Device: {DEVICE}")

    # ===== 1. LOAD AUDIO =====
    audio, _ = librosa.load(args.wav_path, sr=TARGET_SR)

    # ===== 2. CHUNKING =====
    window_size = 2 * TARGET_SR
    stride = 1 * TARGET_SR

    X = []
    for start in range(0, len(audio) - window_size + 1, stride):
        chunk = audio[start:start + window_size]
        processed_wav = cyclic_padding(chunk, TARGET_SAMPLES)
        X.append(processed_wav)

    if len(X) == 0:
        print("⚠️ 音频太短, 使用整体 padding")
        X.append(cyclic_padding(audio, TARGET_SAMPLES))

    X = np.array(X, dtype=np.float32)
    print(f"🔍 Total chunks: {len(X)}")

    # ===== 3. FEATURE EXTRACTOR =====
    processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    loader = DataLoader(
        CustomDataset(X, processor),
        batch_size=args.batch_size,
        shuffle=False
    )

    # ===== 4. MODEL =====
    model = CustomAST(num_classes=4).to(DEVICE)
    try:
        state_dict = torch.load(args.model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"⚠️ Load error: {e}")
        return

    model.eval()

    # ===== 5. INFERENCE =====
    all_logits = []

    with torch.no_grad():
        for inputs in tqdm(loader, desc="Infer"):
            inputs = inputs.to(DEVICE)
            logits = model(inputs) # [B, 4]
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)  # [N_chunks, 4]

    # ===== 6. AGGREGATION =====

    chunk_preds = torch.argmax(all_logits, dim=1)  # [N_chunks]
    counts = torch.bincount(chunk_preds, minlength=4)  # 统计每个类别数量

    count_normal = counts[0].item()
    count_crackle = counts[1].item()
    count_wheeze = counts[2].item()
    count_both = counts[3].item()

    print(f"📊 Chunk counts: Normal={count_normal}, Crackle={count_crackle}, Wheeze={count_wheeze}, Both={count_both}")

    # Rule 1: Both 优先
    if count_both >= THRESHOLD or (count_crackle >= THRESHOLD and count_wheeze >= THRESHOLD):
        final_pred = 3

    # Rule 2: 单类达到阈值
    elif count_crackle >= THRESHOLD:
        final_pred = 1
    elif count_wheeze >= THRESHOLD:
        final_pred = 2
    elif count_normal >= THRESHOLD:
        final_pred = 0

    # Rule 3: fallback (soft voting)
    else:
        avg_logits = torch.mean(all_logits, dim=0)
        final_pred = torch.argmax(avg_logits).item()

    # ===== 7. 输出 =====
    label_map = {
        0: "Normal",
        1: "Crackle",
        2: "Wheeze",
        3: "Both"
    }

    print(f"\n🎯 Final Prediction: {label_map[final_pred]}")

    return final_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pth", help="Path to trained model (.pth)")
    parser.add_argument("--wav_path", type=str, default="./data/ICBHI_final_database/103_2b2_Ar_mc_LittC2SE.wav", help="Path to audio file (.wav)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")

    args = parser.parse_args()
    inference(args)