import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import get_dataloaders
from model import AgeGenderResNet50

DATASET_PATH = r"E:\utkface_aligned_cropped\UTKFace"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader = get_dataloaders(DATASET_PATH, batch_size=BATCH_SIZE, num_workers=0)

model = AgeGenderResNet50()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

all_true, all_pred = [], []

with torch.no_grad():
    for images, ages, genders in test_loader:
        images = images.to(device)
        genders = genders.to(device)

        _, gender_pred = model(images)
        _, predicted = torch.max(gender_pred, 1)
        all_true.extend(genders.cpu().numpy())
        all_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(all_true, all_pred)
precision = precision_score(all_true, all_pred)
recall = recall_score(all_true, all_pred)
f1 = f1_score(all_true, all_pred)

print("===== Test Set Evaluation =====")
print(f"Accuracy: {100*acc:.2f}%")
print(f"Precision: {100*precision:.2f}%")
print(f"Recall: {100*recall:.2f}%")
print(f"F1 Score: {100*f1:.2f}%")