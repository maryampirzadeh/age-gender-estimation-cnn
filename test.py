import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_loader import get_dataloaders
from model import AgeGenderResNet50


# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = r"E:\utkface_aligned_cropped\UTKFace"
BATCH_SIZE = 8

MODEL_PATH = "best_age_gender_model.pth"
#MODEL_PATH = "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Device:", DEVICE)


# =========================================================
# LOAD TEST DATA ONLY
# =========================================================

_, _, test_loader = get_dataloaders(
    DATASET_PATH,
    batch_size=BATCH_SIZE,
    num_workers=0
)


# =========================================================
# LOAD MODEL
# =========================================================

model = AgeGenderResNet50().to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.eval()

print("✅ Model loaded from:", MODEL_PATH)


# =========================================================
# LOSS (for MAE)
# =========================================================

age_criterion = nn.L1Loss()


# =========================================================
# TEST LOOP
# =========================================================

test_age_loss = 0
test_true = []
test_pred = []

with torch.no_grad():

    for images, ages, genders in test_loader:

        images = images.to(DEVICE)
        ages = ages.float().to(DEVICE)
        genders = genders.to(DEVICE)

        age_pred, gender_pred = model(images)

        # Age MAE
        test_age_loss += age_criterion(age_pred.squeeze(), ages).item() * images.size(0)

        # Gender predictions
        _, predicted = torch.max(gender_pred, 1)

        test_true.extend(genders.cpu().numpy())
        test_pred.extend(predicted.cpu().numpy())


# =========================================================
# METRICS
# =========================================================

test_mae = test_age_loss / len(test_loader.dataset)

test_acc = accuracy_score(test_true, test_pred)
test_precision = precision_score(test_true, test_pred)
test_recall = recall_score(test_true, test_pred)
test_f1 = f1_score(test_true, test_pred)


# =========================================================
# RESULTS
# =========================================================

print("\nTEST RESULTS")
print("===================================")

print(f"Age MAE: {test_mae:.3f} years")

print(f"Gender Accuracy: {100*test_acc:.2f}%")
print(f"Precision: {100*test_precision:.2f}%")
print(f"Recall:    {100*test_recall:.2f}%")
print(f"F1 Score:  {100*test_f1:.2f}%")

print("===================================")