import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_loader import get_dataloaders
from model import AgeGenderResNet50

# ---------------------------
# Settings
# ---------------------------
DATASET_PATH = r"E:\utkface_aligned_cropped\UTKFace"
BATCH_SIZE = 8
EPOCHS = 30
LR = 0.00000001
PATIENCE = 90   # Early stopping patience

CHECKPOINT_PATH = "checkpoint.pth"
BEST_MODEL_PATH = "best_age_gender_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ---------------------------
# Load data
# ---------------------------
train_loader, val_loader, test_loader = get_dataloaders(
    DATASET_PATH, batch_size=BATCH_SIZE, num_workers=0
)

# ---------------------------
# Model
# ---------------------------
model = AgeGenderResNet50().to(DEVICE)

age_criterion = nn.L1Loss()
gender_criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)

start_epoch = 0
best_val_acc = 0.0
epochs_no_improve = 0

# ---------------------------
# Resume from checkpoint (best option)
# ---------------------------
if os.path.exists(CHECKPOINT_PATH):

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint["best_val_acc"]

    print(f"✅ Resumed from checkpoint at epoch {start_epoch}")

# ---------------------------
# If only best model exists
# ---------------------------
elif os.path.exists(BEST_MODEL_PATH):

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    print("✅ Loaded best saved model weights")

# ---------------------------
# Training Loop
# ---------------------------
for epoch in range(start_epoch, EPOCHS):

    model.train()
    total_age_loss = 0
    total_correct = 0
    total_samples = 0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
        dynamic_ncols=True
    )

    for images, ages, genders in train_bar:

        images = images.to(DEVICE)
        ages = ages.float().to(DEVICE)
        genders = genders.to(DEVICE)

        optimizer.zero_grad()

        age_pred, gender_pred = model(images)

        age_loss = age_criterion(age_pred.squeeze(), ages)
        gender_loss = gender_criterion(gender_pred, genders)

        loss = age_loss + gender_loss
        loss.backward()
        optimizer.step()

        total_age_loss += age_loss.item() * images.size(0)

        _, predicted = torch.max(gender_pred, 1)
        total_correct += (predicted == genders).sum().item()
        total_samples += genders.size(0)

        train_bar.set_postfix({
            "Age MAE": f"{total_age_loss/total_samples:.3f}",
            "G": f"{100*total_correct/total_samples:.2f}%"
        })

    # ---------------------------
    # Validation
    # ---------------------------
    model.eval()

    val_age_loss = 0
    val_true = []
    val_pred = []

    with torch.no_grad():
        for images, ages, genders in val_loader:

            images = images.to(DEVICE)
            ages = ages.float().to(DEVICE)
            genders = genders.to(DEVICE)

            age_pred, gender_pred = model(images)

            val_age_loss += age_criterion(
                age_pred.squeeze(), ages
            ).item() * images.size(0)

            _, predicted = torch.max(gender_pred, 1)

            val_true.extend(genders.cpu().numpy())
            val_pred.extend(predicted.cpu().numpy())

    val_acc = accuracy_score(val_true, val_pred)
    val_precision = precision_score(val_true, val_pred)
    val_recall = recall_score(val_true, val_pred)
    val_f1 = f1_score(val_true, val_pred)

    print(f"\nEpoch {epoch+1} Summary")
    print(f"Train Age MAE: {total_age_loss/total_samples:.3f}")
    print(f"Train Gender Acc: {100*total_correct/total_samples:.2f}%")
    print(f"Val Age MAE: {val_age_loss/len(val_loader.dataset):.3f}")
    print(f"Val Gender Acc: {100*val_acc:.2f}%")
    print(f"Precision: {100*val_precision:.2f}% | Recall: {100*val_recall:.2f}% | F1: {100*val_f1:.2f}%")

    # ---------------------------
    # Check improvement
    # ---------------------------
    if val_acc > best_val_acc:

        best_val_acc = val_acc
        epochs_no_improve = 0

        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("✅ New best model saved")

    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")

    # ---------------------------
    # Save checkpoint every epoch
    # ---------------------------
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc
    }, CHECKPOINT_PATH)

    print("💾 Checkpoint saved")

    # ---------------------------
    # Early stopping (ANTI-overfitting)
    # ---------------------------
    if epochs_no_improve >= PATIENCE:

        print("🛑 Early stopping triggered (overfitting prevention)")
        break


# ---------------------------
# Save final model
# ---------------------------
torch.save(model.state_dict(), "final_model.pth")

print("\nTraining complete.")
print("Best model saved as:", BEST_MODEL_PATH)
print("Final model saved as: final_model.pth")
