import os
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================================================
# EXPORTING AGE AND GENDER
# =========================================================
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_age=100):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        #In the dataset every img has a description. first number is age
        #Second number is gender. 0 is male and 1 is female.
        #In 21_1_2_20170116165009080.jpg.chip, 21 is age and the person is female
        for file in os.listdir(root_dir):
            if not file.lower().endswith(".jpg"):
                continue
            try:
                parts = file.split("_")
                age = int(parts[0])
                gender = int(parts[1])
                if age > max_age:
                    continue
                self.samples.append((file, age, gender))
            except:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file, age, gender = self.samples[idx]
        img_path = os.path.join(self.root_dir, file)
        #3 layers
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        age = torch.tensor(age, dtype=torch.float32)
        gender = torch.tensor(gender, dtype=torch.long)
        return image, age, gender

# =========================================================
# DATA PROCESSING
# =========================================================
def get_dataloaders(dataset_path, batch_size=32, num_workers=4, seed=42):
    #80% for train, 10% for validation and 10% for test data
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_transform = transforms.Compose([
        #Resizes all input images to a fixed size of 224x224 pixels. 
        #This is common because most CNNs expect inputs of a specific size.
        transforms.Resize((224, 224)),
        #Randomly flips the images horizontally with a probability of 50% to make model learn better
        transforms.RandomHorizontalFlip(),
        #Rotates the images randomly by up to 10 degrees
        transforms.RandomRotation(10),
        #Converts the PIL Image objects into PyTorch tensors.
        #It also normalizes the pixel values to the range [0, 1].
        transforms.ToTensor(),
        #The values [0.485, 0.456, 0.406] are the mean values for RGB
        #[0.229, 0.224, 0.225] are the standard deviation values for the dataset 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    full_dataset = UTKFaceDataset(dataset_path)

    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(
        range(dataset_size),
        [train_size, val_size, test_size],
        generator=generator
    )
    # to apply all the ops on train, val and test data sets.
    train_dataset = Subset(UTKFaceDataset(dataset_path, transform=train_transform), train_idx.indices)
    val_dataset = Subset(UTKFaceDataset(dataset_path, transform=eval_transform), val_idx.indices)
    test_dataset = Subset(UTKFaceDataset(dataset_path, transform=eval_transform), test_idx.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader