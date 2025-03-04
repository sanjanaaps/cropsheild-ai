import os
import torch
from torchvision import datasets, transforms
from PIL import Image

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir, transform=None, partition_index=None, num_partitions=4):
    self.root_dir = root_dir
    self.transform = transform
        
        # Assuming the dataset is organized with one folder per class
    self.classes = sorted(os.listdir(root_dir))
        
        # Build a list of (image_path, label) tuples
    self.samples = []
    for label, class_name in enumerate(self.classes):
        class_dir = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.samples.append((os.path.join(class_dir, fname), label))
   
    if partition_index is not None:
            total_samples = len(self.samples)
            partition_size = total_samples // num_partitions
            start_index = partition_index * partition_size
            # For the last partition, include all remaining samples
            if partition_index == num_partitions - 1:
                self.samples = self.samples[start_index:]
            else:
                self.samples = self.samples[start_index:start_index + partition_size]
                print(f"Using partition {partition_index+1}/{num_partitions}: {len(self.samples)} samples")
    else:
        print(f"Using full dataset: {len(self.samples)} samples")

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    image_path, label = self.samples[idx]
    image = Image.open(image_path).convert('RGB')
    if self.transform:
        image = self.transform(image)
    return image, label