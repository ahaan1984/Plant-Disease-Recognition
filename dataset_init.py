import os
from torchvision.io import read_image
from torch.utils.data import Dataset 

class PlantDiseaseDataset(Dataset):
    def __init__(self, path, image_size=(256, 256), channels=("RGB"), 
                 transform=None, target_transform=None):
        self.__image_labels = []
        self.image_size = image_size
        self.channels = channels
        self.transform = transform
        self.target_transform = target_transform

        if os.path.exists(path):
            self.labels = os.listdir(path)
            for label in self.labels:
                label_path = os.path.join(path, label)
                if os.path.isdir(label_path):
                    files = os.listdir(label_path)
                    for file in files:
                        if file.endswith("jpg"):
                            image_path = os.path.join(label_path, file)
                            self.__image_labels.append((image_path, label))
                        else:
                            pass
                else:
                    pass
        else:
            pass

    def __len__(self):
        return len(self.__image_labels)
    
    def __getitem__(self, idx):
        path, label = self.__image_labels[idx]
        image = read_image(path)
        label = self.labels.index(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label