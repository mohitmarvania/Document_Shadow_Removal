from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


# SD7k Dataset Class
class SD7KDataset(Dataset):
    def __init__(self, root_dir, phase="train", target_res=(1024, 1024)):
        """
        :param root_dir: Path to SD7K dataset
        :param phase: 'train' or 'test'
        :param target_size: Resolution to which images will be resized. (By default = 1024 x 1024)
        """
        self.phase = phase
        self.input_dir = os.path.join(root_dir, phase, "input")
        self.target_dir = os.path.join(root_dir, phase, "target")
        self.filenames = [f for f in sorted(os.listdir(self.input_dir)) if f.lower().endswith(".png")]
        self.target_res = target_res
        self.transform = transforms.Compose([
            transforms.Resize(target_res),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.filenames[idx])
        target_path = os.path.join(self.target_dir, self.filenames[idx])

        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)

        return {
            'input': input_tensor,  # High-res shadow image.
            'target': target_tensor,  # High-res shadow-free image.
            'filenames': self.filenames[idx]
        }
