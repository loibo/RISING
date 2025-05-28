import glob
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MayoDataset(Dataset):
    def __init__(self, data_path, convergence_path=None, data_shape=256):
        super().__init__()

        self.data_path = data_path
        self.convergence_path = convergence_path
        self.data_shape = data_shape

        # We expect data_path to be like "./data/Mayo/train" or "./data/Mayo/test"
        self.fname_list = glob.glob(f"{data_path}/*/*.png")
        if self.convergence_path:
            self.convergence_fname_list = glob.glob(f"{convergence_path}/*/*.png")

        # Define transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.data_shape, antialias=True),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        if self.convergence_path:
            # Load the idx's image from fname_list
            true_path = self.fname_list[idx]
            convergence_path = self.convergence_fname_list[idx]

            # To load the image as grey-scale
            x = self.transform(Image.open(true_path).convert("L"))
            x_IS = self.transform(Image.open(convergence_path).convert("L"))
            return x, x_IS

        # Load the idx's image from fname_list
        img_path = self.fname_list[idx]

        # To load the image as grey-scale
        x = self.transform(Image.open(img_path).convert("L"))
        return x

    def get_path(self, idx):
        return self._parse_image_path(self.fname_list[idx])

    def _parse_image_path(self, path_str):
        # Convert to a Path object and normalize slashes
        path = Path(path_str).resolve()

        # Search for 'Mayo' in the path parts
        try:
            mayo_index = path.parts.index("Mayo")
            subfolder = Path(
                *path.parts[mayo_index:-1]
            )  # From 'Mayo' to parent of the file
        except ValueError:
            raise ValueError("'Mayo' not found in path: " + str(path))

        filename = path.name  # The image filename like '95.png'

        return subfolder.as_posix(), filename
