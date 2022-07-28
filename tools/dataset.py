from paddle.io import Dataset
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed


class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        super(Repeat, self).__init__()
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]


class MVTecAT(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root_dir, defect_name, size, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            defect_name (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        super(MVTecAT, self).__init__()
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size

        if self.mode == "train":
            self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            self.imgs = Parallel(n_jobs=10)(
                delayed(lambda file: Image.open(file).resize((size, size)).convert("RGB"))(file) for file in
                self.image_names)
        else:
            self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.image_names[idx]
            # filename每一个部分都是分开的
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size, self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "good"
