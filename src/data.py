from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

def load_data(base_dir):
    path = Path(base_dir) / "metadata.csv"
    df = pd.read_csv(str(path))
    records = []
    for key, row in df.iterrows():
        if "COVID-19" in row["finding"]:
            label = 1
        else:
            label = 0
        ext = row["filename"].split(".")[-1]
        if not (ext == "jpg" or ext == "png"):
            continue
        records += [{
                "label": label,
                "filename": row["filename"]
            }]
    return records

class CovidChestxrayDataset(Dataset):
    def __init__(self, records, base_dir, transfrom=None):
        self.records = records
        self.transfrom = transfrom
        self.base_dir = str(Path(base_dir) / "images")
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        item = self.records[index]
        label = item["label"]
        img = Image.open(self.base_dir + "/" + item["filename"]).convert("L")
        if self.transfrom:
            img = self.transfrom(img)
        return img, label

if __name__ == "__main__":
    import numpy as np
    base_dir = "/home/masashi/projects/opt/covid-chestxray-dataset"
    records = load_data(base_dir)
    dataset = CovidChestxrayDataset(records, base_dir)
    img, label = dataset[0]
    print(np.asarray(img).shape, label)