import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class URLDataset(Dataset):
    def __init__(self, structural_csv, embeddings_csv=None, merge_embeddings=False, transform=None):
        """
        Initializes the dataset by loading the structural CSV and optionally merging the embeddings CSV.
        :param structural_csv: Path to the structural CSV file.
        :param embeddings_csv: Path to the embeddings CSV file (optional).
        :param merge_embeddings: Boolean flag to merge embeddings CSV.
        :param transform: Optional transformations.
        """
        self.data = pd.read_csv(structural_csv)
        
        if merge_embeddings and embeddings_csv:
            embeddings = pd.read_csv(embeddings_csv)
            self.data = pd.concat([self.data, embeddings.drop(columns=['type'])], axis=1)
        
        self.features = self.data.drop(columns=['type', 'url'])  # Drop non-numeric columns
        self.labels = self.data['type']
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def get_dataloaders(structural_csv, embeddings_csv=None, merge_embeddings=False, batch_size=32, val_split=0.1, test_split=0.1):
    """
    Loads the dataset and splits it into train, validation, and test sets.
    :param structural_csv: Path to the structural CSV file.
    :param embeddings_csv: Path to the embeddings CSV file (optional).
    :param merge_embeddings: Boolean flag to merge embeddings CSV.
    :param batch_size: Batch size for DataLoader.
    :param val_split: Fraction of data for validation.
    :param test_split: Fraction of data for testing.
    :return: Train, validation, and test DataLoaders.
    """
    dataset = URLDataset(structural_csv, embeddings_csv, merge_embeddings)
    
    total_size = len(dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    structural_path = "./data/merged/processed_urls.csv"
    embeddings_path = "./data/merged/url_embeddings.csv"
    
    train_loader, val_loader, test_loader = get_dataloaders(structural_path, embeddings_path, merge_embeddings=True)
    
    for batch in train_loader:
        inputs, labels = batch
        print("Batch Inputs Shape:", inputs.shape)
        print("Batch Labels Shape:", labels.shape)
        break
