import torch
from dataset import LiverDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


CSV_LABELS = './SeriesClassificationKey.csv'
CSV_LABELS_TYPE = './SequenceTypes.csv'
ROOT_DIR = './Series_Classification'


transformer = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()
])

dataset = LiverDataset(csv_labels=CSV_LABELS, csv_labels_type=CSV_LABELS_TYPE, root_dir=ROOT_DIR, transform=transformer)
loader = DataLoader(dataset=dataset, batch_size=512, num_workers=1)

def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    loop = tqdm(loader)

    for data, _ in loop:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std


if __name__ == "__main__":
    mean, std = get_mean_std(loader)
    print(mean)
    print(std)

    # data = next(iter(loader))
    # print(data[0].mean())
    # print(data[0].std())