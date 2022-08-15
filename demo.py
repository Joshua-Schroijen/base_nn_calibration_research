import fire
import os
import torch
import torchvision as tv
from torch.utils.data.sampler import SubsetRandomSampler
import statistics

from models import DenseNet
from temperature_scaling import ModelWithTemperature

def demo(data, save, depth=40, growth_rate=12, batch_size=256):
    model_filename = os.path.join(save, 'model.pth')
    if not os.path.exists(model_filename):
        raise RuntimeError('Cannot find file %s to load' % model_filename)
    state_dict = torch.load(model_filename)

    valid_indices_filename = os.path.join(save, 'valid_indices.pth')
    if not os.path.exists(valid_indices_filename):
        raise RuntimeError('Cannot find file %s to load' % valid_indices_filename)
    valid_indices = torch.load(valid_indices_filename)

    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    valid_set = tv.datasets.CIFAR100(data, train=True, transform=test_transforms, download=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(valid_indices))

    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]
    orig_model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=100
    ).cuda()
    orig_model.load_state_dict(state_dict)

    model = ModelWithTemperature(orig_model)
    diffs = []
    for _ in range(100):
      old_ece = model.set_temperature(valid_loader)
      new_ece = model.set_temperature(valid_loader)
      diffs.append((new_ece - old_ece))

    print(f"Mean ECE difference after calibrating twice: {statistics.mean(diffs)}")

if __name__ == '__main__':
    fire.Fire(demo)
