import torch

import utils.models as models
import utils.dataset as dataset

if __name__ == '__main__':
    pretrained = True
    use_cuda = False

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    model = models.get_model("alexnet", pretrained=pretrained).to(device)

    data = dataset.get_dataset("mnist")