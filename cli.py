"""Running the nst."""
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
from nst import transfer, VGG_Activation
from utils import show_grid_tensor

if __name__ == "__main__":
    available_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{available_device} is available")
    device = torch.device(device=available_device)
    # device
    content_path = sys.argv[1]
    style_path = sys.argv[2]

    content = Image.open(content_path)
    style = Image.open(style_path)

    transformImage2Tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    content_tensor = transformImage2Tensor(content).unsqueeze(0).to(device)
    style_tensor = transformImage2Tensor(style).unsqueeze(0).to(device)
    generate_tensor = content_tensor.clone()

    layer_to_watch = [0, 5, 10, 19, 28]  # layers to get the activation from

    vgg_layer_act = VGG_Activation(layer_to_watch).to(device)

    out_image, generation_lists = transfer(vgg_layer_act,
                                           generate_tensor,
                                           content_tensor,
                                           style_tensor,
                                           layer_to_watch,
                                           alpha=1,
                                           beta=0.1,
                                           lr=0.4,
                                           iteration=600,
                                           device=device)
    # clamp = 386
    # no clamp = 380
    # 0.1 = 630 - 701
    # 0.001 = 5.63e+3

    with torch.no_grad():
        plt.imshow(out_image.cpu().squeeze()
                   .permute(1, 2, 0).clamp(max=1, min=0))
        plt.show()

    log_tensor = torch.stack(generation_lists)
    show_grid_tensor(log_tensor, n=6)
