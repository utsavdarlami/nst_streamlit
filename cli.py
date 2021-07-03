"""Running the nst."""
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
from nst import transfer, VGG_Activation
from utils import show_grid_tensor
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    available_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{available_device}, is available")
    # device
    device = torch.device(device=available_device)

    content_path = sys.argv[1]
    style_path = sys.argv[2]

    # Writer will output to Neural_Style_Logs + content image name

    writer = SummaryWriter("Neural_Style_Logs_"+content_path.replace(".jpg",""))

    content = Image.open(content_path)
    style = Image.open(style_path)

    rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    rgb_std = torch.tensor([0.229, 0.224, 0.225])

    transformImage2Tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std)
    ])

    content_tensor = transformImage2Tensor(content).unsqueeze(0).to(device)
    style_tensor = transformImage2Tensor(style).unsqueeze(0).to(device)
    generate_tensor = content_tensor.clone()

    layer_to_watch = [0, 5, 10, 19, 28]  # layers to get the activation from

    vgg_layer_act = VGG_Activation(layer_to_watch).to(device)

    out_image, _ = transfer(vgg_layer_act,
                                           generate_tensor,
                                           content_tensor,
                                           style_tensor,
                                           layer_to_watch,
                                           alpha=1,
                                           beta=1e-3,
                                           lr=0.4,
                                           iteration=600,
                                           device=device,
                                           writer=writer)
    # clamp = 386
    # no clamp = 380
    # 0.1 = 630 - 701
    # 0.001 = 5.63e+3

    with torch.no_grad():
        # image_gen = (out_image.cpu().squeeze()
                        # .permute(1, 2, 0).clamp(max=1, min=0))
        # plt.imshow(image_gen)
        # plt.show()
        img = out_image.squeeze().cpu()
        img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
        writer.add_image('final generated image', img.permute(2, 0, 1), 0)


    # log_tensor = torch.stack(generation_lists)
    # show_grid_tensor(log_tensor, n=6)
    writer.close()

    img = out_image[0].cpu()

    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)

    pil_img = transforms.ToPILImage()(img.permute(2, 0, 1))

    pil_img.save("generated_img.png")
    print("DOne")
