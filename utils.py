"""helper functions."""
import matplotlib.pyplot as plt
from torchvision import transforms


transformTensor2Image = transforms.ToPILImage()


def show_tensor(batch, title=None):
    """Plot the tensor of size (1,c,h,w)."""
    image = batch.cpu().clone()
    image = batch.squeeze(0)
    image = transformTensor2Image(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()


def show_grid_tensor(grid_im, n):
    """To show the activation based on the given tensor grid."""
    fig = plt.figure(figsize=(15, 15))
    for i in range(n):
        ax = plt.subplot(5, 2, i+1)
        plt.imshow(grid_im[i].permute(1, 2, 0).clamp(max=1, min=0))
        plt.axis("off")
        plt.title(f"{i}")
    plt.savefig("logs.png")
    return ax, fig
