"""Nst."""
import torch
from torch import nn, optim
# import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from collections import OrderedDict
from tqdm import tqdm


class VGG_Activation(nn.Module):
    """Module to extract activations using Pretrained VGG."""

    def __init__(self, layers_to_watch):
        """Initialize the layers."""
        super().__init__()
        self.vgg_features = torchvision.models.vgg19(pretrained=True).features
        # layers idx whose activations are to be stored
        self.layers_to_watch = layers_to_watch
        # calling the hook setter
        self.set_hooks()

        # to keep the selected layers activation
        self.layer_activation = OrderedDict()

    def layer_watch_hooks(self, layer_idx):
        """Define what hook should do."""
        def hook(module, input, output):
            """Add to the layer activation dic."""
            """
            Run hook.
            hook(module, input, output) -> None or modified output

            Parameters
            ----------
            module : torch.nn.Module
                The layer we want to attach the hook to.
            inp : tuple of torch.Tensor
                The input to the `forward` method.
            out : torch.Tensor
                The output of the `forward` method.

            """
            # print(layer_idx)
            # print(module)
            self.layer_activation[layer_idx] = output
        return hook

    def selected_layers(self):
        """Show the selcted layers."""
        for layer_idx, layer in enumerate(self.vgg_features):
            if layer_idx in self.layers_to_watch:
                print(layer)

    def set_hooks(self):
        """Set hooks for the layers that matches the layer idx."""
        for layer_idx, layer in enumerate(self.vgg_features):
            if layer_idx in self.layers_to_watch:
                layer.register_forward_hook(self.layer_watch_hooks(layer_idx))

    def forward(self, x):
        """Forward pass."""
        self.vgg_features.eval()
        _ = self.vgg_features(x)
        return self.layer_activation.copy()


def compute_content_loss(content_activation,
                         generate_activation,
                         layer_to_watch):
    """Compute content loss."""
    c_loss = 0.0
    for l in layer_to_watch:
        ca = content_activation[l]  # .detach()
        ga = generate_activation[l]  # .detach()
        # mse or l2 norm
        c_loss += torch.square(ga - ca.detach()).mean()
    return c_loss


def gram_matrix(in_):
    _, c, h, w = in_.size()
    features = in_.view(c, h * w)  # making Nl and Ml shape feature
    G = torch.mm(features, features.t())
    return G


def compute_style_loss(style_activation, generate_activation, layer_to_watch):
    # gram_matrix_style = OrderedDict()
    # gram_matrix_generate = OrderedDict()
    s_loss = 0
    for l in layer_to_watch:
        gram_matrix_style = gram_matrix(style_activation[l]) #.detach()
        gram_matrix_generate = gram_matrix(generate_activation[l]) #.detach()

        # l2 norm or mse
        s_loss += torch.mean((gram_matrix_generate - gram_matrix_style.detach()) ** 2)

        # el = F.mse_loss(gram_matrix_style[style_l] , gram_matrix_generate[style_l])
        # s_loss += el
    return s_loss

def compute_tv_loss(generate_tensor):
    """noise reduction method is total variation denoising."""

    # xi,j - xi+1,j
    noise_along_h = torch.abs(generate_tensor[:, :, 1:, :] -
                                generate_tensor[:, :, :-1, :]).mean()
    # xi,j - xi,j+1
    noise_along_w = torch.abs(generate_tensor[:, :, :, 1:] -
                                generate_tensor[:, :, :, :-1]).mean()
    return 0.5 * (noise_along_h  + noise_along_w)



def transfer(model, generate_tensor,
             content_tensor, style_tensor,
             layer_to_watch,
             optim_='lbfgs',
             alpha=1, beta=0.1,
             tv_weight=10,
             lr=1, iteration=30,
             device=torch.device(device='cpu'),
             st_bar=None,
             writer=None):
    """Transfering function."""
    saved_tensor_lists = []
    generate_tensor.requires_grad = True
    print(alpha/beta)
    rgb_mean = torch.tensor([0.485, 0.456, 0.406])

    rgb_std = torch.tensor([0.229, 0.224, 0.225])

    if optim_.lower() == 'adam':
        optimizer = optim.Adam([generate_tensor],
                               lr=lr)
    else:
        optimizer = optim.LBFGS([generate_tensor],
                                lr=lr)

    loop = tqdm(range(iteration), total=iteration, leave=True)

    for i in loop:
        loss_GC = loss_GS = torch.tensor(0.0).to(device)

        # generate_tensor.data.clamp_(0, 1)
        optimizer.zero_grad()

        with torch.no_grad():
            style_activation = model(style_tensor)
            content_activation = model(content_tensor)

        generate_activation = model(generate_tensor)

        loss_GC = compute_content_loss(content_activation,
                                       generate_activation,
                                       layer_to_watch)
        loss_GS = compute_style_loss(style_activation,
                                     generate_activation,
                                     layer_to_watch)
        content_loss = alpha*loss_GC
        style_loss = beta*loss_GS
        tv_loss = tv_weight * compute_tv_loss(generate_tensor)
        loss = content_loss + style_loss + tv_loss
        loss.backward()

        def closure():
            return loss

        loop.set_description(f"Epoch [{i + 1}/{iteration}]")
        loop.set_postfix(loss=loss.item())
        if st_bar:
            st_bar.progress((i+1)/iteration)
        if i % 100 == 0:
            # save_image(generate_tensor,
                       # "generated/generated_gogh_224_" + str(i) + ".png")
            # saved_tensor_lists.append(generate_tensor.clone().
            #                           detach().cpu().squeeze())

            # image_tensor =  generate_tensor.clone().detach().cpu().squeeze()
            # image_to_log = image_tensor.permute(1, 2, 0).clamp(max=1, min=0)
            if writer:
                img = generate_tensor.clone().squeeze().cpu()
                img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean,
                                0, 1)

                writer.add_image('images', img.permute(2, 0, 1), i)
                # writer.add_graph(model, images)

        optimizer.step(closure)

    # if writer:
        # writer.close()

    generate_tensor.requires_grad = False
    return generate_tensor, saved_tensor_lists
