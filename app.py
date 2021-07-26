"""Streamlit app."""
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from src.nst import VGG_Activation, transfer


def image_uploaders():
    """Image uploader section."""
    style_img = None
    content_img = None

    col1, col2 = st.beta_columns(2)

    # col1.subheader('Style Image')
    style_upload = col1.file_uploader('Choose Style Image',
                                      type=['png', 'jpg'],
                                      )
    if style_upload is not None:
        # To read file as bytes:
        # style_img = style_upload.getvalue()
        style_img = Image.open(style_upload)
        col1.image(style_img, caption="Style Image")

    # col2.subheader('Content Image')
    content_upload = col2.file_uploader("Choose Content Image",
                                        type=['png', 'jpg'],
                                        )
    if content_upload is not None:
        # To read file as bytes:
        # content_image = content_upload.getvalue()
        content_img = Image.open(content_upload)
        col2.image(content_img, caption="Content Image")

    return style_img, content_img


@st.cache
def load_model(device):
    """Load the feature extractor model."""
    layer_to_watch = [0, 5, 10, 19, 28]  # layers to get the activation from
    vgg_layer_act = VGG_Activation(layer_to_watch).to(device)
    return vgg_layer_act


if __name__ == "__main__":

    available_device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"{available_device} is available")
    device = torch.device(device=available_device)

    vgg_layer_act = load_model(device)

    st.title('Neural Style Transfer')

    style_img, content_img = image_uploaders()

    # Sidebar
    optim_ = st.sidebar.selectbox(
        "Select an optimizer",
        ("Adam", "LGBFS"),
        index=1
    )

    alpha = st.sidebar.number_input('Alpha', value=1.0, format='%f')
    beta = st.sidebar.number_input('Beta', value=0.1, format='%f')
    lr = st.sidebar.number_input('Learning Rate',
                                 value=0.1,
                                 format="%f")
    iteration = st.sidebar.number_input('Iteration',
                                        min_value=100,
                                        value=600,
                                        step=10)

    if st.sidebar.button('Style the image'):
        st.write('Styling image with : ', optim_, alpha, beta, lr, iteration)
        if style_img is None or content_img is None:
            st.error('Images are missing')
            # print(
        else:
            with st.spinner(text='Styling the Image...'):
                my_bar = st.progress(0.0)

                transformImage2Tensor = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])

                transformTensor2Image = transforms.ToPILImage()

                content_tensor = (transformImage2Tensor(content_img)
                                  .unsqueeze(0).to(device))
                style_tensor = (transformImage2Tensor(style_img)
                                .unsqueeze(0).to(device))
                generate_tensor = content_tensor.clone()

                # layers to get the activation from
                layer_to_watch = [0, 5, 10, 19, 28]

                out_image, generation_lists = transfer(vgg_layer_act,
                                                       generate_tensor,
                                                       content_tensor,
                                                       style_tensor,
                                                       layer_to_watch,
                                                       optim_=optim_,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       lr=lr,
                                                       iteration=iteration,
                                                       device=device,
                                                       st_bar=my_bar)

                with torch.no_grad():
                    gen_image = (out_image.cpu().squeeze()
                                 .clamp(max=1, min=0))

                    gen_image = transformTensor2Image(gen_image)

                    st.success('Done')

            col_im_1, col_im_2, col_im_3 = st.beta_columns(3)
            col_im_2.image(gen_image, caption="Generated Image")
