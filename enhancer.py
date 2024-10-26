# import streamlit as st
# from PIL import Image
# import torch
# from torchvision.utils import save_image
# import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import os
# import io
# from streamlit_cropper import st_cropper

# # Import the model class from your model file
# from model import Generator

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the trained model
# gen = Generator().to(device)
# checkpoint_path = "esrgan_checkpoint_epoch_109.pth"  # Path to your checkpoint
# checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
# gen.load_state_dict(checkpoint["state_dict"])
# gen.eval()

# # Function to process and upscale the image
# def process_image(image):
#     if isinstance(image, Image.Image):
#         pil_image = image
#     else:
#         pil_image = Image.open(image).convert("RGB")
    
#     original_width, original_height = pil_image.size
#     low_res_width, low_res_height = original_width // 4, original_height // 4

#     transform = A.Compose(
#         [
#             A.Resize(
#                 width=low_res_width,
#                 height=low_res_height,
#                 interpolation=Image.BICUBIC,
#             ),
#             A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
#             ToTensorV2(),
#         ]
#     )

#     low_res_image = transform(image=np.array(pil_image))["image"].unsqueeze(0).to(device)

#     with torch.no_grad():
#         high_res_fake = gen(low_res_image)

#     high_res_fake = high_res_fake.squeeze(0).cpu()

#     buffer = io.BytesIO()
#     save_image(high_res_fake, buffer, format="PNG")
#     buffer.seek(0)

#     return buffer, original_width, original_height

# def image_enhancer():
#     st.markdown(
#         """
#         <h1 style="text-align: center;">Galaxy Image Resolution Enhancer</h1>
#         """,
#         unsafe_allow_html=True,
#     )

#     st.markdown(
#         """
#         <div style="text-align: center; font-weight: bold; font-size: 18px;">
#             Leverage neural networks such as Generative Adversarial Networks (GANs) to upscale low-resolution galaxy images. This system enhances image clarity and detail, making it easier to identify faint astronomical objects and analyze galaxy structures with greater precision. <br>.
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     # Create four columns for the description
#     col1, col2, col3, col4 = st.columns(4)

#     with col2:
#         with st.container(border=True, height=300):
#             st.markdown(
#                 """
#             <div style="text-align: center; font-weight: bold;">
#                 <p style="font-size:20px"><b>Before</b></p>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )
#             st.image(
#                 "before.jpg",
#                 use_column_width="auto",
#             )

#     with col3:
#         with st.container(border=True, height=300):
#             st.markdown(
#                 """
#             <div style="text-align: center; font-weight: bold;">
#                 <p style="font-size:20px"><b>After</b></p>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )
#             st.image(
#                 "after.jpg",
#                 use_column_width="auto",
#             )

#     # File uploader
#     uploaded_file = st.file_uploader(
#         "Choose a galaxy image...", type=["jpg", "png", "jpeg"]
#     )

#     if uploaded_file is not None:
#         # Open the original image
#         original_image = Image.open(uploaded_file)
#         original_width, original_height = original_image.size

#         # Display enhancement options
#         enhancement_option = st.radio(
#             "Choose Enhancement Option:",
#             ["Full Image Enhancement", "Crop and Enhance"],
#             horizontal=True
#         )

#         # Display the appropriate section based on user choice
#         if enhancement_option == "Full Image Enhancement":
#             st.subheader("Full Image Enhancement")
#             with st.container(border=True):
#                 # Display the uploaded image in its true size
#                 st.image(
#                     original_image,
#                     caption=f"Original Image ({original_width}x{original_height})",
#                     use_column_width=False,  # Changed to False to maintain true size
#                     width=original_width     # Explicitly set the width to maintain aspect ratio
#                 )

#                 if st.button("Upscale Full Image", type="primary"):
#                     with st.spinner("Enhancing image..."):
#                         # Process the image to enhance resolution
#                         buffer, original_width, original_height = process_image(uploaded_file)

#                         # Display enhanced image
#                         enhanced_image = Image.open(buffer)
#                         enhanced_width, enhanced_height = enhanced_image.size

#                         st.image(
#                             enhanced_image,
#                             caption=f"Enhanced Image ({enhanced_width}x{enhanced_height})",
#                             use_column_width=False,  # Changed to False to maintain true size
#                             width=enhanced_width     # Explicitly set the width to maintain aspect ratio
#                         )

#                         # Provide download button for enhanced image
#                         st.download_button(
#                             label="Download Enhanced Image",
#                             data=buffer,
#                             file_name="enhanced_image.png",
#                             mime="image/png",
#                         )

#         else:  # Crop and Enhance option
#             st.subheader("Crop and Enhance")
#             with st.container(border=True):
#                 st.write("Select the region you want to enhance:")
#                 # Get cropped image from the cropper
#                 cropped_img = st_cropper(
#                     original_image,
#                     realtime_update=True,
#                     box_color='#0000FF',
#                     aspect_ratio=None
#                 )
                
#                 if st.button("Enhance Cropped Region", type="primary"):
#                     with st.spinner("Enhancing cropped region..."):
#                         # Process the cropped image
#                         buffer, crop_width, crop_height = process_image(cropped_img)
                        
#                         # Display enhanced cropped image
#                         enhanced_crop = Image.open(buffer)
#                         enhanced_width, enhanced_height = enhanced_crop.size
                        
#                         st.image(
#                             enhanced_crop,
#                             caption=f"Enhanced Cropped Image ({enhanced_width}x{enhanced_height})",
#                             use_column_width=False,  # Changed to False to maintain true size
#                             width=enhanced_width     # Explicitly set the width to maintain aspect ratio
#                         )
                        
#                         # Provide download button for enhanced cropped image
#                         st.download_button(
#                             label="Download Enhanced Cropped Image",
#                             data=buffer,
#                             file_name="enhanced_cropped_image.png",
#                             mime="image/png",
#                         )

# if __name__ == "__main__":
#     image_enhancer()


import streamlit as st
from PIL import Image
import torch
from torchvision.utils import save_image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import io
from streamlit_cropper import st_cropper

# Import the model class from your model file
from model import Generator

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
gen = Generator().to(device)
checkpoint_path = "esrgan_checkpoint_epoch_109.pth"  # Path to your checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
gen.load_state_dict(checkpoint["state_dict"])
gen.eval()

# Function to process and upscale the image
def process_image(image):
    if isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.open(image).convert("RGB")
    
    original_width, original_height = pil_image.size
    low_res_width, low_res_height = original_width // 1, original_height // 1

    transform = A.Compose(
        [
            A.Resize(
                width=low_res_width,
                height=low_res_height,
                interpolation=Image.BICUBIC,
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2(),
        ]
    )

    low_res_image = transform(image=np.array(pil_image))["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        high_res_fake = gen(low_res_image)

    high_res_fake = high_res_fake.squeeze(0).cpu()

    buffer = io.BytesIO()
    save_image(high_res_fake, buffer, format="PNG")
    buffer.seek(0)

    return buffer, original_width, original_height

def display_image_with_conditional_size(image, caption, max_width=700):
    """Helper function to display image with conditional sizing"""
    width = image.size[0]
    if width > max_width:
        st.image(
            image,
            caption=caption,
            use_column_width=True
        )
    else:
        st.image(
            image,
            caption=caption,
            use_column_width=False,
            width=width
        )

def image_enhancer():
    st.markdown(
        """
        <h1 style="text-align: center;">Galaxy Image Resolution Enhancer</h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align: center; font-weight: bold; font-size: 18px;">
            Leverage neural networks such as Generative Adversarial Networks (GANs) to upscale low-resolution galaxy images. This system enhances image clarity and detail, making it easier to identify faint astronomical objects and analyze galaxy structures with greater precision. <br>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create four columns for the description
    col1, col2, col3, col4 = st.columns(4)

    with col2:
        with st.container(border=True, height=300):
            st.markdown(
                """
            <div style="text-align: center; font-weight: bold;">
                <p style="font-size:20px"><b>Before</b></p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.image(
                "before.jpg",
                use_column_width="auto",
            )

    with col3:
        with st.container(border=True, height=400):
            st.markdown(
                """
            <div style="text-align: center; font-weight: bold;">
                <p style="font-size:20px"><b>After</b></p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.image(
                "after.jpg",
                use_column_width=200,
            )

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a galaxy image...", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        # Open the original image
        original_image = Image.open(uploaded_file)
        original_width, original_height = original_image.size

        # Display enhancement options
        enhancement_option = st.radio(
            "Choose Enhancement Option:",
            ["Full Image Enhancement", "Crop and Enhance"],
            horizontal=True
        )

        # Display the appropriate section based on user choice
        if enhancement_option == "Full Image Enhancement":
            st.subheader("Full Image Enhancement")
            with st.container(border=True):
                # Display the uploaded image with conditional sizing
                display_image_with_conditional_size(
                    original_image,
                    f"Original Image ({original_width}x{original_height})"
                )

                if st.button("Upscale Full Image", type="primary"):
                    with st.spinner("Enhancing image..."):
                        # Process the image to enhance resolution
                        buffer, original_width, original_height = process_image(uploaded_file)

                        # Display enhanced image
                        enhanced_image = Image.open(buffer)
                        enhanced_width, enhanced_height = enhanced_image.size
                        
                        # Display the enhanced image with conditional sizing
                        display_image_with_conditional_size(
                            enhanced_image,
                            f"Enhanced Image ({enhanced_width}x{enhanced_height})"
                        )

                        # Provide download button for enhanced image
                        st.download_button(
                            label="Download Enhanced Image",
                            data=buffer,
                            file_name="enhanced_image.png",
                            mime="image/png",
                        )

        else:  # Crop and Enhance option
            st.subheader("Crop and Enhance")
            with st.container(border=True):
                st.write("Select the region you want to enhance:")
                # Get cropped image from the cropper
                cropped_img = st_cropper(
                    original_image,
                    realtime_update=True,
                    box_color='#0000FF',
                    aspect_ratio=None
                )
                
                if st.button("Enhance Cropped Region", type="primary"):
                    with st.spinner("Enhancing cropped region..."):
                        # Process the cropped image
                        buffer, crop_width, crop_height = process_image(cropped_img)
                        
                        # Display enhanced cropped image
                        enhanced_crop = Image.open(buffer)
                        enhanced_width, enhanced_height = enhanced_crop.size
                        
                        # Display the enhanced cropped image with conditional sizing
                        display_image_with_conditional_size(
                            enhanced_crop,
                            f"Enhanced Cropped Image ({enhanced_width}x{enhanced_height})"
                        )
                        
                        # Provide download button for enhanced cropped image
                        st.download_button(
                            label="Download Enhanced Cropped Image",
                            data=buffer,
                            file_name="enhanced_cropped_image.png",
                            mime="image/png",
                        )

if __name__ == "__main__":
    image_enhancer()