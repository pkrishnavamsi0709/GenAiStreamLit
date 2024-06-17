import IPython
import sys
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import streamlit as st
import numpy as np
from PIL import Image
import io

# Define project information
PROJECT_ID = "serious-water-423304-j5"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location=LOCATION)


generation_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
# prompt = "A man sitting on the bench with his dog in a garden"
# prompt = "A ice angle coming to earth from heaven"
# prompt = "Indian freedom fighters group photo"
prompt = "A man wwaiting for his flight in airport"

response = generation_model.generate_images(
    prompt=prompt,
)
st.write(response)
if(response.images.__len__()):
    generated_image = response.images[0]

    # converts it into a bytes
    image_data_bytes = generated_image._loaded_bytes

    # function from NumPy converts the pixel data from the Pillow image object into a NumPy array,
    #  making it suitable for display within your Streamlit application.
    image_data = np.array(Image.open(io.BytesIO(image_data_bytes)))

    st.image(image_data, caption="Generated Image", use_column_width=True)
else:
    st.write("can't load the content")

# with open("generated_image.png", "wb") as f:
#     f.write(response.images[0])


# image = Image.fromarray(response)
# image.save("generated_image.png")


# print(response.images[0])

# response.images[0].show()
# st.image(response.images[0])
# st.image(image_data, caption='Generated Image', use_column_width=True)

# Assuming you have the generated image data (e.g., NumPy array or byte array) in a variable named `image_data`

# from io import BytesIO

# image_buffer = BytesIO(response.images[0])
# st.image(image_buffer, caption="Generated Image", use_column_width=True)


# try:
#   image = Image.open(image_buffer)
#   st.image(image, caption="Generated Image", use_column_width=True)

# except Exception as e:
#   print(f"Error opening image: {e}")
