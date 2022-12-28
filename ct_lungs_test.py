import numpy as np
import streamlit as st
from PIL import Image
import requests
import json
from skimage import io

def load_image():
    uploaded_file = st.file_uploader(label='Загрузите файл изображения')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data, channels="BGR", output_format="PNG")
        return image_data
    else:
        return None

st.title('Распознавание инфекции в легких')
img = load_image()
result = st.button('Распознать изображение')

if result:
    text = 'OK'
    files = {'file': img}
    res = requests.get(f'http://127.0.0.1:80/prediction/{text}', files=files)
    arr = np.asarray(json.loads(res.json()))
    print(np.max(arr))
    io.imsave(f'test_mask.png', arr)
    image = Image.open('test_mask.png')
    st.image(image, output_format="PNG", clamp=False, channels="RGB")