import os
import numpy as np
import streamlit as st
import keras.utils as image
from keras.models import load_model
from tempfile import NamedTemporaryFile
import PIL

st.set_option('deprecation.showfileUploaderEncoding', False)

def pil_image(i):
    imgf = PIL.Image.open(i)
    return imgf

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def loadImage(path):
    im = []
    img = image.load_img(path, target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    im.append(img)
    return np.array(im)

def label(pred):
    predicitions = []
    for i in pred:
        if i == 1:
            predicitions.append('Healthy')
        else:
            predicitions.append('Bad')
    return predicitions

if __name__ == '__main__':
    st.title('Leaf Health Classification Demo')

    file = st.file_uploader('Upload an image', type=['jpg', 'JPG'])
    temp_file = NamedTemporaryFile(delete=False)
    if file is not None:
        temp_file.write(file.getvalue())
        #st.write(temp_file.name)
        filename = temp_file.name
        test = loadImage(filename)
        mod = load_model('cnn_model.h5')
        p = label(np.argmax(mod.predict(test), axis=1))
        st.image(pil_image(file))
        st.write('This leaf in the image you selected is '+str(p[0]))
        
    else:
        st.write('No image selected')
