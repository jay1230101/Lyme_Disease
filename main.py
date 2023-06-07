import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import keras
import tensorflow
import tensorflow as tf
from keras.utils import img_to_array, load_img
from keras.models import load_model
from streamlit_option_menu import option_menu
import hydralit_components as hc
import datetime
import streamlit.components.v1 as components
import time
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout='wide',initial_sidebar_state='collapsed',)

menu_data =[
    {'label':'Overview'},
    {'icon':'fa-light fa-hand-dots','label':'Lyme Negative'},
    {'icon':'fa-light fa-face-eyes-xmarks','label':'Lyme Positive'},
    {'label':'Data Generator'},
    {'label':'Camera Predict'},
    {'label':'Upload Predict'},
    {'label':'Metrics'}
]
over_theme = {'txc_inactive':'white','menu_background':'orange'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    hide_streamlit_markers=False,
    sticky_nav=True,
    sticky_mode='pinned'
)
main_image = 'bullseye rash1.jpg'
Lyme_Negative =['pityriasis rosea378.jpg','serum sickness.jpg','shingles_rash.jpg','spider bite rash3.jpg']
Lyme_Positive = ['bullseye rash11.jpg','EM rash8.jpg','Erythema Migrans45.jpg','Erythema Migrans476.jpg']

if menu_id=='Overview':
    st.write("<div align='center'><h2>Binary Classification Model : Lyme Disease Positive or Negative </h2></div>",unsafe_allow_html=True)
    st.write("<h5> Lyme Disease known as the 'Silent Epidemic' affects more than 300,000 people each year. "
             "It is transmitted to humans through the bite of infected blacklegged ticks </h5>",unsafe_allow_html=True)
    st.write("The dataset was scraped from the internet by @credits to Edward Zhang and it consists of 634 images")
    st.write("The model architecture and webapp were developed to the best of my knowledge. i hope you like it ❤️. 'Johny El Achkar'.")
    st.write("")
    col1,col2,col3=st.columns(3)
    with col1:
        ""
    with col2:
        image = Image.open(main_image)
        st.image(image)
    with col3:
        ""

width=200
height=300
if menu_id =='Lyme Negative':
    st.write("<div align='center'> <h2>Sample Images for Negative Lyme Disease</h2></div>",unsafe_allow_html=True)
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        img=Image.open(Lyme_Negative[0])
        img = ImageOps.fit(img, (width, height), Image.Resampling.BICUBIC)
        st.image(img,caption='Pityriasis Rosea',use_column_width=False,output_format='auto')

    with col2:
        img = Image.open(Lyme_Negative[1])
        img = ImageOps.fit(img, (width, height), Image.Resampling.BICUBIC)
        st.image(img,caption='Serum Sickness',use_column_width=False,output_format='auto')

    with col3:
        img = Image.open(Lyme_Negative[2])
        img = ImageOps.fit(img, (width, height), Image.Resampling.BICUBIC)
        st.image(img, caption='Shingle Rash', use_column_width=False, output_format='auto')

    with col4:
        img = Image.open(Lyme_Negative[3])
        img = ImageOps.fit(img, (width, height), Image.Resampling.BICUBIC)
        st.image(img, caption='Spider Bite', use_column_width=False, output_format='auto')



if menu_id=='Lyme Positive':
    st.write("<div align='center'><h2>Sample Images for Positive Lyme Disease</h2></div>",unsafe_allow_html=True)
    col1,col2,col3,col4=st.columns(4)
    with col1:
        img = Image.open(Lyme_Positive[0])
        img = ImageOps.fit(img,(width,height),Image.Resampling.BICUBIC)
        st.image(img,caption='Bulls Eye Rash',use_column_width=False,output_format='auto')

    with col2:
        img = Image.open(Lyme_Positive[1])
        img = ImageOps.fit(img,(width,height),Image.Resampling.BICUBIC)
        st.image(img,caption='Erythema Migrans Rash',use_column_width=False,output_format='auto')

    with col3:
        img = Image.open(Lyme_Positive[2])
        img = ImageOps.fit(img, (width, height), Image.Resampling.BICUBIC)
        st.image(img, caption='Erythema Migrans Rash', use_column_width=False, output_format='auto')

    with col4:
        img = Image.open(Lyme_Positive[3])
        img = ImageOps.fit(img, (width, height), Image.Resampling.BICUBIC)
        st.image(img, caption='Erythema Migrans Rash', use_column_width=False, output_format='auto')

train_path = 'imagedatagenerator'
if menu_id=='Data Generator':
    st.write("<div align='center'><h2> Image Data Generator</h2></div>",unsafe_allow_html=True)
    st.write(
             "The ImageDataGenerator class offers various image augmentation techniques, such as rotation, shifting, "
             "zooming, shearing, flipping, and adjusting brightness, among others. These techniques help increase "
             "the diversity and variability of the training data, "
             "which can improve the model's generalization and robustness.")
    # st.write("The goal from using Image Data Generator is to train the model on manipulated images")
    st.write("This is an example of how our images will look like after Image Data Generation")


    train_gen =tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                               horizontal_flip=True,
                                                               vertical_flip=True,
                                                               rotation_range=50,
                                                               height_shift_range=0.2,
                                                               width_shift_range=0.3)
    train_ds = train_gen.flow_from_directory(
        directory=train_path,
        target_size=(height,width),
        class_mode='binary'
    )
    class_names=['Negative_Lyme','Positive_Lyme']
    for batch in train_ds:
        images = batch[0]
        labels = batch[1]

        col1,col2,col3,col4= st.columns(4)

        with col1:
            st.image(images[0],caption=f"{class_names[int(labels[0])]}")

        with col2:
            st.image(images[1],caption=f"{class_names[int(labels[1])]}")

        with col3:
            st.image(images[2],caption=f"{class_names[int(labels[2])]}")

        with col4:
            st.image(images[3],caption=f"{class_names[int(labels[3])]}")
            break

if menu_id == 'Camera Predict':
    st.write(
        "<div align='center'><h2> Take a photo <i class='fas fa-camera fa-3x'></i> of your rash: check if it's Lyme Disease or Not?</h2></div>",
        unsafe_allow_html=True)

    # Display the camera icon
    components.html("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
        <div style="display: flex; justify-content: center;">
            <i class="fas fa-camera fa-5x"></i>
        </div>
    """)



    # uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png', 'gif'])

    img_file_buffer = st.camera_input("Take a picture first then click Predict")
    model = load_model('lyme_disease.h5')
    if st.button('Predict'):
        if img_file_buffer is not None:
            img = load_img(img_file_buffer, target_size=(256, 256))
            img = img_to_array(img)
            img = np.expand_dims(img,axis=0)
            img = img/255.0
            pred=model.predict(img)
            if pred >0.5:
                st.write(f"The Model predicts with 70% accuracy that the image shows Positive Lyme disease")
            else:
                st.write(f"The Model perdicts with 70% accuracy that the images shows No Signs of Lyme disease")


if menu_id=='Upload Predict':
    model = load_model('lyme_disease.h5')
    image1='image1_positive.jpg'
    image2= 'image2_positive.jpg'
    image3 = 'image3_negative.jpg'
    image4 = 'image4_negative.jpg'
    class_names = ['Negative','Positive']
    uploaded_file = st.file_uploader("", type=['jpg','jpeg','png','gif'])

    if st.button('Predict'):
        if uploaded_file is not None:
            img = load_img(uploaded_file,target_size=(256,256))
            img = img_to_array(img)
            img = np.expand_dims(img,axis=0)
            img = img/255.0
            pred=model.predict(img)
            if pred >0.5:
                st.write(f"The Model predicts with 70% accuracy that the image shows Positive Lyme disease")
            else:
                st.write(pred)
                st.write(f"The Model perdicts with 70% accuracy that the images shows No Signs of Lyme disease")


    col1,col2,col3,col4=st.columns(4)
    with col1:
        img = Image.open(image1)
        img=img.resize((200,200))
        st.image(img,caption='Positive Lyme')

    with col2:
        img=Image.open(image2)
        img = img.resize((200,200))
        st.image(img,caption='Positive Lyme')

    with col3:
        img=Image.open(image3)
        img = img.resize((200,200))
        st.image(img,caption='Negative Lyme')

    with col4:
        img = Image.open(image4)
        img = img.resize((200,200))
        st.image(img,caption='Negative Lyme')

if menu_id=='Metrics':
    st.write("<div align='center'><h2> Accuracy & Loss Line Chart</h2></div>",unsafe_allow_html=True)
    chart_data=pd.DataFrame(
        {
            'Loss':[0.8,0.9,0.68,0.75,0.60,0.40],
            'Val Loss':[0.85,0.85,0.70,0.68,0.60,0.51],
            'Accuracy':[0.40,0.60,0.48,0.75,0.78,0.85],
            'Val Accuracy':[0.38,0.58,0.32,0.70,0.68,0.75]
        }


    )
    line_colors = {
        'Loss': 'red',
        'Val Loss': 'blue',
        'Accuracy': 'green',
        'Val Accuracy': 'orange'
    }

    fig, ax = plt.subplots()

    for column, color in line_colors.items():
        ax.plot(chart_data[column], color=color, label=column)

    ax.legend(loc='lower right')

    # Save the figure to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # st.markdown(
    #     """
    #     <style>
    #     .centered-image-container {
    #         display: flex;
    #         justify-content: center;
    #         align-items: center;
    #         max-height: 300px;  /* Adjust the max-height as desired */
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )

    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(img_buffer, caption="Line Chart", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

