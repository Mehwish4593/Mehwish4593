#!/usr/bin/env python
# coding: utf-8

# # EDDense-Net

# In[1]:


#Required Libraries
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


#ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# In[3]:


training_set = train_datagen.flow_from_directory(
    'OneDrive - Higher Education Commission/Data/Drishti-GS1_files/Drishti-GS1_files/Drishti-GS1_files/Training',
    class_mode='input',color_mode='rgb',batch_size=32
)


# In[4]:


test_datagen = ImageDataGenerator(rescale=1./255) 


# In[5]:


test_set = test_datagen.flow_from_directory(
    'OneDrive - Higher Education Commission/Data/Drishti-GS1_files/Drishti-GS1_files/Drishti-GS1_files/Test',
    target_size=(640,640),
    class_mode='input', color_mode='rgb',batch_size=32
)


# In[6]:


cnn = tf.keras.models.Sequential()


# In[7]:


#Layer 1

cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size =2 ,  activation='relu', 
                               input_shape = [640,640,32],padding="valid"))
cnn.add(BatchNormalization())

cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size =2 ,  activation='relu', groups=4,
                               input_shape = [640,640,32],padding="valid"))


# In[8]:


#Layer 2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, input_shape = [320,320,32]))
cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size =2 ,  activation='relu', 
                               input_shape = [320,320,64],padding="valid"))
cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size =2 ,  activation='relu', groups=2,
                               input_shape = [320,320,64],padding="valid"))


# In[9]:


#Layer 3
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, input_shape = [160,160,64]))
cnn.add(tf.keras.layers.Conv2D(filters = 128, kernel_size =2 ,  activation='relu', 
                               input_shape = [160,160,128], padding="valid" ))
cnn.add(tf.keras.layers.Conv2D(filters = 128, kernel_size =2 ,  activation='relu', groups=4,
                               input_shape = [160,160,128], padding="valid"))


# In[10]:


#Layer 4
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, input_shape = [80,80,128]))
cnn.add(tf.keras.layers.Conv2D(filters = 128, kernel_size =2 ,  activation='relu', 
                               input_shape = [80,80,128],padding="valid"))
cnn.add(tf.keras.layers.Conv2D(filters = 128, kernel_size =2 ,  activation='relu', groups=2,
                               input_shape = [80,80,128],padding="valid" ))


# In[11]:


#Layer 5
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, input_shape = [40,40,128]))
cnn.add(tf.keras.layers.Conv2D(filters = 128, kernel_size =2 ,  activation='relu', 
                               input_shape = [40, 40,128],padding="valid"))
cnn.add(tf.keras.layers.Conv2D(filters = 128, kernel_size =2 ,  activation='relu', groups=2,
                               input_shape = [80,80,128],padding="valid" ))


# In[12]:


#Layer 6
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, input_shape = [20,20,128]))


# In[13]:


#Decoder
#layer 5
cnn.add(tf.keras.layers.UpSampling2D(input_shape = [40,40,128]))
cnn.add(tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size =2 ,  activation='relu', 
                               input_shape = [40, 40,128],padding="valid" ))
cnn.add(tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size =2 ,  activation='relu', groups=2,
                               input_shape = [40,40,256],padding="valid" ))


# In[14]:


#layer 4
cnn.add(tf.keras.layers.UpSampling2D(input_shape = [80,80,128]))
cnn.add(tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size =2 ,  activation='relu', 
                               input_shape = [80,80,128],padding="valid" ))
cnn.add(tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size =2 ,  activation='relu', groups=2,
                               input_shape = [80,80,128],padding="valid"))


# In[15]:


#layer 3
cnn.add(tf.keras.layers.UpSampling2D(input_shape = [160,160,128]))
cnn.add(tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size =2 ,  activation='relu', 
                               input_shape = [160,160,128],padding="valid" ))
cnn.add(tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size =2 ,  activation='relu', groups=2,
                               input_shape = [160,160,128], padding="valid" ))


# In[16]:


#layer 2
cnn.add(tf.keras.layers.UpSampling2D(input_shape = [320,320,64]))
cnn.add(tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size =2 ,  activation='relu', 
                               input_shape = [320,320,64],padding="valid"))
cnn.add(tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size =2 ,  activation='relu', groups=2,
                               input_shape = [320,320,64],padding="valid"))


# In[17]:


#layer 1
cnn.add(tf.keras.layers.UpSampling2D(input_shape = [640,640,32]))
cnn.add(tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size =2 ,  activation='relu', 
                               input_shape = [640,320,64] ,padding="valid"))

cnn.add(tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size =2 ,  activation='relu', groups=2,
                               input_shape = [640,320,64] ,padding="valid"))


# In[18]:


cnn.add(tf.keras.layers.Flatten())


# In[19]:


cnn.add(tf.keras.layers.Dense(units = 32, activation = 'relu'))


# In[20]:


cnn.add(tf.keras.layers.Dense(units =1, activation = 'sigmoid'))


# In[21]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[22]:


print(tf.one_hot(3, depth=1))
print(cnn.input.shape)
print(cnn.output.shape)


# In[23]:


cnn.build(input_shape = [640,640])


# In[24]:


cnn.summary()


# In[25]:


cnn.fit(x=training_set,validation_data = test_set, epochs = 10)


# In[ ]:





