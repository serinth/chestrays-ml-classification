import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
import shutil
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

# Parameters
learning_rate = 0.01
num_steps = 100
#num_steps = 2000
batch_size = 10
display_step = 100

# Network Parameters
dropout = 0.5 # Dropout, probability to keep units

# Images
IMG_HEIGHT = 150
IMG_WIDTH = 150
CH = 3
image_dir = "M:\\DataSets\\chestrays\\source\\" # XPS
rows = 4001 # number 
train_rows = 3600 # 90/10 split
test_rows = 400


df = pd.read_csv("chestrays.csv", header=None, na_values="?")
df = df.iloc[1:rows]
df.head()


# Prepare train and test sets

# Factorize the labels and make the directories, convert all | to _'s, remove spaces
labels, names = pd.factorize(df[1])
image_names = image_dir + df.iloc[0:rows,0].values
d = dict() # dictionary of classification -> count pairs

# data mover function, also populates the dictionary so we can see the distribution of data
def copyImages(dataframe, idx, directory="train"):
    classification = dataframe.iloc[idx][1].replace(" ","").replace("|","_")
    
    if classification in d:
        d[classification] += 1
    else:
        d[classification] = 1
        
    source = image_dir + dataframe.iloc[idx][0]
    destination = directory + "/" + classification
    shutil.copy(source, destination)

# Make train and test directories, replaces spaces and |'s with _
for n in names:
    dirname = n.replace(" ","").replace("|","_")
    pathlib.Path("train/" + dirname).mkdir(parents=True, exist_ok=True)
    pathlib.Path("test/" + dirname).mkdir(parents=True, exist_ok=True)


for r in range(train_rows):
    copyImages(df, r, "train")

for r in range(test_rows):
    copyImages(df, train_rows + r, "test")


num_classes = len(list(set(labels)))

print('Number of classes: {}'.format(num_classes))
print('Number of rows: {}'.format(len(labels))) 
print(names[:10])
print(image_names)


# Build the TF model
model = Sequential()
# input: 250x250 images with 1 channel
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, CH)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of './train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'train',  # this is the target directory
        target_size=(IMG_WIDTH, IMG_HEIGHT),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical')


model.fit_generator(
        train_generator,
        steps_per_epoch=num_steps // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

