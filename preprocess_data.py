from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_folder = "/home/indra/Documents/XRVision/imgcls/dataset/"
train_dir = image_folder + "train/"
val_dir = image_folder + "val/" 

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        classes = ['faulty', 'not_faulty'],
        target_size=(55, 55),  # All images will be resized to 55x55
        batch_size=32,
        # Use binary labels
        class_mode='binary')

# Flow validation images in batches of 32 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        val_dir,  # This is the source directory for training images
        classes = ['faulty', 'not_faulty'],
        target_size=(55, 55),  # All images will be resized to 55x55
        batch_size=32,
        # Use binary labels
        class_mode='binary',
        shuffle=False)