#comment
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import os
from PIL import Image
datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2) 
filename= './data/'
classes = os.listdir(filename)
for class_name in classes:
    class_path=os.path.join(filename, class_name)
    images_names = os.listdir(class_path)
    for image_name in images_names:

        image_file_dir = os.path.join(class_path, image_name)
        new_image_name= os.path.splitext(image_name)[0]
        im = Image.open(image_file_dir).resize((224, 224))

        x = img_to_array(im)
        # Reshape the input image 
        x = x.reshape((1, ) + x.shape)  
        i = 0

        # generate 5 new augmented images 
        for batch in datagen.flow(x, batch_size = 1, 
                        save_to_dir ='newdata'+'/'+class_name,save_prefix =new_image_name+str(i), save_format ='JPG'):
            i += 1
            if i > 4: 
                break
