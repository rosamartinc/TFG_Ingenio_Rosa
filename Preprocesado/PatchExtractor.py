import os
from PIL import Image
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from skimage.color import label2rgb

# Define  paths

input_folder = '/home/rmartin/projects/INGENIO/data/parches_proyectos'
output_folder = '/home/rmartin/projects/INGENIO/data/parches_mascaras'

# Crear directorio de output, si no existe 

os.makedirs(output_folder, exist_ok=True)

#Recorrer directorio de entrada

for subdir in os.listdir(input_folder):
    subdir_path = os.path.join(input_folder, subdir)

    if os.path.isdir(subdir_path):  

        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)

            if filename.endswith('.jpg'):  

                #IMÁGENES

                open_img_jpg = Image.open(file_path)
                image = np.asarray(open_img_jpg)

                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_image.npy"
                output_path = os.path.join(output_folder, output_filename)
                np.save(output_path, image)

            elif filename.endswith('.png'):

                #MÁSCARAS

                open_img = Image.open(file_path)
                type_map = np.asarray(open_img)
                type_map = np.where(type_map == 2, 5, type_map)
                type_map = np.where(type_map == 4, 2, type_map)
                type_map = np.where(type_map == 5, 0, type_map)

                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_type_map.npy"
                output_path = os.path.join(output_folder, output_filename)
                np.save(output_path, np.expand_dims(type_map, axis=-1)) #Añadir la tervera dimensión con valor 1

                inst_map, num_features = label(type_map, None, None)
                inst_map = inst_map.astype("uint32")

                base_name_inst = os.path.splitext(filename)[0]
                output_filename_inst = f"{base_name}_inst_map.npy"
                output_path_inst = os.path.join(output_folder, output_filename_inst)
                np.save(output_path_inst, np.expand_dims(inst_map, axis=-1))

            #fig, axes = plt.subplots(1, 2)
            #axes[0].imshow(type_map, cmap = "gray")
            #axes[0].set_title('Máscara de células por tipos')
            #axes[1].imshow(label2rgb(inst_map, bg_label = 0))
            #axes[1].set_title('Máscara de células individuales')
            #plt.axis('off') 
            #plt.show()

print ("Done")