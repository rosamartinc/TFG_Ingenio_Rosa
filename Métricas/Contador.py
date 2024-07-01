import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import label, find_objects
import shutil
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

def find_matching_files(directory):
    """
    Encuentra archivos en un directorio que tienen el mismo nombre antes de la extensión.
    """
    nii_type_files = sorted(glob.glob(os.path.join(directory, "*_type_map.nii.gz")))
    nii_instance_files = sorted(glob.glob(os.path.join(directory, "*_instance_map.nii.gz")))
    jpg_files = sorted(glob.glob(os.path.join(directory, "*.jpg")))
    
    matched_files = []
    nii_basenames = [os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0].replace('_type_map', '') for f in nii_type_files]
    jpg_basenames = [os.path.splitext(os.path.basename(f))[0] for f in jpg_files]
    
    for nii_base in nii_basenames:
        nii_type_path = os.path.join(directory, nii_base + '_type_map.nii.gz')
        nii_instance_path = os.path.join(directory, nii_base + '_instance_map.nii.gz')
        jpg_path = os.path.join(directory, nii_base + '.jpg')
        if nii_base in jpg_basenames and os.path.exists(nii_type_path) and os.path.exists(nii_instance_path):
            matched_files.append((nii_type_path, nii_instance_path, jpg_path, nii_base))
    
    return matched_files

def load_data_and_count_nuclei(nii_type_path, nii_instance_path):
    nii_img = nib.load(nii_type_path)
    img_data = nii_img.get_fdata()
    nii_img_i = nib.load(nii_instance_path)
    img_data_i = nii_img_i.get_fdata()

    # Procesar las imágenes para corregir la orientación
    rotated_img_data = np.rot90(img_data, k=3, axes=(0, 1))
    rotated_img_data_i = np.rot90(img_data_i, k=3, axes=(0, 1))
    mirrored_img_data = np.fliplr(rotated_img_data)
    mirrored_img_data_i = np.fliplr(rotated_img_data_i)

    unique_types = np.unique(mirrored_img_data)
    nucleo_count = {"Tumor": 0, "Normal": 0, "Stroma": 0}

    for nucleo_type in unique_types:
        if nucleo_type == 0:
            continue
        mask_type = (mirrored_img_data == nucleo_type)
        unique_nuclei_values = np.unique(mirrored_img_data_i[mask_type])
        if 0 in unique_nuclei_values:
            unique_nuclei_values = unique_nuclei_values[unique_nuclei_values != 0]
        nucleo_count_key = "Tumor" if nucleo_type == 1 else "Normal" if nucleo_type == 2 else "Stroma"
        nucleo_count[nucleo_count_key] += len(unique_nuclei_values)

    return nucleo_count, mirrored_img_data, mirrored_img_data_i

def save_files(base_name, nii_type_data, nii_instance_data, img_png, output_directory):
    """
    Guarda las imágenes procesadas en el directorio de salida especificado con los nombres adecuados.
    """
    os.makedirs(output_directory, exist_ok=True)

    # Crear y guardar la imagen de tipos de núcleos
    cmap_types = ListedColormap(['white', 'blue', 'black', 'green'])
    plt.imsave(os.path.join(output_directory, f"{base_name}_tipo.jpg"), nii_type_data, cmap=cmap_types)

    # Crear y guardar la imagen de instancias de núcleos
    plt.imsave(os.path.join(output_directory, f"{base_name}_ints.jpg"), nii_instance_data, cmap='gray')

    # Guardar la imagen JPG original
    shutil.copy(img_png, os.path.join(output_directory, f"{base_name}_img.jpg"))

# Directorio de los archivos
directory = '/home/rmartin/projects/Deep_INGENIO/OUTPUT_RING/RING21'
output_directory = '/home/rmartin/projects/Deep_INGENIO/OUTPUT_RING/RING21/SAVED_FILES'

# Obtener archivos que coincidan en el nombre antes de la extensión
matched_files = find_matching_files(directory)

all_counts = {"Tumor": 0, "Normal": 0, "Stroma": 0}

for nii_type_path, nii_instance_path, jpg_path, base_name in matched_files:
    nucleo_count, nii_type_data, nii_instance_data = load_data_and_count_nuclei(nii_type_path, nii_instance_path)
    all_counts["Tumor"] += nucleo_count.get("Tumor", 0)
    all_counts["Normal"] += nucleo_count.get("Normal", 0)
    all_counts["Stroma"] += nucleo_count.get("Stroma", 0)
    
    # Guardar los archivos en la carpeta de salida especificada con los nombres adecuados
    save_files(base_name, nii_type_data, nii_instance_data, jpg_path, output_directory)

# Mostrar resultados
print("Conteo total de núcleos por tipo:")
print(f"Tumor: {all_counts['Tumor']}")
print(f"Normal: {all_counts['Normal']}")
print(f"Stroma: {all_counts['Stroma']}")
