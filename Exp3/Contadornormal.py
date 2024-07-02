import os
import glob
import numpy as np
import shutil
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def find_matching_files(directory):
    """
    Encuentra archivos en un directorio que tienen el mismo nombre antes de la extensión.
    """
    type_files = sorted(glob.glob(os.path.join(directory, "*_type_map.npy")))
    instance_files = sorted(glob.glob(os.path.join(directory, "*_inst_map.npy")))
    image_files = sorted(glob.glob(os.path.join(directory, "*_image.npy")))
    
    matched_files = []
    basenames = [os.path.splitext(os.path.basename(f))[0].replace('_type_map', '') for f in type_files]
    
    for base in basenames:
        type_path = os.path.join(directory, base + '_type_map.npy')
        instance_path = os.path.join(directory, base + '_inst_map.npy')
        image_path = os.path.join(directory, base + '_image.npy')
        if os.path.exists(type_path) and os.path.exists(instance_path) and os.path.exists(image_path):
            matched_files.append((type_path, instance_path, image_path, base))
    
    return matched_files

def load_data_and_count_nuclei(type_path, instance_path):
    type_data = np.load(type_path)
    instance_data = np.load(instance_path)

    unique_types = np.unique(type_data)
    nucleo_count = {"Tumor": 0, "Normal": 0, "Stroma": 0}

    for nucleo_type in unique_types:
        if nucleo_type == 0:
            continue
        mask_type = (type_data == nucleo_type)
        unique_nuclei_values = np.unique(instance_data[mask_type])
        if 0 in unique_nuclei_values:
            unique_nuclei_values = unique_nuclei_values[unique_nuclei_values != 0]
        nucleo_count_key = "Tumor" if nucleo_type == 1 else "Normal" if nucleo_type == 2 else "Stroma"
        nucleo_count[nucleo_count_key] += len(unique_nuclei_values)

    return nucleo_count, type_data, instance_data

# Directorio de los archivos
directory = '/home/rmartin/projects/Deep_INGENIO/destino/Train'

# Obtener archivos que coincidan en el nombre antes de la extensión
matched_files = find_matching_files(directory)

all_counts = {"Tumor": 0, "Normal": 0, "Stroma": 0}

for type_path, instance_path, image_path, base_name in matched_files:
    nucleo_count, type_data, instance_data = load_data_and_count_nuclei(type_path, instance_path)
    all_counts["Tumor"] += nucleo_count.get("Tumor", 0)
    all_counts["Normal"] += nucleo_count.get("Normal", 0)
    all_counts["Stroma"] += nucleo_count.get("Stroma", 0)
    
# Mostrar resultados
print("Conteo total de núcleos por tipo:")
print(f"Tumor: {all_counts['Tumor']}")
print(f"Normal: {all_counts['Normal']}")
print(f"Stroma: {all_counts['Stroma']}")