'''
import torch
from collections.abc import Callable
from monai.metrics import PanopticQualityMetric
from monai.utils import MetricReduction
from monai.handlers.ignite_metric import IgniteMetricHandler
import nibabel as nib
import numpy as np


#PQ: PANOPTIC QUALITY

#y_pred

type_map_path = '/home/rmartin/projects/Deep_INGENIO/OUTPUT_5/Ingenio_RYC_00005_HE [x=37575,y=28718,w=540,h=540]_type_map.nii.gz'
instance_map_path = '/home/rmartin/projects/Deep_INGENIO/OUTPUT_5/Ingenio_RYC_00005_HE [x=37575,y=28718,w=540,h=540]_instance_map.nii.gz'

type_map_img = nib.load(type_map_path)
instance_map_img = nib.load(instance_map_path)
#print (type_map_img)

type_map_data = type_map_img.get_fdata()
instance_map_data = instance_map_img.get_fdata()

type_map_data = torch.tensor(type_map_data, dtype=torch.int).unsqueeze(0)  # Shape: (1, height, width)
instance_map_data = torch.tensor(instance_map_data, dtype=torch.int).unsqueeze(0)  # Shape: (1, height, width)

y_pred = torch.stack([instance_map_data, type_map_data], dim=1)  # Shape: (1, 2, height, width)
#print(y_pred.shape)

#y

type_map_path_y = '/home/rmartin/projects/Deep_INGENIO/inference_prueba/pf/Ingenio_RYC_00005_HE [x=37575,y=28718,w=540,h=540]_type_map.npy'
instance_map_path_y = '/home/rmartin/projects/Deep_INGENIO/inference_prueba/pf/Ingenio_RYC_00005_HE [x=37575,y=28718,w=540,h=540]_inst_map.npy'

type_map_data_y = np.load(type_map_path_y)
instance_map_data_y = np.load(instance_map_path_y)
#print(type_map_data_y.shape)
#print(instance_map_data_y.shape)
type_map_data_y_ = type_map_data_y.squeeze()
instance_map_data_y_ = instance_map_data_y.squeeze()
#print(type_map_data_y_.shape)
#print(instance_map_data_y_.shape)

type_map_tensor_y = torch.tensor(type_map_data_y_, dtype=torch.int).unsqueeze(0) # Shape: (1, height, width)
instance_map_tensor_y = torch.tensor(instance_map_data_y_, dtype=torch.int).unsqueeze(0)  # Shape: (1, height, width)

y = torch.stack([instance_map_tensor_y, type_map_tensor_y], dim=1)  # Shape: (1, 2, height, width)
#print(y.shape)

#Métricas

pq = PanopticQualityMetric(num_classes=3, metric_name="pq") # Número de clases excluyendo el fondo
panoptic_quality_pq = pq._compute_tensor(y_pred, y)
print(panoptic_quality_pq) 


pq_values = []
for i in range(panoptic_quality_pq.size(1)):
    tp = panoptic_quality_pq[0, i, 0]
    fp = panoptic_quality_pq[0, i, 1]
    fn = panoptic_quality_pq[0, i, 2]
    iou = panoptic_quality_pq[0, i, 3]
    pq = iou / (tp + 0.5 * fp + 0.5 * fn)
    pq_values.append(pq)
    # `q.item()` is a method in PyTorch that is used to extract the scalar value
    # from a PyTorch tensor. When you call `q.item()`, it returns the Python number
    # stored in the tensor `q`. This is useful when you have a tensor that contains
    # only one element and you want to extract that element as a Python number for
    # further calculations or comparisons.
    # q.item()

# Imprimir los resultados
for i, pq in enumerate(pq_values):
    print(f"PQ para la fila {i+1}: {pq}")



sq = PanopticQualityMetric(num_classes=3, metric_name="sq") # Número de clases excluyendo el fondo
panoptic_quality_sq = sq._compute_tensor(y_pred, y)
print(panoptic_quality_sq)

rq = PanopticQualityMetric(num_classes=3, metric_name="rq") # Número de clases excluyendo el fondo
panoptic_quality_rq = rq._compute_tensor(y_pred, y)
print(panoptic_quality_rq)

'''


import os
import glob
import torch
import numpy as np
import nibabel as nib
from monai.metrics import PanopticQualityMetric
import numpy as np

def find_matching_files(nii_directory, npy_directory):
    """
    Encuentra archivos en dos directorios que tienen el mismo nombre antes de la extensión.
    """
    nii_type_files = sorted(glob.glob(os.path.join(nii_directory, "*_image_type_map.nii.gz")))
    nii_instance_files = sorted(glob.glob(os.path.join(nii_directory, "*_image_instance_map.nii.gz")))
    npy_type_files = sorted(glob.glob(os.path.join(npy_directory, "*_type_map.npy")))
    npy_instance_files = sorted(glob.glob(os.path.join(npy_directory, "*_inst_map.npy")))

    matched_files = []
    nii_basenames = [os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0].replace('_image_type_map', '') for f in nii_type_files]
    npy_basenames = [os.path.splitext(os.path.basename(f))[0].replace('_type_map', '') for f in npy_type_files]

    for nii_base in nii_basenames:
        nii_type_path = os.path.join(nii_directory, nii_base + '_image_type_map.nii.gz')
        nii_instance_path = os.path.join(nii_directory, nii_base + '_image_instance_map.nii.gz')
        npy_type_path = os.path.join(npy_directory, nii_base + '_type_map.npy')
        npy_instance_path = os.path.join(npy_directory, nii_base + '_inst_map.npy')
        if nii_base in npy_basenames and os.path.exists(nii_type_path) and os.path.exists(nii_instance_path):
            matched_files.append((nii_type_path, nii_instance_path, npy_type_path, npy_instance_path))

    return matched_files

def load_nifti_data(nii_path):
    nii_img = nib.load(nii_path)
    return nii_img.get_fdata()

def load_npy_data(npy_path):
    return np.load(npy_path)

def calculate_panoptic_quality(nii_directory, npy_directory, num_classes):
    matched_files = find_matching_files(nii_directory, npy_directory)
    all_pq_tensors = []

    pq_metric = PanopticQualityMetric(num_classes=num_classes, metric_name="pq")

    for nii_type_path, nii_instance_path, npy_type_path, npy_instance_path in matched_files:
        # Cargar datos de predicción
        y_pred_type = load_nifti_data(nii_type_path)
        y_pred_instance = load_nifti_data(nii_instance_path)
        y_pred_type = torch.tensor(y_pred_type, dtype=torch.int).unsqueeze(0)
        y_pred_instance = torch.tensor(y_pred_instance, dtype=torch.int).unsqueeze(0)
        y_pred = torch.stack([y_pred_instance, y_pred_type], dim=1)

        # Cargar datos de verdad 
        y_type_ = load_npy_data(npy_type_path)
        y_instance_ = load_npy_data(npy_instance_path)
        y_type = y_type_.squeeze()
        y_instance = y_instance_.squeeze()
        y_type = torch.tensor(y_type, dtype=torch.int).unsqueeze(0)
        y_instance = torch.tensor(y_instance, dtype=torch.int).unsqueeze(0)
        y = torch.stack([y_instance, y_type], dim=1)

        # Calcular la calidad panóptica
        panoptic_quality_pq = pq_metric._compute_tensor(y_pred, y)
        all_pq_tensors.append(panoptic_quality_pq)

    return all_pq_tensors

# Directorios
nii_directory = '/home/rmartin/projects/Deep_INGENIO/InferenceValidexp1'
npy_directory = '/home/rmartin/projects/Deep_INGENIO/destino/Valid'

# Número de clases (excluyendo el fondo)
num_classes = 3

# Calcular la calidad panóptica
pq_tensors = calculate_panoptic_quality(nii_directory, npy_directory, num_classes)

# Imprimir resultados
#for i, pq_tensor in enumerate(pq_tensors):
#   print(f"Tensor panoptic_quality_pq {i}: {pq_tensor}")

all_pq_values = []  # Lista para almacenar los valores de PQ de todos los tensores

for pq_tensor in pq_tensors:
    pq_values = []  # Lista para almacenar los valores de PQ de un tensor específico
    for i in range(pq_tensor.size(1)):
        tp = pq_tensor[0, i, 0].item()
        fp = pq_tensor[0, i, 1].item()
        fn = pq_tensor[0, i, 2].item()
        iou = pq_tensor[0, i, 3].item()
        if (tp == 0 and fp == 0 and fn == 0): 
            pq = "nan"
        else: pq = iou / (tp + 0.5 * fp + 0.5 * fn)
        pq_values.append(pq)
    all_pq_values.append(pq_values)

#print(all_pq_values)

all_precision_values = []

for pq_tensor in pq_tensors:
    precision_values = []  # Lista para almacenar los valores de PQ de un tensor específico
    for i in range(pq_tensor.size(1)):
        tp = pq_tensor[0, i, 0].item()
        fp = pq_tensor[0, i, 1].item()
        fn = pq_tensor[0, i, 2].item()
        iou = pq_tensor[0, i, 3].item()
        if (tp == 0 and fp == 0): 
            precision = "nan"
        else: precision = tp / (tp + fp)
        precision_values.append(precision)
    all_precision_values.append(precision_values)
    
#print(all_precision_values)


#PQ medio

all_pq_values = [[np.nan if val == 'nan' else val for val in sublist] for sublist in all_pq_values] #Reemplazar nan con np.nan
all_pq_values_array =np.array(all_pq_values, dtype=np.float64)
mean_pq = np.nanmean(all_pq_values_array, axis=0)
print(mean_pq)

#Precision media

all_precision_values = [[np.nan if val == 'nan' else val for val in sublist] for sublist in all_precision_values] #Reemplazar nan con np.nan
all_precision_values_array =np.array(all_precision_values, dtype=np.float64)
mean_pre = np.nanmean(all_precision_values_array, axis=0)
print(mean_pre)




'''
sums = [0.0, 0.0, 0.0]
counts = [0, 0, 0]

for sublist in all_pq_values:
    for i in range(3):
        if sublist[i] != 'nan':
            sums[i] += float(sublist[i])
            counts[i] += 1

means = [sums[i] / counts[i] if counts[i] != 0 else 'nan' for i in range(3)]

print (means) #[3.2362234436804584e-05, 0.00018365992144913121, 7.386031383200415e-05]


'''