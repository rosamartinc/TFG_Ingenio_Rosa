#https://github.com/Project-MONAI/tutorials/blob/main/pathology/hovernet/training.py
#https://github.com/Project-MONAI/tutorials/blob/main/pathology/hovernet/training.py
import torch
import os
import glob
import logging
import numpy as np
import torch.distributed as dist
from argparse import ArgumentParser
from monai.data import DataLoader, partition_dataset, CacheDataset, Dataset, PersistentDataset 
from monai.networks.nets import HoVerNet #https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/hovernet.py
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    TorchVisiond,
    Lambdad,
    Activationsd,
    OneOf,
    MedianSmoothd,
    AsDiscreted,
    Compose,
    CastToTyped,
    ComputeHoVerMapsd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandFlipd,
    RandAffined,
    RandGaussianSmoothd,
    CenterSpatialCropd,
)
from monai.handlers import (
    MeanDice,
    #PanopticQuality,
    CheckpointSaver,
    LrScheduleHandler,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.utils import set_determinism
from monai.utils.enums import HoVerNetBranch
from monai.apps.pathology.handlers.utils import from_engine_hovernet
from monai.apps.pathology.engines.utils import PrepareBatchHoVerNet
from monai.apps.pathology.losses import HoVerNetLoss
from skimage import measure

#Dado un diccionario que almacena directorio del registro y etapa de entrenamiento, nos devuelve el directorio del registro. Es crucial tener una buena organización de los registros y modelos para poder monitorear el progreso del entrenamiento
def create_log_dir(cfg): #cfg es un diccionario de configuración con los parámetros para el entrenamiento. Contiene directorio de registro y etapa de entrenamiento (entre otras cosas): cfg = { "log_dir": "path", "stage": 0, ...}
    log_dir = cfg["log_dir"] #se obtiene el directorio de un registro
    if cfg["stage"] == 0: #para la etapa 0, se añade un subdirectorio al diccionario de registros llamado stage0 
        log_dir = os.path.join(log_dir, "stage0")
    print(f"Logs and models are saved at '{log_dir}'.")
    if not os.path.exists(log_dir): #Si no existe el directorio log_dir de un registro, se crea
        os.makedirs(log_dir, exist_ok=True)
    return log_dir

#Preparación de una lista de datos para usar en el entrenamiento
def prepare_data(data_dir, phase): #data_dir es el directorio donde se encuentran los datos, y phase es la fase del proceso ("train", "test")
    data_dir = os.path.join(data_dir, phase) #se actualiza data_dir para que apunte al subdirectorio correspondiente a la fase especificada
    images = sorted(glob.glob(os.path.join(data_dir, "*image.npy"))) 
    inst_maps = sorted(glob.glob(os.path.join(data_dir, "*inst_map.npy")))
    type_maps = sorted(glob.glob(os.path.join(data_dir, "*type_map.npy")))
    #*image.npy: Archivos que contienen las imágenes. *inst_map.npy: Archivos que contienen los mapas de instancias. *type_map.npy: Archivos que contienen los mapas de tipos
    #glob es un módulo en Python que se utiliza para encontrar todas las rutas de archivos que coincidan con un patrón especificado 
    #sorted oredena alfabeticamente
    data_list = [ #Creación de la lista de diccionarios
        {"image": _image, "label_inst": _inst_map, "label_type": _type_map}
        for _image, _inst_map, _type_map in zip(images, inst_maps, type_maps)
    ]
    return data_list

#Devuelve DataLoaders dando cfg, transformzciónes en datos de entrenamiento y de validación
def get_loaders(cfg, train_transforms, val_transforms): #Devuelve DataLoaders dando cfg, transformzciónes en datos de entrenamiento y de validación
    multi_gpu = True if torch.cuda.device_count() > 1 else False #True if hay más de una gpu libre
   
    #A continuación las dos siguientes líneas obtienen las listas de datos de validación y de entrenamiento. cfg["root"] root es la ruta del directorio donde están los datos, devuleve dicho directorio
    train_data = prepare_data(cfg["root"], "Train")
    valid_data = prepare_data(cfg["root"], "Valid")
    if multi_gpu: #Condición: si hay multi_gpu true (más de una gpu libre) -> Partición del dataset en tantas partes como gpus disponibles tanto para entrenamiento como para validación
        train_data = partition_dataset(
            data=train_data,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
            shuffle=True,
            seed=cfg["seed"],
        )[dist.get_rank()] #Asigna a cada división del dataset una gpu
        valid_data = partition_dataset(
            data=valid_data,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
            shuffle=False,
            seed=cfg["seed"],
        )[dist.get_rank()]
    #¿En cuántas partes se ha dividido el dataset?:
    print("train_files:", len(train_data))
    print("val_files:", len(valid_data))
    #Creación de los conjuntos de datos: (CacheDataset es una clase que carga y transforma los datos, almacenándolos en caché para acelerar el acceso; cache_rate se refiere al % de datos a almacenar en caché siendo 1 el 100%; num_workers es el número de subprecesos para cargar los datos)
    #train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=0.12, num_workers=16)
    #valid_ds = CacheDataset(data=valid_data, transform=val_transforms, cache_rate=0.12, num_workers=16)
    train_ds = PersistentDataset(data=train_data, transform=train_transforms, cache_dir=None)
    valid_ds = PersistentDataset(data=valid_data, transform=val_transforms, cache_dir=None)
    #Creación de los cargadores (DataLoader)
    train_loader = DataLoader( #Creación del cargador (DataLoader)
        train_ds,
        batch_size=cfg["batch_size"], #tamaño de lote 
        num_workers=cfg["num_workers"], #número de subprocesos para cargar los datos
        shuffle=True, #True si los datos deben mezclarse aleatoriamente cada vez que se cargue un lote (solo para el conjunto de entrenamiento para que el modelo no se aprende el orden de los datos, no se aprende un patrón del orden de los datos)
        pin_memory=torch.cuda.is_available(), #Utilización de memoria fija basada en Cuda para transferencias de datos rápidas entre CPU y GPU: es una función que devuelve True si hay una GPU disponible y False si no la hay
    )
    val_loader = DataLoader(
        valid_ds, 
        batch_size=cfg["batch_size"], 
        num_workers=cfg["num_workers"], 
        shuffle=False, #Los datos se cargarán en el orden en que están almacenados en el dataset
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader

#Creación de modelo
def create_model(cfg, device): #device es el dispositivo CPU o GPU donde se ejecutará el modelo
    # Each user is responsible for checking the content of models/datasets and the applicable licenses and
    # determining if suitable for the intended use.
    # The license for the below pre-trained model is different than MONAI license.
    # Please check the source where these weights are obtained from:
    # https://github.com/vqdang/hover_net#data-format
    pretrained_model = "https://drive.google.com/u/1/uc?id=1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5&export=download"
    #pretrained_model = "https://drive.google.com/uc?id=1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR&export=download"
    #https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view
    if cfg["stage"] == 0:
        model = HoVerNet( #https://docs.monai.io/en/stable/networks.html
            mode=cfg["mode"], 
            in_channels=3, # número de canales de entrada; 3 corresponden a imagen RGB
            out_classes=cfg["out_classes"], #número de clases de salida para la segmentación: en este caso será tumor, stroma, normal
            act=("relu", {"inplace": True}),
            norm="batch",
            pretrained_url=pretrained_model, #solo se usa en la primera etapa
            freeze_encoder=True, #Indica que se congelarán las capas del encoder durante el entrenamiento. Los pesos y biases de las capas del encoder no se actualizarán en el entrenamiento. Encouder
        ).to(device)
        print(f'stage{cfg["stage"]} start!')
    else:
        model = HoVerNet( #Si no es la etapa cero se configura el modelo sin utilizar pesos preentrenados
            mode=cfg["mode"],
            in_channels=3,
            out_classes=cfg["out_classes"],
            act=("relu", {"inplace": True}),
            norm="batch",
            pretrained_url=None,
            freeze_encoder=False,
        ).to(device)
        model.load_state_dict(torch.load(cfg["ckpt"])["model"]) #carga los pesos desde un checkpoint
        print(f'stage{cfg["stage"]}, success load weight!')
    return model

#Configuración y ejecución del entrenamiento
def run(log_dir, cfg):
    set_determinism(seed=cfg["seed"])

    if cfg["mode"].lower() == "original": #https://github.com/vqdang/hover_net/blob/master/README.md
        cfg["patch_size"] = [270, 270]
        cfg["out_size"] = [80, 80]
    elif cfg["mode"].lower() == "fast":
        cfg["patch_size"] = [256, 256]
        cfg["out_size"] = [164, 164]

    multi_gpu = True if torch.cuda.device_count() > 1 else False
    if multi_gpu:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda:{}".format(dist.get_rank()))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if cfg["use_gpu"] else "cpu")

    # --------------------------------------------------------------------------
    # Data Loading and Preprocessing
    # --------------------------------------------------------------------------
    # __________________________________________________________________________
    # Build MONAI preprocessing
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label_inst", "label_type"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label_inst", "label_type"], channel_dim=-1), #Para garantizar compatibilidad: (3, 256, 256)
            Lambdad(keys="label_inst", func=lambda x: measure.label(x)), #Etiqueta instancias
            RandAffined( #Estas transformaciones aleatpria (rotación, escale, cizalla y traslación) introducen variabilidad en los datos de entrenamiento
                keys=["image", "label_inst", "label_type"],
                prob=1.0,
                rotate_range=((np.pi), 0),
                scale_range=((0.2), (0.2)),
                shear_range=((0.05), (0.05)),
                translate_range=((6), (6)),
                padding_mode="zeros",
                mode=("nearest"),
            ),
            CenterSpatialCropd( #recorta la imagen, crea parches 
                keys="image",
                roi_size=cfg["patch_size"],
            ),
            RandFlipd(keys=["image", "label_inst", "label_type"], prob=0.5, spatial_axis=0),  
            RandFlipd(keys=["image", "label_inst", "label_type"], prob=0.5, spatial_axis=1),
            OneOf( #elige una de las tres
                transforms=[
                    RandGaussianSmoothd(keys=["image"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=1.0),
                    MedianSmoothd(keys=["image"], radius=1),
                    RandGaussianNoised(keys=["image"], prob=1.0, std=0.05),
                ]
            ),
            CastToTyped(keys="image", dtype=np.uint8),
            TorchVisiond(
                keys=["image"],
                name="ColorJitter",
                brightness=(229 / 255.0, 281 / 255.0),
                contrast=(0.95, 1.10),
                saturation=(0.8, 1.2),
                hue=(-0.04, 0.04),
            ),
            #AsDiscreted(keys=["label_type"], to_onehot=[cfg["out_classes"]]),
            AsDiscreted(keys=["label_type"], to_onehot=[4]),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            CastToTyped(keys="label_inst", dtype=torch.int),
            ComputeHoVerMapsd(keys="label_inst"),
            Lambdad(keys="label_inst", func=lambda x: x > 0, overwrite="label"),
            CenterSpatialCropd(
                keys=["label", "hover_label_inst", "label_inst", "label_type"],
                roi_size=cfg["out_size"],
            ),
            AsDiscreted(keys=["label"], to_onehot=2),
            CastToTyped(keys=["image", "label_inst", "label_type"], dtype=torch.float32),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label_inst", "label_type"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label_inst", "label_type"], channel_dim=-1),
            Lambdad(keys="label_inst", func=lambda x: measure.label(x)),
            CastToTyped(keys=["image", "label_inst"], dtype=torch.int),
            CenterSpatialCropd(
                keys="image",
                roi_size=cfg["patch_size"],
            ),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            ComputeHoVerMapsd(keys="label_inst"),
            Lambdad(keys="label_inst", func=lambda x: x > 0, overwrite="label"),
            CenterSpatialCropd(
                keys=["label", "hover_label_inst", "label_inst", "label_type"],
                roi_size=cfg["out_size"],
            ),
            CastToTyped(keys=["image", "label_inst", "label_type"], dtype=torch.float32),
        ]
    )

    # __________________________________________________________________________
    # Create MONAI DataLoaders
    train_loader, val_loader = get_loaders(cfg, train_transforms, val_transforms)

    # --------------------------------------------------------------------------
    # Create Model, Loss, Optimizer, lr_scheduler
    # --------------------------------------------------------------------------
    # __________________________________________________________________________
    # initialize model
    model = create_model(cfg, device)
    if multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.get_rank()], output_device=dist.get_rank()
        )
    loss_function = HoVerNetLoss(lambda_hv_mse=1.0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"], weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step_size"])
    post_process_np = Compose(
        [
            Activationsd(keys=HoVerNetBranch.NP.value, softmax=True),
            AsDiscreted(keys=HoVerNetBranch.NP.value, argmax=True),
        ]
    )
    post_process = Lambdad(keys="pred", func=post_process_np)

    # --------------------------------------------
    # Ignite Trainer/Evaluator
    # --------------------------------------------
    # Evaluator
    val_handlers = [
        CheckpointSaver(
            save_dir=log_dir,
            save_dict={"model": model},
            save_key_metric=True,
        ),
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: None),
    ]
    if multi_gpu:
        val_handlers = val_handlers if dist.get_rank() == 0 else None
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        prepare_batch=PrepareBatchHoVerNet(extra_keys=["label_type", "hover_label_inst"]),
        network=model,
        postprocessing=post_process,
        key_val_metric={
            "val_dice": MeanDice(
                include_background=False,
                output_transform=from_engine_hovernet(keys=["pred", "label"], nested_key=HoVerNetBranch.NP.value),
            )
        },
        additional_metrics={
            "val_dice_NC": MeanDice(
                include_background=False,
                output_transform=from_engine_hovernet(keys=["pred", "label"], nested_key=HoVerNetBranch.NC.value),
            ),
        },
        val_handlers=val_handlers,
        amp=cfg["amp"],
    )

    # Trainer
    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=cfg["val_freq"], epoch_level=True),
        CheckpointSaver(
            save_dir=log_dir,
            save_dict={"model": model, "opt": optimizer},
            save_interval=cfg["save_interval"],
            save_final=True,
            final_filename="model.pt",
            epoch_level=True,
        ),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        TensorBoardStatsHandler(
            log_dir=log_dir, tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
        ),
    ]
    if multi_gpu:
        train_handlers = train_handlers if dist.get_rank() == 0 else train_handlers[:2]
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=cfg["n_epochs"],
        train_data_loader=train_loader,
        #epoch_length = 350,
        prepare_batch=PrepareBatchHoVerNet(extra_keys=["label_type", "hover_label_inst"]),
        network=model,
        optimizer=optimizer,
        loss_function=loss_function,
        postprocessing=post_process,
        key_train_metric={
            "train_dice": MeanDice(
                include_background=False,
                output_transform=from_engine_hovernet(keys=["pred", "label"], nested_key=HoVerNetBranch.NP.value),
            )
        },
        additional_metrics={
            "train_dice_NC": MeanDice(
                include_background=False,
                output_transform=from_engine_hovernet(keys=["pred", "label"], nested_key=HoVerNetBranch.NC.value),
            ),
        },
        train_handlers=train_handlers,
        amp=cfg["amp"],
    )
    trainer.run()

    if multi_gpu:
        dist.destroy_process_group()


def main():
    parser = ArgumentParser(description="Tumor detection on whole slide pathology images.")
    parser.add_argument(
        "--root",
        type=str,
        #default="/workspace/Data/Pathology/CoNSeP/Prepared",
        default="/home/rmartin/projects/Deep_INGENIO/destino",
        help="root data dir",
    )
    #parser.add_argument("--log-dir", type=str, default="./logs/", help="log directory")
    parser.add_argument("--log-dir", type=str, default="/home/rmartin/projects/Deep_INGENIO/logs_exp3", help="log directory")
    parser.add_argument("-s", "--seed", type=int, default=25)

    parser.add_argument("--bs", type=int, default=10, dest="batch_size", help="batch size")
    parser.add_argument("--ep", type=int, default=100, dest="n_epochs", help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, dest="lr", help="initial learning rate")
    parser.add_argument("--step", type=int, default=25, dest="step_size", help="period of learning rate decay")
    parser.add_argument("-f", "--val_freq", type=int, default=5, help="validation frequence")
    parser.add_argument("--stage", type=int, default=1, help="training stage")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="deactivate amp")
    parser.add_argument("--classes", type=int, default=4, dest="out_classes", help="output classes")
    parser.add_argument("--mode", type=str, default="original", help="choose either `original` or `fast`")

    parser.add_argument("--save_interval", type=int, default=25)
    parser.add_argument("--cpu", type=int, default=8, dest="num_workers", help="number of workers")
    parser.add_argument("--no-gpu", action="store_false", dest="use_gpu", help="deactivate use of gpu")
    parser.add_argument("--ckpt", type=str, dest="ckpt", help="model checkpoint path")

    args = parser.parse_args()
    cfg = vars(args)
    if cfg["stage"] == 1 and not cfg["ckpt"] and cfg["log_dir"]:
        cfg["ckpt"] = os.path.join(cfg["log_dir"], "stage0", "model.pt")
    print(cfg)

    logging.basicConfig(level=logging.INFO)
    log_dir = create_log_dir(cfg)
    run(log_dir, cfg)


if __name__ == "__main__":
    main()

'''
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {"CUDA_VISIBLE_DEVICES": "-1"},
            "args": ["--bs", "8", "--step", "10", "--classes", "3", "--log-dir", "${env:HOME}/projects/INGENIO/data/logs_debug", "--no-gpu"]
        }
    ]
}
''' 