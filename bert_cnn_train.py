import os
import torch
import wandb
import random
import config
import pandas as pd
import lightning as L
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from modules.multimodal_classifier import BERT_CNN_Classifier
from modules.image_text_dataset import Multi_Modal_Dataset_Tensors
from modules.collate_fn import collate_X_Y_Z



def main():
    # Reproducibility
    seed = 1
    L.seed_everything(seed=seed, workers=True)
    torch.manual_seed(seed)
    random.seed(seed)

    # load the dataframe
    data_dir = "data"
    train_dir = os.path.join(data_dir, "batch")        
    df_train = pd.read_csv(os.path.join(data_dir, 'multimodal_train.tsv'), sep='\t')

    # Prepare labels from dataframe
    df_train_labels = df_train[['id','clean_title', '2_way_label', '3_way_label', '6_way_label']]
    df_train_labels.set_index('id', inplace=True)
    img_ids = [img.split('.')[0] for img in os.listdir(train_dir)]
    df_train_labels = df_train_labels.loc[img_ids]


    # Create a transform to convert image to tensor
    img_to_tensor = transforms.Compose([
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define the number of classifications
    classes = 2


    # Create training dataset
    dataset = Multi_Modal_Dataset_Tensors(
        img_dir=batch_dir,
        df_labels=df_train_labels,
        transform=img_to_tensor,
        classes=classes,
    )

    # Create the dataloader for the training dataset
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_X_Y_Z,
        num_workers=3,
        persistent_workers=True,
        pin_memory=True
    )


    # Create the multimodal model
    multimodal_model = BERT_CNN_Classifier(
        input_size=dataset.max_seq_len,
        num_layers=1,
        num_channels=3,
        classes=classes
    )

    print("test?")

    # Setup weights and biases for logging
    os.environ['WANDB_API_KEY'] = config.WANDB_API_KEY
    try:
        print("already logged on wandb...")
        wandb.init(project='dat550_Bert_EffNetB4', reinit=True, name=f'run_{"multi_modal"}_{"adam"}')
        wandb_logger = WandbLogger()
    except wandb.errors.UsageError:
        print("logging on wandb...")
        wandb.login()
        wandb.init(project='dat550_Bert_EffNetB4', reinit=True, name=f'run_{"multi_modal"}_{"adam"}')
        wandb_logger = WandbLogger()

    # Create the trainer
    trainer = L.Trainer(
        logger=wandb_logger, 
        max_epochs=10, 
        accelerator="auto", 
        devices="auto",
        log_every_n_steps=len(dataset)//dataloader.batch_size,
        profiler="simple",
        deterministic=True,
        accumulate_grad_batches=len(dataset)//dataloader.batch_size
    )

    # set model to train mode
    multimodal_model.train()
    trainer.fit(multimodal_model,train_dataloaders=dataloader)
    wandb.finish()


    # Save the weights of the model
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')

    custom_args =  {
        "input_size": multimodal_model.input_size,
        "num_layers": multimodal_model.num_layers,
        "num_channels": multimodal_model.num_channels,
        "classes": multimodal_model.classes
    }
    data_to_save = multimodal_model.state_dict()
    data_to_save["__custom_arguments__"] = custom_args
    torch.save(data_to_save, 'trained_models/BERT_CNN.pth')


if __name__ == "__main__":
    main()