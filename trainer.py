import os
import shutil
import random
import mlflow
import logging
import torch
from torch.optim import AdamW
from models import BaseModel, LoraModel
from utils.load_dataset import LoadDatasetAndSaveDataset
from utils.create_test_train_dataset import split_dataset
from peft import LoraConfig
from torch.utils.data import DataLoader
from datasets import CustomDataset

# logging setup
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train.log')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# set seed
random.seed(42)


def train(model_name: str, file_paths: list[str], load_mode: str = 'excel', local_path: str = '', batch_size: int = 1,
          lora_method: bool = False, num_epochs: int = 10, train_ratio: float = 0.8, chunk_size: int = 30,
          lora_rank: int = 7, lora_alpha: int = 16, init_lora_weights: str = 'gaussian', lora_dropout: float = 0.0,
          target_modules=None, learning_rate: float = 1e-6, weight_decay: float = 0.001, tracking_uri: str = ''):
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj', 'k_proj']
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run():

        # print model information
        mlflow.log_param('model_name', model_name)
        mlflow.log_param('file_paths', file_paths)
        mlflow.log_param('load_mode', load_mode)
        mlflow.log_param('local_path', local_path)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        mlflow.log_param('epochs', num_epochs)
        mlflow.log_param('train_ratio', train_ratio)
        mlflow.log_param('seed', 42)
        mlflow.log_param('data_chunk_size', chunk_size)
        mlflow.log_param('learning rate', learning_rate)
        mlflow.log_param('weight_decay', weight_decay)

        if lora_method:
            mlflow.log_param('lora_method', lora_method)
            mlflow.log_param('lora_rank', lora_rank)
            mlflow.log_param('lora_alpha', lora_alpha)
            mlflow.log_param('init_lora_weights', init_lora_weights)
            mlflow.log_param('lora_dropout', lora_dropout)
            mlflow.log_param('target_modules', target_modules)

        # create the storage
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        os.makedirs(local_path, exist_ok=True)

        # check the availability for GPU/CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'device: {device}')

        # load the dataset
        logger.info(f'Dataset : Loading and Saving started')
        load_save_data = LoadDatasetAndSaveDataset(file_paths=file_paths, mode=load_mode, local_path=local_path,
                                                   model_name=model_name, logger=logger, chunk_size=chunk_size)
        load_save_data.start()
        logger.info(f'Dataset : Loading and Saving Completed')

        # split dataset
        logger.info(f' Split train and test dataset')
        split_dataset(local_path, train_ratio=0.8, logger=logger)

        # initialize the dataset
        logger.info(f' Create the pytorch dataset')
        train_dataset = CustomDataset(root_path=local_path, mode='train')
        val_dataset = CustomDataset(root_path=local_path, mode='eval')

        # initialize dataloader
        logger.info(f'create the pytorch dataset loader')
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # load the model
        logger.info(f'Load the Model')
        base_model = BaseModel(device, model_name).load_model()

        # enable the lora configuration
        if lora_method:
            llm_lora_config = LoraConfig(r=lora_rank,
                                         lora_alpha=lora_alpha,
                                         init_lora_weights=init_lora_weights,
                                         bias='none',
                                         lora_dropout=lora_dropout,
                                         target_modules=target_modules)

            model = LoraModel(llm_lora_config, base_model).load_lora_model()
        else:
            model = base_model

        # initialize the optimizer
        logger.info(f'Initialize the optimizer')
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # add log into artifact (initial information)
        mlflow.log_artifact(log_file)

        for epoch in range(num_epochs):

            # training module
            total_loss = 0.0
            train_step = 0
            mlflow.set_tag('phase', 'training')
            for data in train_loader:
                optimizer.zero_grad()

                data['input_ids'] = data['input_ids'].to(device)
                data['attention_mask'] = data['attention_mask'].to(device)
                data['labels'] = data['labels'].to(device)

                output = model(data['input_ids'], data['attention_mask'], labels=data['labels'])

                output.loss.backward()
                optimizer.step()

                total_loss += output.loss.item()
                train_step += 1

                mlflow.log_metric(f'steps_{epoch}', train_step)
                mlflow.log_metric(f'training_loss_{epoch}', output.loss.item(), step=train_step)

            logger.info("Training Loss: {}".format(total_loss / train_step))
            mlflow.log_artifact(log_file)

            # evaluation module
            model.eval()
            with torch.no_grad():
                total_eval_loss = 0.0
                eval_step = 0
                mlflow.set_tag('phase', 'evaluation')
                for data in val_loader:
                    data['input_ids'] = data['input_ids'].to(device)
                    data['attention_mask'] = data['attention_mask'].to(device)
                    data['labels'] = data['labels'].to(device)

                    output = model(data['input_ids'], data['attention_mask'], labels=data['labels'])

                    total_eval_loss += output.loss.item()
                    eval_step += 1
                    mlflow.log_metric(f'eval_loss_{epoch}', output.loss.item(), step=eval_step)

                logger.info("Validation Loss: {}".format(total_eval_loss / eval_step))

            mlflow.log_metrics({
                "epoch_train_loss": total_loss / train_step,
                "epoch_val_loss": total_eval_loss / eval_step,
                'epoch': epoch + 1
            }, step=epoch)
            model_saving_path = os.path.join(local_path, f'model_{epoch}.pth')

            mlflow.log_artifact(log_file)


if __name__ == '__main__':
    train(model_name='google/gemma-3-1b-pt',
          local_path='G:/WorkSpace/llm_finetune/working_dir',
          file_paths=["C:/Users/praja/Downloads/llm_fine_tune_100_rows.xlsx"], lora_method=True,
          tracking_uri='http://localhost:5000'
          )
