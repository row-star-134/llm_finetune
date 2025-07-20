import json
import pickle
import os

import mlflow
import torch
import pandas as pd
from typing import Literal, Any
import uuid as uuid4
from pydantic import BaseModel
from utils.tokenize import CustomTokenizer


class LoadDatasetAndSaveDataset(BaseModel):
    mode: str = Literal['excel', 'csv', 'json']
    file_paths: list[str] = []
    local_path: str = ''
    model_name: str = ''
    chunk_size: int = 30
    tokenizer: Any = None
    logger: Any = None

    def save_dict(self, response):

        """


        :param response:  It contains the list of dictionary for prompt and answer
        :process : it tokenizer all prompt and answer, next it chunk data and save into pickle file
        :return:
        """

        tokenizer_response = []
        tokenizer = CustomTokenizer(model_name=self.model_name)
        for res in response:
            prompt_token = tokenizer.tokenization(res['prompt'])
            answer_token = tokenizer.answer_tokenization(res['answer'])

            prompt_ids = prompt_token['input_ids']
            answer_ids = answer_token['input_ids']

            input_ids = torch.concat([prompt_ids, answer_ids], dim=1)
            labels = torch.concat([torch.tensor([-100] * len(prompt_ids[0])), answer_ids[0]], dim=0)

            attention_mask = torch.concat([prompt_token['attention_mask'], answer_token['attention_mask']], dim=1)

            tokenizer_response.append({"input_ids": input_ids.squeeze(0),
                                       "labels": labels,
                                       "attention_mask": attention_mask.squeeze(0)
                                       })
        pickle_file = os.path.join(self.local_path, uuid4.uuid4().hex + ".pkl")
        self.logger.info(f'Saving pickle file : {pickle_file}')
        with open(pickle_file, 'wb') as f:
            pickle.dump(tokenizer_response, f)

    def load_excel(self):
        total_data = 0
        for file in self.file_paths:
            self.logger.info(f'Loading excel file : {file}')
            df = pd.read_excel(file)

            total_data = total_data + len(df)

            data_batch = []
            for index, row in df.iterrows():
                prompt = row['prompt']
                answer = row['answer']
                output = {
                    "prompt": prompt,
                    "answer": answer,

                }

                data_batch.append(output)

                if len(data_batch) == self.chunk_size:
                    self.save_dict(data_batch)
                    data_batch.clear()

            if data_batch:
                self.save_dict(data_batch)
        mlflow.log_param('total_data', total_data)

    def load_csv(self):
        total_data = 0
        self.logger.info(f'Loading csv file : {self.local_path}')
        for file in self.file_paths:
            df = pd.read_csv(file)

            total_data += len(df)
            data_batch = []
            for index, row in df.iterrows():
                prompt = row['prompt']
                answer = row['answer']
                output = {
                    "prompt": prompt,
                    "answer": answer,
                }

                data_batch.append(output)

                if len(data_batch) == self.chunk_size:
                    self.save_dict(data_batch)
                    data_batch.clear()

            if data_batch:
                self.save_dict(data_batch)
        mlflow.log_param('total_data', total_data)

    def load_json(self):
        for file in self.file_paths:
            self.logger.info(f'Loading json file : {file}')
            df = json.load(open(file, 'r'))
            data_batch = []

            for row in df:
                prompt = row['prompt']
                answer = row['answer']
                output = {
                    "prompt": prompt,
                    "answer": answer,
                }
                data_batch.append(output)

                if len(data_batch) == self.chunk_size:
                    self.save_dict(data_batch)
                    data_batch.clear()

            if data_batch:
                self.save_dict(data_batch)

    def load_dataset(self):
        """
            Load the dataset

        """

        if self.mode == 'excel':
            self.load_excel()

        elif self.mode == 'csv':
            self.load_csv()

        elif self.mode == 'json':
            self.load_json()

    def start(self):
        self.load_dataset()
