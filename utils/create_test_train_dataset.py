import os
import shutil


def split_dataset(root_path, train_ratio=0.8, logger=None):
    num_files = len(os.listdir(root_path))

    logger.info(f' Total number of pickle files : {num_files}')

    spliter = int(max(1, num_files * train_ratio))
    train_files = os.listdir(root_path)[:spliter]
    eval_files = os.listdir(root_path)[spliter:]

    logger.info(f'Train files: {len(train_files)}')
    logger.info(f'Eval files: {len(eval_files)}')

    os.makedirs(os.path.join(root_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'eval'), exist_ok=True)

    logger.info(f'Create the train folder and copy the data')
    for file in train_files:
        source_path = os.path.join(root_path, file)
        target_path = os.path.join(root_path, "train", file)
        shutil.copy2(source_path, target_path)

    logger.info(f'Create the eval folder and copy the data')
    for file in eval_files:
        source_path = os.path.join(root_path, file)
        target_path = os.path.join(root_path, "eval", file)
        shutil.copy2(source_path, target_path)
