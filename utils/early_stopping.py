from pydantic import BaseModel
from typing import Literal, Any
class EarlyStopping(BaseModel):

    """
    Args:
        patience (int): How many epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as improvement.
        mode (str): 'min' for loss, 'max' for accuracy/score.
    """
    patience: int = 5
    min_delta: float = 0.01
    mode : Literal['min', 'max'] = 'min'
    counter: int = 0
    best_score: float = None
    early_stop: bool = False
    logging: Any = None
    save_best_model: bool = False

    def __call__(self, current_score):
        
        # first reset the save best model flag
        self.save_best_model = False
        
        if self.best_score is None:
            self.best_score = current_score
            self.save_best_model = True
            return False

        improvement = (current_score < self.best_score - self.min_delta) if self.mode == 'min' \
            else (current_score > self.best_score + self.min_delta)

        if improvement:
            self.best_score = current_score
            self.counter = 0
            self.save_best_model = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
