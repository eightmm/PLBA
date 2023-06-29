import torch
from datetime import datetime

class EarlyStopping(object):
    def __init__(self, mode='lower', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.now()
            filename = f'./save/early_stop_{dt.date()}_{dt.hour:02d}-{dt.minute:02d}-{dt.second:02d}.pth'
        
        metric = metric.lower()
        if metric is not None:
            assert metric in ['rp', 'rs', 'mae', 'rmse', 'roc_auc', 'pr_auc'],\
                f"Expect metric to be 'rp' or 'rs' or 'mae' or 'rmse' or 'roc_auc', got {metric}"
            if metric in  ['rp', 'rs', 'roc_auc', 'pr_auc']:
                print(f'For metric {metric}, the higher the better.')
                model = 'higher'
            if metric in  ['mae', 'rmse']:
                print(f'For metric {metric}, the lower the better.')
                model = 'lower'
        
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower
        
        self.patience = patience
        self.counter  = 0
        self.timestep = 0
        self.filename = filename
        
        self.best_score = None
        self.early_stop = False
        
    def _check_higer(self, score, prev_best_score):
        return score > prev_best_score
    
    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score
    
    def step(self, score, model):
        self.timestep += 1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint( model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def save_checkpoint(self, model):
        torch.save({'model_state_dict': model.state_dict(), 'timestep': self.timestep}, self.filename)
        
    def load_checkpoint(self, model):
        model.load_state_dick( torch.load(self.filename)['model_state_dict'] )