class EarlyStopper:
    def __init__(self, max_patient: int, mode: str):
        self.best_target_value: int = 0
        self.patience: int = 0
        self.finish: bool = False
        self.max_patient: int = max_patient
        self.mode: str = mode

    def update(self, target_value):
        if self.mode == "min":
            target_value = -target_value
        if self.best_target_value < target_value:
            self.patience = 0
            self.best_target_value = target_value
        else:
            self.patience += 1
            if self.max_patient <= self.patience:
                self.finish = True
