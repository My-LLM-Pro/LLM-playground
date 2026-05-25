def build_optimizer(self, cfg):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    elif cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)

def build_scheduler(self, cfg):
    return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

def build_criterion(self, cfg):
    return torch.nn.CrossEntropyLoss()

def build_metric(self, cfg):
    return

