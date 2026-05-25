class Trainer:
    def __init__(self, model, optimizer, cfg):
        self.optimizer = build_optimizer(cfg)
        self.scheduler = build_scheduler(cfg)
        self.criterion = build_criterion(cfg)
        self.metric = build_metric(cfg)
        self.model = model
        self.device = cfg.device
        self.train_loader = cfg.train_loader
        self.valid_loader = cfg.valid_loader
        self.eval_loader = cfg.eval_loader
        self.valid_num = cfg.valid_num

    def train(self):
        for epoch in range(num_epochs):
            self.train_one_epoch()
            if epoch % self.valid_num == 0:
                self.validate()
                self.save_checkpoint()

    def train_one_epoch(self):
        self.model.train()
        loss_total = 0
        for batch in self.train_loader:
            loss_total += self.train_step(batch)
        loss_avg = loss_total / len(self.train_loader)
        self.logging(loss_avg)

    def train_step(self, batch):
        img, gt = batch
        img, gt = img.to(self.device), gt.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(img)
        loss = self.criterion(pred,gt)
        loss.backward()
        self.optimizer.step()
        self.logging(loss)
        return loss.item()

    def validate(self):
        val_score = 0
        self.model.eval()
        with torch.no_grad():
            for img, gt in self.valid_loader:
                img, gt = img.to(self.device), gt.to(self.device)
                pred = self.model(img)
                pred = torch.argmax(pred, dim=1)
                val_score += self.metric(pred,gt).item()
        val_score_average = val_score/ len(self.valid_loader)
        self.logging(val_score_average)
        self.model.train()
        if val_score_average > best_score:
            best_score = val_score_average
            torch.save(self.model.state_dict(), 'best_path')

    def evaluate(self):
        test_score = 0
        self.model.eval()
        with torch.no_grad():
            for img, gt in self.eval_loader:
                img, gt = img.to(device), gt.to(device)
                pre = self.model(img)
                pred = torch.argmax(pred, dim=1)
                test_score += self.metric(pre,gt).item()
        test_score_avg = test_score/ len(self.eval_loader)
        self.logging(test_score_avg)