from data_utils import *

output_path = '/output/'
USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# evaluation metric
def f1_threshold(y_true, preds):
    best_f1 = 0
    for i in np.arange(0.1, 0.51, 0.01):
        f1 = f1_score(y_true, preds > i)
        if f1 > best_f1:
            threshold = i
            best_f1 = f1
    return best_f1, threshold


# solver of model with validation
class NetSolver(object):

    def __init__(self, model, **kwargs):
        self.model = model

        # hyperparameters
        self.lr_init = kwargs.pop('lr_init', 0.001)
        self.lr_decay = kwargs.pop('lr_decay', 0.1)
        self.step_size = kwargs.pop('step_size', 10)
        self.print_every = kwargs.pop('print_every', 2000)
        self.checkpoint_name = kwargs.pop('checkpoint_name', 'qiqc')

        # setup optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.lr_init)
        # self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
        #                            lr=self.lr_init, momentum=0.9)
        # self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.lr_decay)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, factor=self.lr_decay, patience=10)

        self.model = self.model.to(device=device)
        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        self.best_val_loss = 0.
        self.best_val_f1 = 0.
        self.loss_history = []
        self.val_loss_history = []
        self.f1_history = []
        self.val_f1_history = []

    def _save_checkpoint(self, epoch, l_val, f_val):
        torch.save(self.model.state_dict(),
            output_path+self.checkpoint_name+'_%.3f_%.3f_epoch_%d.pth.tar' %(l_val, f_val, epoch))
        checkpoint = {
            'optimizer': str(type(self.optimizer)),
            'scheduler': str(type(self.scheduler)),
            'lr_init': self.lr_init,
            'lr_decay': self.lr_decay,
            'step_size': self.step_size,
            'epoch': epoch,
        }
        with open(output_path+'hyper_param_optim.json', 'w') as f:
            json.dump(checkpoint, f)


    def forward_pass(self, x, y):
        x = x.to(device=device, dtype=torch.long)
        y = y.to(device=device, dtype=dtype)
        scores = self.model(x)
        loss = F.binary_cross_entropy_with_logits(scores, y)
        return loss, torch.sigmoid(scores)


    def lr_range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
        epochs = int(np.ceil(num_it/len(train_loader)))

        lrs_log, loss_log = [], []
        n, curr_lr = 0, start_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = curr_lr

        for e in range(epochs):
            self.model.train()
            for x, y in train_loader:
                loss, _ = self.forward_pass(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lrs_log.append(curr_lr)
                loss_log.append(loss.item())

                # update best loss
                if n == 0:
                    best_loss, n_best = loss.item(), n
                else:
                    if loss.item() < best_loss:
                        best_loss, n_best = loss.item(), n

                # update lr per iter
                n += 1
                curr_lr = start_lr * (end_lr/start_lr) ** (n/num_it)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = curr_lr

                # stopping condition
                if n == num_it or (stop_div and (loss.item() > 4*best_loss or torch.isnan(loss))):
                    break

        print('minimum loss {}, at lr {}'.format(best_loss, lrs_log[n_best]))
        return lrs_log, loss_log


    def train(self, loaders, epochs):
        train_loader, val_loader = loaders

        # Start training for epochs
        for e in range(epochs):
            self.model.train()
            print('\nEpoch %d / %d:' % (e + 1, epochs))
            #self.scheduler.step()   # for StepLR
            running_loss = 0.

            for t, (x, y) in enumerate(train_loader):
                loss, _ = self.forward_pass(x, y)

                if (t + 1) % self.print_every == 0:
                    print('t = %d, loss = %.4f' % (t+1, loss.item()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * x.size(0)

            N = len(train_loader.dataset)
            train_loss = running_loss / N
            print('For train set,')
            train_f1, train_thres, _ = self.check_f1(train_loader, num_batches=50)
            print('For val set,')
            val_f1, val_thres, val_loss = self.check_f1(val_loader)
            #self.scheduler.step(val_loss)   # for ReduceLROnPlateau

            self.log_and_checkpoint(e, train_loss, val_loss, train_f1, val_f1)


    def train_one_cycle(self, loaders, epochs, max_lr, moms=(.95, .85), div_factor=25,
                        sep_ratio=0.3, final_div=None):
        train_loader, val_loader = loaders
        if final_div is None: final_div = div_factor*1e4

        # one-cycle setup
        tot_it = epochs * len(train_loader)
        up_it = int(tot_it * sep_ratio)
        down_it = tot_it - up_it
        min_lr = max_lr / div_factor
        final_lr = max_lr / final_div
        n, curr_lr, curr_mom = 0, min_lr, moms[0]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = curr_lr
            param_group['betas'] = (curr_mom, 0.999)

        # Start training for epochs
        for e in range(epochs):
            self.model.train()
            print('\nEpoch %d / %d:' % (e + 1, epochs))
            running_loss = 0.

            for t, (x, y) in enumerate(train_loader):
                loss, _ = self.forward_pass(x, y)

                if (t + 1) % self.print_every == 0:
                    print('t = %d, loss = %.4f' % (t+1, loss.item()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * x.size(0)

                # update lr, mom per iter
                n += 1
                if n <= up_it:
                    curr_lr = max_lr + (min_lr - max_lr)/2 * (np.cos(np.pi*n/up_it)+1)
                    curr_mom = moms[1] + (moms[0] - moms[1])/2 * (np.cos(np.pi*n/up_it)+1)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = curr_lr
                        param_group['betas'] = (curr_mom, 0.999)
                else:
                    curr_lr = final_lr + (max_lr - final_lr)/2 * (np.cos(np.pi*(n-up_it)/down_it)+1)
                    curr_mom = moms[0] + (moms[1] - moms[0])/2 * (np.cos(np.pi*(n-up_it)/down_it)+1)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = curr_lr
                        param_group['betas'] = (curr_mom, 0.999)

            N = len(train_loader.dataset)
            train_loss = running_loss / N
            print('For train set,')
            train_f1, train_thres, _ = self.check_f1(train_loader, num_batches=50)
            print('For val set,')
            val_f1, val_thres, val_loss = self.check_f1(val_loader)

            self.log_and_checkpoint(e, train_loss, val_loss, train_f1, val_f1)


    def log_and_checkpoint(self, e, train_loss, val_loss, train_f1, val_f1):
        # Checkpoint and record/print metrics at epoch end
        self.loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.f1_history.append(train_f1)
        self.val_f1_history.append(val_f1)

        # for floydhub metric graphs
        print('{"metric": "F1", "value": %.4f, "epoch": %d}' % (train_f1, e+1))
        print('{"metric": "Val. F1", "value": %.4f, "epoch": %d}' % (val_f1, e+1))
        print('{"metric": "Loss", "value": %.4f, "epoch": %d}' % (train_loss, e+1))
        print('{"metric": "Val. Loss", "value": %.4f, "epoch": %d}' % (val_loss, e+1))

        is_updated = False
        if e == 0:
            self.best_val_f1 = val_f1
            self.best_val_loss = val_loss
        if val_f1 > self.best_val_f1:
            print('updating best val f1...')
            self.best_val_f1 = val_f1
            is_updated = True
        if val_loss < self.best_val_loss:
            print('updating best val loss...')
            self.best_val_loss = val_loss
            is_updated = True
        if e > 1 and is_updated:
            print('Saving model...')
            self._save_checkpoint(e+1, val_loss, val_f1)
        print()


    def check_f1(self, loader, num_batches=None, save_scores=False):
        self.model.eval()
        targets, scores, losses = [], [], []
        with torch.no_grad():
            for t, (x, y) in enumerate(loader):
                l, score = self.forward_pass(x, y)
                targets.append(y.cpu().numpy())
                scores.append(score.cpu().numpy())
                losses.append(l.item())
                if num_batches is not None and (t+1) == num_batches:
                    break

        targets = np.concatenate(targets)
        scores = np.concatenate(scores)
        if save_scores:
            self.val_scores = scores  # to access from outside

        best_f1, threshold = f1_threshold(targets, scores)
        loss = np.mean(losses)
        print('Best threshold is {:.4f}, with F1 score: {:.4f}'.format(threshold, best_f1))

        return best_f1, threshold, loss


