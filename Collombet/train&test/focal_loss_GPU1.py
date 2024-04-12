import torch
import torch.nn as nn



def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# This method is used to calculate the Focal loss
class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[], gamma=2, reduction='mean'):
        """
        :param alpha: A list of weight coefficients, where the weight obtained for each of the four categories is the reciprocal of the frequency of occurrence of that category
        :param gamma: Difficult samples are mined with gamma,gamma is given to the wrongly scored samples with weights turned up.
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        device = try_gpu(1)
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # Assign category weights to the samples within the current batch one by one, shape=(batch_size), 1D vector
        #log_softmax = torch.log_softmax(pred, dim=1) # Softmax the model output and take log, shape=(batch_size, 4)
        log_softmax = pred # Because the last step of the deep model has already performed log_softmax on the model, there is no need to do it again here
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # Fetch the log_softmax value for each sample at the category label position, shape=(batch_size, 1)
        logpt = logpt.view(-1)  # Dimensionality reduction, shape=(batch_size)
        ce_loss = -logpt  # Take another negative to log_softmax and it's cross entropy
        pt = torch.exp(logpt)  # Take exp for log_softmax, eliminate log, that's the softmax value for each sample at the category label position, shape=(batch_size)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # Calculate the focal loss to get the loss value for each sample according to the formula, shape=(batch_size)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss