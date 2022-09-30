def dice_coeff(pred, label):
    # inter = torch.dot(pred.view(-1), label.view(-1))
    # union = torch.sum(pred) + torch.sum(label)
    # dice = (2 * inter.float() ) / union.float()
    inter = pred.reshape(-1).dot(label.reshape(-1))
    union = pred.sum() + label.sum()
    dice = (2. * inter) / union
    return dice
