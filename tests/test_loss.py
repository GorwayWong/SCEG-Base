"""Independent scalar and derivative checks, including the official binary edge."""
import math
import torch
from losses import IntraInterLoss


def test_loss_matches_scalar_formula(config):
    scores = torch.tensor([[0.9, 0.2, -0.1], [0.1, 0.95, 0.3]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    criterion = IntraInterLoss(3, **config["loss"]).double()
    expected = []
    for row, target in zip(scores.tolist(), labels.tolist()):
        positive = math.exp(-criterion.lamda.item()*(row[target]-criterion.intra))
        negatives = sum(math.exp(criterion.lamda.item()*(score-(criterion.intra-criterion.inter)))
                        for index, score in enumerate(row) if index != target)
        expected.append(math.log(1+positive)+math.log(1+negatives))
    torch.testing.assert_close(criterion(scores, labels),
                               torch.tensor(sum(expected)/len(expected), dtype=torch.float64))


def test_loss_gradcheck(config):
    criterion = IntraInterLoss(3, **config["loss"]).double()
    scores = torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.9, 0.2]],
                           dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(criterion, (scores, torch.tensor([0, 1])))


def test_binary_broadcast_is_preserved(config):
    criterion = IntraInterLoss(2, **config["loss"])
    scores = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    expected = (torch.log(1+torch.exp(criterion.lamda*(scores[0]-(criterion.intra-criterion.inter))).sum())
                + torch.log(1+torch.exp(-criterion.lamda*(scores[1]-criterion.intra)).sum()))/2
    torch.testing.assert_close(criterion(scores, torch.tensor([0, 1])), expected)
