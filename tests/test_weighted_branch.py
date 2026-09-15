import copy

import pytest
import torch

from maud_qwen.train import normalized_example_loss


def test_rare_gradient_has_threefold_relative_influence():
    # Two independent example losses expose the gradient contribution directly.
    losses = torch.tensor([2.0, 2.0], requires_grad=True)
    total = normalized_example_loss(losses[0], 1.0, 4.0) + normalized_example_loss(losses[1], 3.0, 4.0)
    total.backward()
    assert losses.grad.tolist() == [0.25, 0.75]
    assert total.item() == 2.0


def test_uniform_weights_preserve_loss_and_gradient():
    for weights in ([1., 1., 1.], [3., 3., 3.]):
        values = torch.tensor([0.5, 2., 4.], requires_grad=True)
        actual = sum(normalized_example_loss(loss, weight, sum(weights)) for loss, weight in zip(values, weights))
        expected = values.mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, values)[0], torch.autograd.grad(expected, values)[0])


def test_singleton_final_batch_does_not_amplify_learning_rate():
    value = torch.tensor(2., requires_grad=True)
    actual = normalized_example_loss(value, 3., 3.)
    actual.backward()
    assert actual.item() == 2. and value.grad.item() == 1.


def test_weighted_accumulation_matches_independent_full_batch_objective_and_resume():
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False).double()
    reference = copy.deepcopy(model)
    x = torch.randn(4, 3, dtype=torch.double)
    y = torch.tensor([0, 1, 0, 1])
    weights = torch.tensor([1., 3., 1., 3.], dtype=torch.double)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-5, weight_decay=.01)
    for xi, yi, wi in zip(x, y, weights):
        loss = torch.nn.functional.cross_entropy(model(xi[None]), yi[None])
        normalized_example_loss(loss, float(wi), float(weights.sum())).backward()
    loss = (torch.nn.functional.cross_entropy(reference(x), y, reduction='none') * weights).sum() / weights.sum()
    loss.backward()
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    optimizer.step(); reference_optimizer.step()
    torch.testing.assert_close(model.weight, reference.weight)
    saved_model, saved_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    restored = copy.deepcopy(model)
    restored.load_state_dict(saved_model)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-5, weight_decay=.01)
    restored_optimizer.load_state_dict(saved_optimizer)
    for m, opt in [(model, optimizer), (restored, restored_optimizer)]:
        opt.zero_grad(set_to_none=True)
        for xi, yi, wi in zip(x, y, weights):
            normalized_example_loss(torch.nn.functional.cross_entropy(m(xi[None]), yi[None]), float(wi), float(weights.sum())).backward()
        opt.step()
    torch.testing.assert_close(model.weight, restored.weight, rtol=0, atol=0)
