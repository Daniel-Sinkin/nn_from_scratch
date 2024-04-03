import numpy as np

from src.tensor import Tensor
import torch


def test_tensor_autograd_complicated():
    """Tests the autogradient with a compliated expression."""
    x = np.array([[1, 2.3, -7.3], [0, -5.2, 1.7]]).astype(np.float32)

    a = Tensor(x)
    pa: torch.Tensor = torch.tensor(x, requires_grad=True)

    intermediate_steps: list[Tensor] = []
    intermediate_steps.append(a.exp())
    intermediate_steps.append(a.sin())
    intermediate_steps.append(intermediate_steps[-1] * intermediate_steps[-2])
    intermediate_steps.append(intermediate_steps[-1] ** 2)
    intermediate_steps.append(intermediate_steps[-1].sum())

    intermediate_steps_pytorch: list[torch.Tensor] = []
    intermediate_steps_pytorch.append(pa.exp())
    intermediate_steps_pytorch.append(pa.sin())
    intermediate_steps_pytorch.append(
        intermediate_steps_pytorch[-1] * intermediate_steps_pytorch[-2]
    )
    intermediate_steps_pytorch.append(intermediate_steps_pytorch[-1] ** 2)
    intermediate_steps_pytorch.append(intermediate_steps_pytorch[-1].sum())

    intermediate_steps[-1].backward()
    intermediate_steps_pytorch[-1].backward()

    assert (
        intermediate_steps[-1].value == intermediate_steps_pytorch[-1].detach().numpy()
    )
    assert np.allclose(a.grad, pa.grad)


def test_tensor_autograd_relu() -> None:
    x = np.array([[3.2, 1.1, 7.3], [0, -2, 3.7]]).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    b = a.relu()
    pb = pa.relu()

    c = a.sum()
    pc = pa.sum()

    c.backward()
    pc.backward()

    assert c.value == pc.detach().numpy()
    assert np.allclose(a.grad, pa.grad)


def test_tensor_autograd_exp() -> None:
    x = np.array([[0.5, -1.5, 2.2], [-0.9, 2.1, -0.6]]).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    b = a.exp()
    pb = pa.exp()

    c = b.sum()
    pc = pb.sum()

    c.backward()
    pc.backward()

    assert np.allclose(c.value, pc.detach().numpy())
    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_autograd_sin() -> None:
    x = np.array(
        [[np.pi / 6, -np.pi / 4, np.pi / 3], [np.pi / 2, -np.pi / 2, 0]]
    ).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    b = a.sin()
    pb = pa.sin()

    c = b.sum()
    pc = pb.sum()

    c.backward()
    pc.backward()

    assert np.allclose(c.value, pc.detach().numpy())
    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_autograd_pow() -> None:
    x = np.array([[1.2, -1.3, 2.5], [2.2, -2.1, 0.5]]).astype(np.float32)
    power = 2

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    b = a**power
    pb = pa**power

    c = b.sum()
    pc = pb.sum()

    c.backward()
    pc.backward()

    assert np.allclose(c.value, pc.detach().numpy())
    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_autograd_tanh() -> None:
    x = np.array([[0.25, -0.75, 1.5], [-1.1, 1.3, -0.5]]).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    b = a.tanh()
    pb = pa.tanh()

    c = b.sum()
    pc = pb.sum()

    c.backward()
    pc.backward()

    assert np.allclose(c.value, pc.detach().numpy())
    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_autograd_p_relu() -> None:
    x = np.array([[1.0, -1.0, 0.5], [-0.5, 2.0, -2.5]]).astype(np.float32)
    alpha = 0.01

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    b = a.p_relu(alpha)
    pb = torch.nn.functional.prelu(pa, torch.tensor([alpha]))

    c = b.sum()
    pc = pb.sum()

    c.backward()
    pc.backward()

    assert np.allclose(c.value, pc.detach().numpy())
    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_autograd_sigmoid() -> None:
    x = np.array([[2.0, -1.0, 0.5], [-0.5, 1.5, -1.5]]).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    b = a.sigmoid()
    pb = torch.sigmoid(pa)

    c = b.sum()
    pc = pb.sum()

    c.backward()
    pc.backward()

    assert np.allclose(c.value, pc.detach().numpy())
    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_autograd_sigmoid_swish() -> None:
    x = np.array([[0.6, -1.2, 2.4], [-0.7, 1.8, -2.3]]).astype(np.float32)
    beta = 1.0

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    b = a.sigmoid_swish(beta)
    pb = pa * torch.sigmoid(beta * pa)

    c = b.sum()
    pc = pb.sum()

    c.backward()
    pc.backward()

    assert np.allclose(c.value, pc.detach().numpy())
    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_autograd_log() -> None:
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    b = a.log()
    pb = torch.log(pa)

    c = b.sum()
    pc = pb.sum()

    c.backward()
    pc.backward()

    assert np.allclose(c.value, pc.item())
    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_autograd_cos() -> None:
    x = np.array(
        [[np.pi / 4, -np.pi / 3, np.pi / 6], [2 * np.pi / 3, -np.pi / 2, 0]]
    ).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    b = a.cos()
    pb = torch.cos(pa)

    c = b.sum()
    pc = pb.sum()

    c.backward()
    pc.backward()

    assert np.allclose(c.value, pc.item())
    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_autograd_matmul() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float32)
    y = np.array([[2.0, 0.0], [0.0, 2.0]]).astype(np.float32)

    a = Tensor(x)
    b = Tensor(y)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    pb = torch.tensor(y, dtype=torch.float32, requires_grad=True)

    c = a @ b
    pc = pa @ pb

    c.sum().backward()
    pc.sum().backward()

    assert np.allclose(c.value, pc.detach().numpy())
    assert np.allclose(a.grad, pa.grad.numpy())
    assert np.allclose(b.grad, pb.grad.numpy())


def test_tensor_autograd_mean() -> None:
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    b = a.mean()
    pb = pa.mean()

    b.backward()
    pb.backward()

    assert np.allclose(b.value, pb.item())
    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_autograd_sum() -> None:
    x = np.array([[3.2, 1.1, 7.3], [0, -2, 3.7]]).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    c = a.sum()
    pc = pa.sum()

    c.backward()
    pc.backward()

    assert c.value == pc.detach().numpy()
    assert np.allclose(a.grad, pa.grad)


def test_tensor_autograd_mean_extended() -> None:
    shapes = [(2, 3), (4, 4, 4), (1, 100)]
    for shape_idx, shape in enumerate(shapes):
        x = np.random.rand(*shape).astype(np.float32)

        a = Tensor(x)
        pa = torch.tensor(x, requires_grad=True)

        b = a.mean()
        pb = pa.mean()

        b.backward()
        pb.backward()

        assert np.allclose(b.value, pb.item()), "Mean value mismatch"
        assert np.allclose(
            a.grad, pa.grad.numpy()
        ), "Gradient mismatch for mean across all dimensions"

        if len(shape) > 1:
            for dim in range(len(shape)):
                a.zero_grad()
                pa.grad = None

                b = a.mean(axis=dim)
                pb = pa.mean(dim=dim)

                b.backward()
                grad_tensor = torch.ones_like(pb)
                pb.backward(gradient=grad_tensor)

                assert np.allclose(
                    b.value, pb.detach().numpy()
                ), f"{shape_idx=} : Mean value mismatch across dimension {dim}"

                print(0)

                assert np.allclose(
                    a.grad, pa.grad.numpy()
                ), f"{shape_idx=} : Gradient mismatch for mean across dimension {dim}"


def test_tensor_autograd_sum_extended() -> None:
    shapes = [(3, 2), (2, 2, 2), (10,)]
    for shape in shapes:
        x = np.random.rand(*shape).astype(np.float32) * 10 - 5

        a = Tensor(x)
        pa = torch.tensor(x, requires_grad=True)

        c = a.sum()
        pc = pa.sum()

        c.backward()
        pc.backward()

        assert np.allclose(c.value, pc.item()), "Sum value mismatch"
        assert np.allclose(
            a.grad, pa.grad.numpy()
        ), "Gradient mismatch for sum across all dimensions"

        if len(shape) > 1:
            for dim in range(len(shape)):
                a.zero_grad()
                pa.grad = None  # Reset


def test_tensor_autograd_add_sub_with_sum() -> None:
    x = np.random.rand(2, 3).astype(np.float32) * 10 - 5
    y = np.random.rand(2, 3).astype(np.float32) * 10 - 5

    # Create Tensor instances for x and y
    a = Tensor(x)
    b = Tensor(y)
    # Create PyTorch tensors with gradients enabled
    pa = torch.tensor(x, requires_grad=True)
    pb = torch.tensor(y, requires_grad=True)

    # Perform addition and subtraction operations
    c = a + b - a  # Custom Tensor operations
    pc = pa + pb - pa  # PyTorch operations

    # Apply sum reduction
    d = c.sum()
    pd = pc.sum()

    # Backpropagate
    d.backward()
    pd.backward()

    # Verify the final value after operations and sum reduction
    assert np.allclose(
        d.value, pd.detach().numpy()
    ), "Final values mismatch after add, sub, and sum"

    # Verify gradients w.r.t. the original inputs
    assert np.allclose(a.grad, pa.grad.numpy()), "Gradient mismatch for 'a'"
    assert np.allclose(b.grad, pb.grad.numpy()), "Gradient mismatch for 'b'"
