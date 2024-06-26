import numpy as np

from src.tensor import Tensor
import torch


def test_tensor_complicated():
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


def test_tensor_relu() -> None:
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


def test_tensor_exp() -> None:
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


def test_tensor_sin() -> None:
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


def test_tensor_pow() -> None:
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


def test_tensor_tanh() -> None:
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


def test_tensor_p_relu() -> None:
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


def test_tensor_sigmoid() -> None:
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


def test_tensor_sigmoid_swish() -> None:
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


def test_tensor_log() -> None:
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


def test_tensor_cos() -> None:
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


def test_tensor_matmul() -> None:
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


def test_tensor_mean() -> None:
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    b = a.mean()
    pb = pa.mean()

    b.backward()
    pb.backward()

    assert np.allclose(b.value, pb.item())
    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_sum() -> None:
    x = np.array([[3.2, 1.1, 7.3], [0, -2, 3.7]]).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    c = a.sum()
    pc = pa.sum()

    c.backward()
    pc.backward()

    assert c.value == pc.detach().numpy()
    assert np.allclose(a.grad, pa.grad)


def test_tensor_mean_extended() -> None:
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


def test_tensor_sum_extended() -> None:
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
                pa.grad = None


def test_tensor_add_sub_with_sum() -> None:
    x = np.random.rand(2, 3).astype(np.float32) * 10 - 5
    y = np.random.rand(2, 3).astype(np.float32) * 10 - 5

    a = Tensor(x)
    b = Tensor(y)

    pa = torch.tensor(x, requires_grad=True)
    pb = torch.tensor(y, requires_grad=True)

    c = a + b - a
    pc = pa + pb - pa

    d = c.sum()
    pd = pc.sum()

    d.backward()
    pd.backward()

    assert d == pd

    assert np.allclose(a.grad, pa.grad.numpy())
    assert np.allclose(b.grad, pb.grad.numpy())


def test_tensor_max() -> None:
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, requires_grad=True)

    b = a.max()
    pb = torch.max(pa)

    b.backward()
    pb.backward()

    assert b == pb

    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_min() -> None:
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, requires_grad=True)

    b = a.min()
    pb = torch.min(pa)

    b.backward()
    pb.backward()

    assert b == pb

    assert np.allclose(a.grad, pa.grad.numpy())


def test_tensor_truediv() -> None:
    x: np.ndarray[np.float32] = np.array([[2.0, 4.0], [-6.0, 8.0]]).astype(np.float32)
    y: np.ndarray[np.float32] = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float32)

    numerator = Tensor(x)
    denominator = Tensor(y)

    numerator_pt: torch.Tensor = torch.tensor(
        x, dtype=torch.float32, requires_grad=True
    )
    denominator_pt: torch.Tensor = torch.tensor(
        y, dtype=torch.float32, requires_grad=True
    )

    c: torch.Tensor = numerator / denominator
    c_pt: torch.Tensor = numerator_pt / denominator_pt

    assert c == c_pt

    c_sum: torch.Tensor = c.sum()
    c_pt_sum: torch.Tensor = c_pt.sum()

    c_sum.backward()
    c_pt_sum.backward()

    assert np.allclose(numerator.grad, numerator_pt.grad.numpy())
    assert np.allclose(denominator.grad, denominator_pt.grad.numpy())


def test_tensor_softmax_by_hand():
    x: np.ndarray[np.float32] = np.array([-6.0, 8.0, 1.0, 5.3]).astype(np.float32)

    pa: torch.Tensor = torch.tensor(x, requires_grad=True)
    pb: torch.Tensor = pa - pa.max(dim=-1, keepdim=True).values
    pc: torch.Tensor = pb.exp()
    pd: torch.Tensor = pc / pc.sum(dim=-1, keepdim=True)
    pe = pd.sum()
    pe.backward()

    a: Tensor = Tensor(x)
    b: Tensor = a - a.max(axis=-1, keepdim=True)
    c: Tensor = b.exp()
    d: Tensor = c / c.sum(axis=-1, keepdim=True)
    e: Tensor = d.sum()
    e.backward()

    assert e == pe

    assert np.allclose(a.grad, pa.grad.numpy()), "Softmax gradients do not match"


def test_tensor_softmax() -> None:
    x: np.ndarray[np.float32] = np.array([-6.0, 8.0, 1.0, 5.3]).astype(np.float32)

    a = Tensor(x)
    pa = torch.tensor(x, requires_grad=True)

    b = a.softmax(axis=-1)
    pb = torch.softmax(pa, dim=-1)

    assert b == pb

    c = b.sum()
    pc = pb.sum()

    assert c == pc

    c.backward()
    pc.backward()

    assert np.allclose(a.grad, pa.grad.numpy()), "Softmax gradients do not match"
