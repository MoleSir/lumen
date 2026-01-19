import math
from lumen import Tensor, DType


# ---- Type & Attributes ----

def test_dtype_conversion():
    t = Tensor([1.5, 2.5], dtype=DType.Float32)
    assert t.dtype() == DType.Float32
    
    t_int = t.to_dtype(DType.Int32)
    assert t_int.dtype() == DType.Int32
    # Expect truncation or rounding depending on implementation, 
    # assuming standard casting behavior: 1, 2
    assert t_int.allclose(Tensor([1, 2], dtype=DType.Int32))

def test_tensor_item():
    t = Tensor([42.0])
    scalar_tensor = t.item()
    assert scalar_tensor == 42.0


# ---- Math Operations ----

def test_unary_math_ops():
    t = Tensor([0.0, math.pi / 2.0, math.pi])
    
    sin_t = t.sin()
    # sin(0)=0, sin(pi/2)=1, sin(pi)~0
    assert sin_t.allclose(Tensor([0.0, 1.0, 0.0]), atol=1e-6)
    
    cos_t = t.cos()
    # cos(0)=1, cos(pi/2)=0, cos(pi)=-1
    assert cos_t.allclose(Tensor([1.0, 0.0, -1.0]), atol=1e-6)

def test_exp_log_ops():
    t = Tensor([1.0, 2.0])
    exp_t = t.exp()
    
    expected = Tensor([math.exp(1.0), math.exp(2.0)])
    assert exp_t.allclose(expected)
    
    ln_t = exp_t.ln()
    assert ln_t.allclose(t)

def test_rounding_ops():
    t = Tensor([-1.7, -1.2, 1.2, 1.7])
    
    assert t.floor().allclose(Tensor([-2.0, -2.0, 1.0, 1.0]))
    assert t.ceil().allclose(Tensor([-1.0, -1.0, 2.0, 2.0]))
    assert t.round().allclose(Tensor([-2.0, -1.0, 1.0, 2.0]))
    assert t.abs().allclose(Tensor([1.7, 1.2, 1.2, 1.7]))


# ---- Matrix Operations ----

def test_matmul_basic():
    # A: (2, 3), B: (3, 2)
    a = Tensor([[1., 2., 3.], [4., 5., 6.]])
    b = Tensor([[7., 8.], [9., 1.], [2., 3.]])
    
    c = a @ b # or a.matmul(b)
    
    assert c.dims() == [2, 2]
    # Row 1: 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    # Row 2: 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    expected = Tensor([[31., 19.], [85., 55.]])
    assert c.allclose(expected)

def test_broadcasting_add():
    # (2, 3) + (1, 3) -> (2, 3)
    a = Tensor([[1., 2., 3.], [4., 5., 6.]])
    b = Tensor([[10., 20., 30.]])
    
    c = a + b
    expected = Tensor([[11., 22., 33.], [14., 25., 36.]])
    assert c.allclose(expected)


# ---- Reductions ----

def test_sum_reduction():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    
    # Sum all
    total = t.sum()
    assert total.allclose(Tensor.new(21))
    
    # Sum dim 0 -> [1+4, 2+5, 3+6]
    col_sum = t.sum(dim=0)
    assert col_sum.dims() == [3]
    assert col_sum.allclose(Tensor([5, 7, 9]))
    
    # Sum dim 1, keep_dim -> [[6], [15]]
    row_sum_kd = t.sum(dim=1, keep_dim=True)
    assert row_sum_kd.dims() == [2, 1]
    assert row_sum_kd.allclose(Tensor([[6], [15]]))

def test_min_max_reduction():
    t = Tensor([[10., 20.], [5., 50.]])
    
    print(t.max())
    assert t.max().allclose(Tensor.new(50.0))
    assert t.min().allclose(Tensor.new(5.0))
    
    # Max along dim 1 -> [20, 50]
    assert t.max(dim=1).allclose(Tensor([20., 50.]))


# ---- Shape Manipulation ----

def test_reshape_flatten():
    t = Tensor.range(0, 6).reshape((2, 3)) if hasattr(Tensor, "range") else Tensor([[0, 1, 2], [3, 4, 5]])
    
    # (2, 3) -> (3, 2)
    reshaped = t.reshape((3, 2))
    assert reshaped.dims() == [3, 2]
    assert reshaped.allclose(Tensor([[0, 1], [2, 3], [4, 5]]))
    
    # Flatten all
    flat = t.flatten_all()
    assert flat.dims() == [6]

def test_transpose_permute():
    t = Tensor([[1, 2, 3], [4, 5, 6]]) # (2, 3)
    
    # Transpose 0, 1 -> (3, 2)
    t_t = t.transpose(0, 1)
    assert t_t.dims() == [3, 2]
    assert t_t.allclose(Tensor([[1, 4], [2, 5], [3, 6]]))
    
    # Permute (synonym for transpose in 2D)
    t_p = t.permute([1, 0])
    assert t_p.dims() == [3, 2]
    assert t_p.allclose(t_t)

def test_squeeze_unsqueeze():
    t = Tensor.zeros((2, 1, 3))
    
    sq = t.squeeze(1)
    assert sq.dims() == [2, 3]
    
    unsq = sq.unsqueeze(0)
    assert unsq.dims() == [1, 2, 3]


# ---- Neural Network Activations ----

def test_activations():
    t = Tensor([-1.0, 0.0, 1.0])
    
    # ReLU: max(0, x) -> [0, 0, 1]
    assert t.relu().allclose(Tensor([0.0, 0.0, 1.0]))
    
    # Sigmoid: 1 / (1 + exp(-x))
    # sig(0) = 0.5
    s = t.sigmoid()
    assert s.allclose(Tensor([0.26894, 0.5, 0.73105]), atol=1e-4)


# ---- Comparison and Conditionals ----

def test_comparison_ops():
    t = Tensor([1, 2, 3])
    
    # Element-wise equality
    # eq returns a Tensor (likely 0s and 1s or bool dtype)
    mask = t.eq(2) 
    assert mask.to_dtype(DType.Int32).allclose(Tensor([0, 1, 0]))
    
    # Greater than
    gt_mask = t.gt(1)
    assert gt_mask.to_dtype(DType.Int32).allclose(Tensor([0, 1, 1]))

def test_where_masked_fill():
    t = Tensor([1, 2, 3, 4])
    mask = t.gt(2) # [F, F, T, T]
    
    # Masked fill
    filled = t.masked_fill(mask, 0) # Replace > 2 with 0
    assert filled.allclose(Tensor([1, 2, 0, 0]))
    
    # If Else (similar to numpy.where)
    # if t > 2 then 10 else -10
    cond = mask.if_else(10, -10)
    assert cond.allclose(Tensor([-10, -10, 10, 10]))


# ---- Concatenation and Splitting ----

def test_cat_stack():
    t1 = Tensor([1, 2])
    t2 = Tensor([3, 4])
    
    # Cat dim 0 -> [1, 2, 3, 4]
    c = Tensor.cat([t1, t2], dim=0)
    assert c.dims() == [4]
    assert c.allclose(Tensor([1, 2, 3, 4]))
    
    # Stack dim 0 -> [[1, 2], [3, 4]]
    s = Tensor.stack([t1, t2], dim=0)
    assert s.dims() == [2, 2]

def test_split_chunk():
    t = Tensor([1, 2, 3, 4, 5, 6])
    
    # Split into size 2 chunks
    # Note: definition of split usually implies size of each chunk or number of sections
    # Assuming `split(self, dim)` splits into equal parts or size 1? 
    # Let's check typical PyTorch behavior: split(split_size, dim). 
    # The stub has `split(self, dim: int) -> List[Tensor]`, ambiguous on size.
    # Assuming it splits into size 1 along dim (like unbind) or chunks?
    # Let's test `chunk` which is more standard for "N parts".
    
    chunks = t.chunk(3, dim=0) # 3 chunks -> size 2 each
    assert len(chunks) == 3
    assert chunks[0].allclose(Tensor([1, 2]))
    assert chunks[2].allclose(Tensor([5, 6]))


# ---- Advanced Autograd ----

def test_autograd_reuse_graph():
    x = Tensor.new(2.0, requires_grad=True)
    
    # y = x * x + x
    # dy/dx = 2x + 1. At x=2 -> 5
    y = x * x + x 
    
    grads = y.backward()
    assert grads[x].allclose(Tensor.new(5.0))

def test_autograd_matmul():
    # A (1, 2), B (2, 1)
    a = Tensor([[1., 2.]], requires_grad=True)
    b = Tensor([[3.], [4.]], requires_grad=True)
    
    c = a @ b # (1, 1) -> 1*3 + 2*4 = 11
    
    grads = c.backward()
    
    # dc/da = b^T = [3, 4]
    assert grads[a].allclose(Tensor([[3., 4.]]))
    
    # dc/db = a^T = [[1], [2]]
    assert grads[b].allclose(Tensor([[1.], [2.]]))

def test_autograd_accumulate():
    x = Tensor.new(2.0, requires_grad=True)
    y = x * 2
    z = x * 3
    
    # w = y + z = 2x + 3x = 5x
    w = y + z
    
    grads = w.backward()
    assert grads[x].allclose(Tensor.new(5.0))
