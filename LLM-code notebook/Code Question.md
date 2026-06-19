# Code Question

日期：2026-06-16

上级：[[1.学习指南|LLM-Algo-LeetCode 学习指南]]

总览：[[03_LLM科研/00_总览/02_线上课总览/00_线上课总览|线上课总览]]

相关章节：[[2.00_PyTorch_Warmup|00_PyTorch_Warmup]]、[[3.01_RMSNorm|01_RMSNorm]]、[[4.02_SwiGLU|02_SwiGLU]]、[[5.03_RoPE|03_RoPE]]、[[6.04_Attention_MHA_GQA|04_Attention_MHA_GQA]]

标签：#LLM-Algo-LeetCode #PyTorch #代码细节 #CodeQuestion

## 使用规则

这个文件只放代码细节问题，不放每节主线知识。

以后遇到这些内容，优先补到这里：

```text
Python 语法
PyTorch API
shape 变化
dtype / device
广播
矩阵乘法
自定义 autograd 细节
容易写错的一行代码
```

主笔记只保留本节的核心结构、公式和 shape 主线。

## 快速口诀

```text
新建张量看 device。
敏感计算用 float32。
返回结果看原 dtype。
shape 不懂先写每一维含义。
矩阵乘法看内维是否对齐。
逐元素操作看是否能广播。
```

## Python 语法

### `x.shape[:-1]` 是什么

来源：[[5.03_RoPE|03_RoPE]]

```python
x.shape = (2, 16, 4, 64)
x.shape[:-1]
```

结果：

```text
(2, 16, 4)
```

意思是取除了最后一维以外的所有维度。

常用于：

```python
x.reshape(*x.shape[:-1], -1, 2)
```

含义：

```text
前面的 batch、seq、head 不动，只改最后一维。
```

### `*x.shape[:-1]` 是什么

来源：[[5.03_RoPE|03_RoPE]]

`*` 在这里是 Python 解包。

```python
x.shape[:-1] = (2, 16, 4)
```

那么：

```python
reshape(*x.shape[:-1], -1, 2)
```

等价于：

```python
reshape(2, 16, 4, -1, 2)
```

### `-1` 在 `reshape` 里是什么意思

来源：[[5.03_RoPE|03_RoPE]]

`-1` 表示让 PyTorch 自动推断这一维。

例如：

```python
x.shape = (2, 16, 4, 64)
x.reshape(2, 16, 4, -1, 2)
```

最后一维 `64` 被拆成：

```text
32 * 2
```

所以 `-1` 自动变成 `32`。

### tuple 里一个元素为什么要加逗号

来源：00 Autograd 相关问题

```python
output, = ctx.saved_tensors
```

这里右边是一个 tuple，即使里面只有一个张量，也要按 tuple 解包。

```python
(output)
```

只是括号，不是 tuple。

```python
(output,)
```

才是只有一个元素的 tuple。

所以：

```python
output, = ctx.saved_tensors
```

意思是：

```text
从 saved_tensors 这个 tuple 里取出唯一一个张量，赋值给 output。
```

## PyTorch 通用 API

### `torch` / `nn` / `F` 怎么分

来源：[[4.02_SwiGLU|02_SwiGLU]]

| 类型 | 位置 | 例子 |
|---|---|---|
| 通用 Tensor 操作 | `torch.xxx` | `torch.chunk`, `torch.outer`, `torch.sqrt`, `torch.rsqrt` |
| 神经网络模块 / 层 | `nn.xxx` | `nn.Linear`, `nn.Embedding`, `nn.Module` |
| 神经网络函数 | `F.xxx` | `F.silu`, `F.relu`, `F.softmax`, `F.linear` |

简单规则：

```text
要注册成模型层，用 nn。
forward 里临时算一下，用 F。
通用张量数学和切分拼接，用 torch。
```

### `torch.chunk`

来源：[[4.02_SwiGLU|02_SwiGLU]]

```python
torch.chunk(input, chunks, dim=0)
```

作用：

```text
沿指定维度，把 Tensor 尽量平均切成 chunks 份。
```

例子：

```python
gate, up = torch.chunk(gate_up, 2, dim=-1)
```

如果：

```text
gate_up: (2, 10, 22016)
```

那么：

```text
gate: (2, 10, 11008)
up:   (2, 10, 11008)
```

`dim=-1` 表示最后一维。

### `int(...)`

来源：[[4.02_SwiGLU|02_SwiGLU]]

```python
int(4096 * 8 / 3)
```

`int` 是直接截断小数，不是四舍五入。

```text
10922.66 -> 10922
```

### `nn.Linear(..., bias=False)`

来源：[[4.02_SwiGLU|02_SwiGLU]]

`bias=False` 可以不写吗？

```text
不可以随便省略。
```

因为：

```python
nn.Linear(in_features, out_features)
```

默认：

```python
bias=True
```

如果模型结构要求没有 bias，必须显式写：

```python
nn.Linear(..., bias=False)
```

### `nn.Parameter`

来源：[[3.01_RMSNorm|01_RMSNorm]]

普通 Tensor 不会自动被优化器更新：

```python
self.weight = torch.ones(hidden_size)
```

要让它成为可训练参数，需要：

```python
self.weight = nn.Parameter(torch.ones(hidden_size))
```

这样它才会出现在：

```python
model.parameters()
```

一句话：

```text
nn.Parameter = 把 Tensor 注册成模型的可训练参数。
```

## dtype 和 device

### 什么时候要转 `float32`

来源：[[3.01_RMSNorm|01_RMSNorm]]、[[5.03_RoPE|03_RoPE]]

常见触发条件：

```text
pow
mean
sum
softmax
exp
norm
rsqrt
复数转换
```

这些操作可能对精度敏感，尤其输入是 `float16` / `bfloat16` 时。

RMSNorm 例子：

```python
x_fp32 = x.float()
variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
```

RoPE 例子：

```python
xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
```

口诀：

```text
统计量、指数、归一化、复数转换，优先 float32 算。
算完再转回原 dtype。
```

### 什么时候要写 `device=...`

来源：[[5.03_RoPE|03_RoPE]]

只要在 forward 或函数里新建 Tensor，就要想它应该放在哪个设备上：

```python
torch.arange(...)
torch.ones(...)
torch.zeros(...)
```

RoPE 例子：

```python
t = torch.arange(end, device=freqs.device, dtype=torch.float32)
```

因为后面 `t` 要和 `freqs` 做运算，所以 `t` 要跟 `freqs` 在同一个 device。

否则 GPU 上可能报错：

```text
Expected all tensors to be on the same device
```

口诀：

```text
新建张量，要贴着后面要相乘/相加的那个张量建。
```

### `.to()` 不只是转类型

来源：[[3.01_RMSNorm|01_RMSNorm]]

常见写法：

```python
x.to(torch.float32)
x.to(torch.float16)
x.to("cuda")
x.to("cpu")
x.to(device="cuda", dtype=torch.float16)
```

本质：

```text
.to() 可以转 dtype，也可以转 device。
```

### `.type_as(x)` 是什么

来源：[[5.03_RoPE|03_RoPE]]

```python
return xq_out.type_as(xq)
```

意思是：

```text
把 xq_out 转成和 xq 一样的 dtype。
```

常用于中间升精度后，输出还原给调用方。

## Shape 和广播

### `keepdim=True`

来源：[[3.01_RMSNorm|01_RMSNorm]]

```python
variance = x.pow(2).mean(dim=-1, keepdim=True)
```

如果：

```text
x: (4, 10, 768)
```

那么：

```text
variance: (4, 10, 1)
```

`keepdim=True` 的作用：

```text
把被压缩的维度保留下来，大小变成 1，方便后面广播。
```

如果不保留，可能变成：

```text
(4, 10)
```

后面和 `(4, 10, 768)` 广播时更容易出错。

### `permute` 和 `reshape`

来源：[[2.00_PyTorch_Warmup|00_PyTorch_Warmup]]

```python
x_native = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
```

如果：

```text
x: (batch, channels, height, width)
```

先：

```text
permute(0, 2, 3, 1)
```

得到：

```text
(batch, height, width, channels)
```

再：

```text
reshape(batch, height * width, channels)
```

得到：

```text
(batch, height * width, channels)
```

一句话：

```text
permute 改维度顺序；reshape 合并或拆分维度。
```

### `reshape` 能不能替代 `transpose`

来源：[[6.04_Attention_MHA_GQA|04_Attention_MHA_GQA]]

一句话答案：

```text
reshape 可以替代 view，但不能替代 transpose。
```

例子：

```python
output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
```

如果此时：

```text
output: [B, H, S, D]
```

目标是：

```text
[B, H, S, D]
-> [B, S, H, D]
-> [B, S, H * D]
```

不能直接：

```python
output = output.reshape(batch_size, seq_len, -1)
```

因为：

```text
reshape 只负责拆分/合并维度；
transpose 才负责交换维度顺序。
```

### 多头注意力切的是哪个维度

来源：[[6.04_Attention_MHA_GQA|04_Attention_MHA_GQA]]

结论：

```text
标准 MHA/GQA 切的是 hidden_dim，不是 seq_len。
```

例子：

```text
x: [B, S, hidden_dim]
hidden_dim = H * D
```

切头：

```text
[B, S, hidden_dim]
-> [B, S, H, D]
-> [B, H, S, D]
```

`S` 不变，意思是：

```text
每个 head 仍然能看完整的 S 个 token。
```

### `reshape_for_broadcast`

来源：[[5.03_RoPE|03_RoPE]]

```python
def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)
```

如果：

```text
xq_:       (2, 16, 4, 32)
freqs_cis: (16, 32)
```

目标：

```text
freqs_cis -> (1, 16, 1, 32)
```

这样能广播到：

```text
(2, 16, 4, 32)
```

含义：

```text
seq 维和特征组维保留；
batch 和 head 维变成 1，用来广播复制。
```

### `flatten(3)`

来源：[[5.03_RoPE|03_RoPE]]

```python
y.shape = (2, 16, 4, 32, 2)
y.flatten(3)
```

`flatten(3)` 表示从第 3 维开始，把后面所有维度压平。

维度编号：

```text
0: 2
1: 16
2: 4
3: 32
4: 2
```

前 3 维不动：

```text
2, 16, 4
```

后面合并：

```text
32 * 2 = 64
```

结果：

```text
(2, 16, 4, 64)
```

## 矩阵乘法和逐元素乘法

### `@` 和 `*` 怎么分

来源：[[2.00_PyTorch_Warmup|00_PyTorch_Warmup]]

```text
@ 是矩阵乘法。
* 是逐元素乘法，也会走广播规则。
```

Linear 反向：

```python
grad_x = grad_z @ weight
grad_weight = grad_z.T @ x
```

ReLU 反向：

```python
grad_z = grad_output * mask
```

原因：

```text
Linear 是矩阵乘法，所以反向还是矩阵乘法。
ReLU 是逐元素函数，所以反向是逐元素乘法。
```

### `Q @ K.T` 到底让 Q 看到了什么

来源：[[6.04_Attention_MHA_GQA|04_Attention_MHA_GQA]]

一句话答案：

```text
Q 不是直接读取 K 的内容，而是和每个 K 做点积匹配，得到 token-token 分数表。
```

单个 head：

```text
Q: [S_q, D]
K: [S_k, D]
K.T: [D, S_k]

Q @ K.T -> [S_q, S_k]
```

其中：

```text
scores[i, j] = q_i · k_j
```

含义：

```text
第 i 个 query token 对第 j 个 key token 的匹配分数。
```

容易误解的点：

```text
计算时用 D 维特征做点积；
结果不是 hidden-hidden 表，而是 token-token 表。
```

### `scores` 是概率吗

来源：[[6.04_Attention_MHA_GQA|04_Attention_MHA_GQA]]

不是。

```python
scores = Q @ K.transpose(-2, -1)
```

得到的是 raw scores / logits。

需要：

```python
probs = torch.softmax(scores, dim=-1)
```

之后才是注意力权重。

流程：

```text
scores = 匹配分数
probs = 看每个 token 的比例
output = probs @ V
```

### `probs @ V` 为什么消掉的是 `S_k`

来源：[[6.04_Attention_MHA_GQA|04_Attention_MHA_GQA]]

形状：

```text
probs: [B, H, S_q, S_k]
V:     [B, H, S_k, D]

probs @ V -> [B, H, S_q, D]
```

原因：

```text
这是对所有被看的 token 做加权求和。
S_k 是被加权汇总的 token 数，所以会被消掉。
D 是每个 value 的内容维度，所以保留下来。
```

### bias 梯度为什么是 `sum(dim=0)`

来源：[[2.00_PyTorch_Warmup|00_PyTorch_Warmup]]

前向：

```python
z = x @ weight.T + bias
```

如果：

```text
z:    (4, 5)
bias: (5,)
```

同一个 bias 会加到 batch 里的每一行。

所以反向时要把 batch 维贡献加起来：

```python
grad_bias = grad_z.sum(dim=0)
```

不是：

```python
mask.sum(dim=0)
```

因为 bias 的梯度要看真正传到 `z` 的梯度大小，也就是 `grad_z`。

## `torch.outer`

### `outer` 和 `*` 有什么区别

来源：[[5.03_RoPE|03_RoPE]]

结论：

```text
* 是按位置相乘 / 广播相乘。
outer 是所有元素两两相乘，生成二维表。
```

例子：

```python
t = torch.tensor([0, 1, 2])      # shape (3,)
freqs = torch.tensor([1, 0.1])   # shape (2,)
```

```python
torch.outer(t, freqs)
```

结果：

```text
[
  [0*1, 0*0.1],
  [1*1, 1*0.1],
  [2*1, 2*0.1],
]
```

shape：

```text
(3, 2)
```

而：

```python
t * freqs
```

因为 `(3,)` 和 `(2,)` 对不上，会报错。

如果想用 `*` 做出 outer，需要手动升维：

```python
t[:, None] * freqs[None, :]
```

它和下面等价：

```python
torch.outer(t, freqs)
```

### `outer` 是叉乘吗

来源：[[5.03_RoPE|03_RoPE]]

不是。

```python
torch.outer
```

是外积：

```text
一维向量 x 一维向量 -> 二维表
```

三维几何里的叉乘是：

```python
torch.cross
```

它的结果还是一个向量。

### 为什么是 `torch.outer(t, freqs)`，不能换顺序

来源：[[5.03_RoPE|03_RoPE]]

RoPE 后面希望旋转表是：

```text
(seq, dim/2)
```

其中：

```text
第 0 维是 token 位置
第 1 维是特征组
```

所以：

```python
torch.outer(t, freqs)
```

如果换成：

```python
torch.outer(freqs, t)
```

shape 会变成：

```text
(dim/2, seq)
```

信息没丢，但轴反了，后面没法直接 reshape 成：

```text
(1, seq, 1, dim/2)
```

如果真的写反，需要转置回来：

```python
torch.outer(freqs, t).T
```

## KV Cache 和 GQA 细节

### `torch.cat([k_cache, xk], dim=2)` 在拼什么

来源：[[6.04_Attention_MHA_GQA|04_Attention_MHA_GQA]]

这里拼的是 token 维。

如果：

```text
k_cache: [B, H_kv, 5, D]
xk:      [B, H_kv, 1, D]
```

那么：

```python
xk = torch.cat([k_cache, xk], dim=2)
```

得到：

```text
xk: [B, H_kv, 6, D]
```

直观：

```text
旧 K: [token0, token1, token2, token3, token4]
新 K: [token5]

拼接后：
[token0, token1, token2, token3, token4, token5]
```

### `torch.cat` 和 `torch.concat`

来源：[[6.04_Attention_MHA_GQA|04_Attention_MHA_GQA]]

在 PyTorch 里常用写法是：

```python
torch.cat([a, b], dim=...)
```

`torch.concat` 也可以用，基本等价，但教程和源码里更常见的是 `torch.cat`。

### `repeat_kv` 为什么不是整体复制

来源：[[6.04_Attention_MHA_GQA|04_Attention_MHA_GQA]]

GQA 要的是分组共享。

如果：

```text
H = 4
H_kv = 2
n_rep = 2
```

目标是：

```text
[KV0, KV1] -> [KV0, KV0, KV1, KV1]
```

而不是：

```text
[KV0, KV1] -> [KV0, KV1, KV0, KV1]
```

前者表示：

```text
Q0 Q1 -> KV0
Q2 Q3 -> KV1
```

后者是交错共享，不是这里的 grouped query。

### `hidden_states[:, :, None, :, :]` 为什么 `None` 放在这里

来源：[[6.04_Attention_MHA_GQA|04_Attention_MHA_GQA]]

原始：

```text
hidden_states: [B, H_kv, S, D]
```

插入新维度：

```python
hidden_states[:, :, None, :, :]
```

变成：

```text
[B, H_kv, 1, S, D]
```

然后：

```python
expand(B, H_kv, n_rep, S, D)
```

直观上：

```text
[
  [KV0, KV0],
  [KV1, KV1],
]
```

最后：

```python
reshape(B, H_kv * n_rep, S, D)
```

得到：

```text
[KV0, KV0, KV1, KV1]
```

### 为什么先拼接 cache 再 `repeat_kv`

来源：[[6.04_Attention_MHA_GQA|04_Attention_MHA_GQA]]

正确顺序：

```text
旧 cache + 当前 K/V
-> 得到完整但小的 K/V: [B, H_kv, S, D]
-> 临时 repeat 成计算用 K/V: [B, H, S, D]
```

原因：

```text
KV Cache 里应该存小的 H_kv 版本。
如果先 repeat 再存，cache 会变成 H 个 head，GQA 就不省显存了。
```

### prefill 和 decode

来源：[[6.04_Attention_MHA_GQA|04_Attention_MHA_GQA]]

简单记法：

```text
prefill = 读题阶段，一次性处理 prompt，建立初始 KV Cache。
decode = 写答案阶段，每次生成 1 个新 token，并追加 KV Cache。
```

例子：

```text
prompt 长度 = 10
prefill 后 K cache 的 S 维 = 10

decode 又生成 3 个 token
K cache 的 S 维 = 13
```

## 复数相关

### `torch.ones_like(freqs)`

来源：[[5.03_RoPE|03_RoPE]]

```python
torch.ones_like(freqs)
```

意思是：

```text
照着 freqs 的形状，生成一个全 1 张量。
```

在 RoPE 里：

```python
torch.polar(torch.ones_like(freqs), freqs)
```

第一个参数是半径，第二个参数是角度。

半径全 1 表示：

```text
只旋转，不拉伸。
```

### `torch.polar`

来源：[[5.03_RoPE|03_RoPE]]

```python
torch.polar(abs, angle)
```

意思是用极坐标生成复数：

```text
abs * (cos(angle) + i sin(angle))
```

RoPE 中：

```python
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
```

含义：

```text
把旋转角度表变成复数旋转因子。
```

### `view_as_complex`

来源：[[5.03_RoPE|03_RoPE]]

```python
torch.view_as_complex(x)
```

要求：

```text
x 的最后一维必须是 2。
```

它会把最后两个数看作：

```text
[实部, 虚部]
```

例如：

```text
[a, b] -> a + bi
```

RoPE 中：

```text
(2, 16, 4, 32, 2)
-> (2, 16, 4, 32)
```

最后的 `2` 藏进复数 dtype 里。

### `view_as_real`

来源：[[5.03_RoPE|03_RoPE]]

```python
torch.view_as_real(x_complex)
```

作用是把复数拆回两个实数：

```text
a + bi -> [a, b]
```

RoPE 中：

```text
(2, 16, 4, 32)
-> (2, 16, 4, 32, 2)
```

注意：

```text
这是拆回复数的实部和虚部，不是放大。
```

## Autograd 细节

### `ctx.save_for_backward`

来源：[[2.00_PyTorch_Warmup|00_PyTorch_Warmup]]

```python
ctx.save_for_backward(x, weight, mask)
```

作用：

```text
把 backward 需要用到的张量存起来。
```

原则：

```text
forward 里算出来，但 backward 还要用的张量，才需要保存。
```

例如 Linear + ReLU：

```python
grad_z = grad_output * mask
grad_x = grad_z @ weight
grad_weight = grad_z.T @ x
```

所以要存：

```text
x, weight, mask
```

### `grad_output` 是什么

来源：[[2.00_PyTorch_Warmup|00_PyTorch_Warmup]]

`grad_output` 不是输出值本身。

```text
grad_output = dL/dy
```

也就是：

```text
上一层传回来的梯度。
```

如果当前层是：

```text
z -> y
```

那么：

```python
grad_z = grad_output * dy_dz
```

在 ReLU 中：

```python
dy_dz = mask
grad_z = grad_output * mask
```

### `mask` 为什么不是普通遮罩

来源：[[2.00_PyTorch_Warmup|00_PyTorch_Warmup]]

```python
mask = (z > 0).float()
```

在 ReLU 反向里，它本质上是局部导数：

```text
z > 0  -> dy/dz = 1
z <= 0 -> dy/dz = 0
```

所以：

```python
grad_z = grad_output * mask
```

不是为了“屏蔽负数输出”这么简单，而是在做链式法则。

## 未来追加模板

以后新增问题按这个格式追加：

````markdown
### 问题标题

来源：[[对应章节]]

一句话答案：

```text
...
```

最小例子：

```python
...
```

易错点：

```text
...
```
````
