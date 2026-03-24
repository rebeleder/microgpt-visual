# MicroGPT.js

一个极简的、无依赖的纯 JavaScript (Node.js) GPT 实现，带有**交互式可视化系统**。

这是对 [Andrej Karpathy](https://github.com/karpathy) “原子级” Python（https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95 ） 实现的直接移植。仅使用 Node.js 标准库 + ES5 语法，演示 Transformer 的核心算法——自动求导、注意力机制和优化器。**单文件 ~456 行，每一行都能看懂。**

## 可视化预览
<img width="1530" height="436" alt="image" src="https://github.com/user-attachments/assets/dc65bd2e-9e92-4497-8630-0b275ff2da86" />
<img width="1534" height="689" alt="image" src="https://github.com/user-attachments/assets/8f5c1e3d-76db-419f-ba6b-da18427b7e9b" />
<img width="1538" height="740" alt="image" src="https://github.com/user-attachments/assets/b36fec94-9f77-42fa-82db-8b89a2cd44f8" />
<img width="1535" height="734" alt="image" src="https://github.com/user-attachments/assets/4150cd69-1975-420a-ac02-878d64343bf8" />
<img width="1542" height="741" alt="image" src="https://github.com/user-attachments/assets/799053e2-f747-4b09-b089-58d060b3ba35" />
<img width="1554" height="539" alt="image" src="https://github.com/user-attachments/assets/e5613ab6-0610-484a-879f-f0a3575dabab" />


## 🌟 特性

- **零依赖**：无需 TensorFlow、PyTorch 或任何 npm 包，仅用 Node.js 标准库
- **完整的自动求导引擎**：从零实现反向传播（`Value` 类）
- **生产级 GPT 架构**：
  - Token + 位置嵌入
  - 多头因果注意力（4 heads）
  - 层归一化 + 残差连接
  - 前馈网络 (MLP)
  - Adam 优化器（完整实现，包括 bias correction）
- **🎨 交互式可视化**：`--trace` 模式生成自包含的 `viz.html`
  - 6 个 Tab：词元化 → 嵌入 → 注意力 → 损失/梯度 → 训练曲线 → 推理
  - Neo-brutalism 设计，点击即可查看深层解释
  - 无需服务器，双击 HTML 即可打开
- **中文友好**：替换input.txt内容即可，支持多字节字符

## 🚀 快速开始

### 环境要求

- [Node.js](https://nodejs.org/) (任意版本)
- `curl` 命令（用于自动下载训练数据，Windows Git Bash 自带）

### 运行

**基础训练模式**：

```bash
node microgpt.js
```

输出示例：

```
num params: 15360
step 10/100, loss: 2.4532
step 20/100, loss: 2.1234
...
=== 推理 ===
seed: emma
generated: emma mia luna zoe ...
```

**🎨 可视化模式（推荐）**：

```bash
node microgpt.js --trace
```

训练完成后自动生成 `viz.html`，双击打开即可查看：

- **Tab 1（词元化）**：字符 → ID 映射，可搜索查看编码结果
- **Tab 2（嵌入）**：点击词元查看 64 维嵌入向量，配有"为什么乘以 16"的解释
- **Tab 3（注意力）**：4 个注意力头的热力图，理解 Q/K/V 计算和因果掩码
- **Tab 4（损失与梯度）**：交叉熵计算、随机猜测 baseline（2.77）、梯度流向
- **Tab 5（训练）**：Loss 曲线，Adam 学习率自适应说明
- **Tab 6（推理）**：温度采样过程，logits → 概率 → 采样


## 📖 代码结构

```
microgpt.js（单文件 ~456 行）
├── Value 类          # 自动求导引擎（add/mul/tanh/softmax + backward）
├── gpt()             # GPT 前向传播（嵌入 → N 层 Transformer → 输出）
├── backward()        # 拓扑排序 + 反向传播
├── adam()            # Adam 优化器（momentum + RMSProp + bias correction）
└── buildVizHtml()    # 可视化生成器（仅在 --trace 模式启用）
```

**关键优化（对比原始 Python 版本）**：

- `charToId` 从 O(n) indexOf 改为 O(1) 哈希表查找
- `backward()` 已访问集合从数组改为对象（O(1) 查找）
- 推理时使用纯数字 `sampleLogits()`，跳过 autograd 图构建（10x 提速）
- KV 缓存按层索引（`keys[li]`, `values[li]`），与 Python 对齐
- Loss 正确归一化（`(1/n) * sum(losses)`）
- Adam beta 幂次提到外层循环，避免重复计算

## 🎓 学习路径

1. **先运行 `--trace` 模式**，打开 `viz.html` 查看可视化
2. **阅读 `Value` 类**（microgpt.js:28-120），理解自动求导
3. **阅读 `gpt()` 函数**（microgpt.js:200-300），理解 Transformer 架构
4. **修改超参数实验**：
   - `nEmbd = 128` → 嵌入维度加倍
   - `nHead = 8` → 注意力头数加倍
   - `nLayer = 3` → 加深模型层数
   - `maxSteps = 200` → 训练更久

## 🔬 技术细节

### 自动求导引擎

```javascript
var a = new Value(2.0);
var b = new Value(3.0);
var c = a.mul(b);  // c.data = 6.0
c.backward();      // a.grad = 3.0, b.grad = 2.0
```

### 多头注意力

```javascript
// 4 个头，每头 16 维（64 / 4）
for (var h = 0; h < nHead; h++) {
  var Q = x.matmul(Wq[h]);  // (T, 16)
  var K = x.matmul(Wk[h]);  // (T, 16)
  var att = Q.matmul(K.T()).scale(1 / Math.sqrt(16));  // (T, T)
  att = causalMask(att).softmax();  // 上三角遮盖
  ...
}
```

### Adam 优化器

```javascript
// beta1 = 0.9, beta2 = 0.999
m = beta1 * m + (1 - beta1) * grad;  // momentum
v = beta2 * v + (1 - beta2) * grad^2;  // RMSProp
m_hat = m / (1 - beta1^step);  // bias correction
v_hat = v / (1 - beta2^step);
param -= lr * m_hat / (sqrt(v_hat) + 1e-8);
```

## 🐛 常见问题

**Q: 训练 loss 一直很高（> 2.5）？**
A: 正常，这是极简模型（15k 参数）。可以：

- 增加 `maxSteps` 到 200+
- 增加 `nEmbd` 到 128
- 增加 `nLayer` 到 3
- 调整学习率 `lr = 0.01`

**Q: 生成的名字很奇怪？**
A: 这是字符级模型，没有词汇表。可以调整 `temperature`：

- `temperature = 0.8` → 更保守
- `temperature = 1.5` → 更随机

**Q: 如何保存模型？**
A: 当前版本未实现持久化。可以手动保存 `params` 数组到 JSON：

```javascript
fs.writeFileSync('model.json', JSON.stringify(params.map(p => p.data)));
```

**Q: 可视化打开是空白？**
A: 确保：

- 使用了 `--trace` 参数
- 完整训练结束（不要中途 Ctrl+C）
- 浏览器允许本地文件访问（Chrome 可能需要 `--allow-file-access-from-files`）

## 🎯 为什么要做这个项目？

1. **教育价值**：不依赖黑盒框架，理解每一行代码
2. **ES5 语法**：没有 class/arrow/async，任何 JS 开发者都能读懂
3. **可移植性**：可轻松移植到浏览器、Deno、嵌入式 JS 引擎
4. **可视化**：交互式解释系统，帮助理解抽象概念（注意力、梯度）

## 📚 参考资料

- [Andrej Karpathy 的原始 Python 实现](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [Attention Is All You Need 论文](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

## 📝 License

MIT

---

**Made with ❤️ for learners who want to understand GPT from first principles.**
