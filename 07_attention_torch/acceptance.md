# Attention 概念验收（不含答案）

请在不看实现代码的情况下回答。共 30 题，至少答对 26 题才通过。

## 概念题

1. Q、K、V 各自表示什么？它们在“检索”视角下分别扮演什么角色？
2. 当 Q 为 `[B,H,Sq,Dh]`、K 为 `[B,H,Sk,Dh]` 时，为什么 `QK^T` 得到 `[B,H,Sq,Sk]`？
3. `scores[b,h,i,j]` 表示什么？
4. `QK^T` 中哪个维度是 dot product 的 reduction 维度？
5. 为什么 softmax 应沿最后一维 `Sk`？
6. 为什么不能沿 query 维 `Sq` 做 softmax？
7. 为什么 scaled dot-product attention 使用 `sqrt(Dh)` 缩放？
8. 多头 Attention 中为什么不是除以 `sqrt(D_model)`？
9. 为什么 B 和 H 可以看成彼此独立的外层任务维？
10. 为什么一个 batch 内会得到一个或多个 `S×S` 矩阵？
11. 从 batched matmul 的 shape 规则解释：为什么不同 batch 样本不会互相 attention？
12. causal mask 为什么必须在 softmax 前应用？
13. 被 mask 的 score 为什么使用 `-inf` 或足够小的值，而不是 0？
14. 采用左上角对齐的 causal mask 时，第 i 个 query 可以看到哪些 key？
15. 拆分 heads 时，reshape 和 transpose 分别完成了什么？
16. 为什么单纯 reshape 不能替代 `[B,S,H,Dh] → [B,H,S,Dh]` 的 transpose？
17. transpose 后 tensor 为什么可能是 non-contiguous？
18. `view` 和 `reshape` 在 non-contiguous tensor 上有什么潜在区别？
19. 多个 head 的结果如何按正确 layout 重新合并？
20. concat heads 后为什么还需要 `out_proj`？
21. 普通 Attention 通常会显式物化哪些 `S×S` tensor？
22. score 的理论内存为什么是 `O(BHS²)`？dtype 会如何改变常数项？
23. `QK^T` 的计算量为什么约为 `O(BHS²Dh)`？
24. B、H、dtype 不变时，S 翻倍为什么 score 内存约变成四倍？
25. FlashAttention 主要准备解决普通 Attention 的哪个系统问题？它不是在改变哪个数学结果？

## Shape 推演题

26. `B=2, Sq=3, Sk=5, Dh=8, Dv=6` 的单头 Attention 中，写出 `K^T`、scores、probs 和 out 的 shape。
27. `B=2, S=7, D_model=16, H=4` 时，写出拆 head 后 Q 的两个连续 layout、每头 score、每头输出及最终合并输出的 shape。
28. `B=3, S=16, D_model=32, H=8` 时，`head_dim` 是多少？score 和 attention probabilities 各有多少个元素？
29. 自注意力输入 `[1,4,8]` 经三个 Linear 投影、拆为 2 heads、完成 Attention、合并 heads、再经 out projection：逐步写出每个中间 shape。
30. cross-attention 中 Q 为 `[2,3,8]`、K/V 为 `[2,5,8]`，拆为 2 heads 后，写出 Q/K/V、scores、probs 与合并输出的 shape，并指出 softmax 所在维度。

