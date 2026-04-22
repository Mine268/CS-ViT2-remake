鉴于LICENSE限制不能直接放出处理后的文件，请按照如下步骤预处理MANO模型文件

1. 对于原始的MANO\_LEFT/RIGHT.pkl文件，请将其中的所有成员转换为np.array类型避免出现错误
2. 请分别从MANO\_LEFT/RIGHT.pkl文件中提取hand\_components成员（形状为45*45）存储为mano\_lr\_pca.npz

```python
# mano_lr_pca.npz
{
    "left": np.array,
    "right": np.array
}
```

3. 从MANO\_RIGHT.pkl中提取hands\_mean为单独的mano\_right\_mean.npy文件