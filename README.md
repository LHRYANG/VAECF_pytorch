# VAECF_pytorch
This is an pytorch implementation for Variational autoencoders for collaborative filtering

- data/: 将 ml-20 解压到当前文件夹
- model.py: VAE 模型
- process.py: 数据处理
- dataloader.py: 加载数据
- train.py: 训练
- validation.py: 测试
- evaluation.py: 包括评价方法的实现 

##Result
Recall@20 大约 0.34 原始实现 0.39
Recall@100 大约0.49 原始实现 0.53 

效果和原始论文还差不少.
