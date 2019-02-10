# nnstocktrain
This is a project to train neural network with tensorflow for predict stock market shares， 这个是 利用 tensorflow 来 训练 神经网络，预测股价变化的 程序


神经 网络的 输入 ：
今天 开盘价 （比昨天的涨跌百分数，用小数表示）， 今天 实际开盘价 ， 前 n天的  以下各项（程序中设置n为30）：
开盘价百分数，  收盘价百分数，最高价百分数 ， 最低价百分数， 本日量 与前30日均量的比率

隐藏层： 一层  300个神经元  激活函数  tanh  

输出层   一个神经元 ，激活函数  tanh,  输出值   本日开盘价 买入， 明日开盘价卖出 收益 百分数 （用小数表示）

