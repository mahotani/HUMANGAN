# 人間GAN

**link:**
http://sython.org/papers/SP/fujii20otogaku.pdf

## 1. はじめに

機械学習による生成モデルは、メディア処理に関連する研究に強く貢献している。  
DNNは強力な非線形変換の恩恵を受けて複雑なデータ分布をモデル化できる。  
GANは画像、音声、言語などの多様な分野で成功例が存在する。  


しかし、GANは実在するデータ(学習データ)の分布しか表現することができない。  
-> 一方、人間の知覚は実在するメディアから逸脱に対してある程度の許容範囲がある。  
膨大なデータを利用しても、通常のGANでは、知覚分布を表現することが不可能。  
-> 人間の知覚分布を表現することを目標とする。  
- 人間を生成データに対して事後確率の許容度を出力するblack-boxシステムとみなし、人間の知覚評価を利用して生成モデルを学習する。  


通常のGANは、DNNのDiscriminatorを使うのに対して、人間GANはDiscriminatorとしてクラウドソーシングサービスの人的リソースから得られる知覚評価を詐称する。  


人間GANの性能評価については、音声の自然性(すなわち、人間が人間GANによって生成された音声を人間の音声としてどの程度許容できるか)における実験的評価を行い  
1. 実在データ分布と知覚分布が異なること
1. 提案する人間GANは通常のGANでは表現できない知覚分布を適切に表現できること
を示す。  

## 2. 通常のGAN

通常のGANの目的は、生成モデルの表現する分布を学習に使う実際のデータの分布に一致させること。  
-> 通常のGANは、データを生成する生成モデル(Generator)G(・)と、実在データと生成データを識別する識別モデル(Discriminator)D(・)を用いる。  
(G(・)とD(・)はDNN。)  

<img width="300" alt="GAN" src="https://user-images.githubusercontent.com/39772824/94514638-af866780-025c-11eb-9899-aa09c6eceb58.png">

N個の実在データを
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;{\bf&space;x}&space;=&space;{x_1,&space;\cdots&space;.&space;x_n&space;,&space;\cdots&space;,&space;x_N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;{\bf&space;x}&space;=&space;{x_1,&space;\cdots&space;.&space;x_n&space;,&space;\cdots&space;,&space;x_N}" title="{\bf x} = {x_1, \cdots . x_n , \cdots , x_N}" /></a>
とする。  
生成モデルG(・)は、既知の確率分布(例えば、一様分布)に従うN個の乱数
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;{\bf&space;z}&space;=&space;{z_1,&space;\cdots&space;.&space;z_n&space;,&space;\cdots&space;,&space;z_N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;{\bf&space;z}&space;=&space;{z_1,&space;\cdots&space;.&space;z_n&space;,&space;\cdots&space;,&space;z_N}" title="{\bf z} = {z_1, \cdots . z_n , \cdots , z_N}" /></a>
を生成データ
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;{\bf&space;\hat{x}}&space;=&space;{\hat{x}_1,&space;\cdots&space;.&space;\hat{x}_n&space;,&space;\cdots&space;,&space;\hat{x}_N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;{\bf&space;\hat{x}}&space;=&space;{\hat{x}_1,&space;\cdots&space;.&space;\hat{x}_n&space;,&space;\cdots&space;,&space;\hat{x}_N}" title="{\bf \hat{x}} = {\hat{x}_1, \cdots . \hat{x}_n , \cdots , \hat{x}_N}" /></a>
に写像する。  

識別モデルD(・)は実在データ
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;x_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;x_n" title="x_n" /></a>
もしくは生成データ
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\hat{x}&space;_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\hat{x}&space;_n" title="\hat{x} _n" /></a>
を入力し、入力が実在データである事後確率を出力する。  

学習時の目的関数
<a href="https://www.codecogs.com/eqnedit.php?latex=V(&space;\cdot&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V(&space;\cdot&space;)" title="V( \cdot )" /></a>
は次の式。  
<a href="https://www.codecogs.com/eqnedit.php?latex=V(G,&space;D)&space;=&space;\sum^N_{n=1}&space;log&space;D(x_n)&space;&plus;&space;\sum^N&space;_{n=1}&space;log(1&space;-&space;D(G(Z_n)))&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;(1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V(G,&space;D)&space;=&space;\sum^N_{n=1}&space;log&space;D(x_n)&space;&plus;&space;\sum^N&space;_{n=1}&space;log(1&space;-&space;D(G(Z_n)))&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;(1)" title="V(G, D) = \sum^N_{n=1} log D(x_n) + \sum^N _{n=1} log(1 - D(G(Z_n))) \ \ \ \ \ \ \ \ (1)" /></a>

### 2.1. 生成モデルの学習

