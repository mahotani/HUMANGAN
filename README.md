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

学習時の目的関数V(・)は次の式。  
<a href="https://www.codecogs.com/eqnedit.php?latex=V(G,&space;D)&space;=&space;\sum^N_{n=1}&space;log&space;D(x_n)&space;&plus;&space;\sum^N&space;_{n=1}&space;log(1&space;-&space;D(G(Z_n)))&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;(1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V(G,&space;D)&space;=&space;\sum^N_{n=1}&space;log&space;D(x_n)&space;&plus;&space;\sum^N&space;_{n=1}&space;log(1&space;-&space;D(G(Z_n)))&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;(1)" title="V(G, D) = \sum^N_{n=1} log D(x_n) + \sum^N _{n=1} log(1 - D(G(Z_n))) \ \ \ \ \ \ \ \ (1)" /></a>

### 2.1. 生成モデルの学習

生成モデルG(・)は、式(1)を最小化するように学習される。  
G(・)のモデルパラメータを
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_G" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_G" title="\theta_G" /></a>
とすると、
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_G" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_G" title="\theta_G" /></a>
は次の式で推定される。

<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_G&space;=&space;\underset{\theta_G}{argmin}&space;\sum^N_{n=1}&space;log&space;(1&space;-&space;D(G(z_n)))&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;(2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_G&space;=&space;\underset{\theta_G}{argmin}&space;\sum^N_{n=1}&space;log&space;(1&space;-&space;D(G(z_n)))&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;(2)" title="\theta_G = \underset{\theta_G}{argmin} \sum^N_{n=1} log (1 - D(G(z_n))) \ \ \ \ \ \ \ \ (2)" /></a>

すなわち、G(・)は、D(・)が生成データを実在データとして識別するように学習される。  

### 2.2. 式別モデルの学習

識別モデルD(・)は、式(1)を-1倍した関数を最小化するように学習される。  
D(・)のモデルパラメータを
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_&space;D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_&space;D" title="\theta_ D" /></a>
とすると、
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_&space;D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_&space;D" title="\theta_ D" /></a>
は次の式で推定される。  

<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_&space;D&space;=&space;\underset{\theta_D}{argmax}&space;-V(G,&space;D)&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;(3)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_&space;D&space;=&space;\underset{\theta_D}{argmax}&space;-V(G,&space;D)&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;(3)" title="\theta_ D = \underset{\theta_D}{argmax} -V(G, D) \ \ \ \ \ \ \ \ (3)" /></a>

### 2.3. 問題点

通常のGANは実在データのようなものを作ることができるが、人間の知覚は実在データ以上に広いので、人間の知覚の範囲内全てで生成することが難しい。  

## 3. 人間GAN

人間GANは通常のGANの識別モデルを人間の知覚評価で置換した手法。  

<img width="300" alt="人間GAN" src="https://user-images.githubusercontent.com/39772824/94516671-48b77d00-0261-11eb-8941-1ee2582024e6.png">

G(・)がDNNを使って作られることや、入力に既知の確率分布に従う乱数を使うことは通常のGANと同様。  
一方で、通常のGANではD(・)には人間の知覚評価を使っている。  
このD(・)は、G(・)から生成された
<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{x}_n" title="\hat{x}_n" /></a>
を入力とし、この入力が「どの程度許容できるか」を0から1の値で事後確率として出力する。  
学習時の目的関数V(・)は次の式になる。  

<a href="https://www.codecogs.com/eqnedit.php?latex=V(G,&space;D)&space;=&space;\sum^N_{n=1}&space;D(G(z_n))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V(G,&space;D)&space;=&space;\sum^N_{n=1}&space;D(G(z_n))" title="V(G, D) = \sum^N_{n=1} D(G(z_n))" /></a>

人間GANでは学習過程には実在データを使用しない。

### 3.1. 生成モデルの学習

G(・)のモデルパラメータ
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_G" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_G" title="\theta_G" /></a>
は、式(4)を最大化するように学習される。  
人間GANでは勾配法による反復学習法を考える。  
すなわち、
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_G" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_G" title="\theta_G" /></a>
は次の式で反復的に更新される。  

<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_G&space;^{(new)}&space;=&space;\theta_G&space;&plus;&space;\alpha&space;\frac{\partial&space;V(G,&space;D)}{\partial&space;\theta_G}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_G&space;^{(new)}&space;=&space;\theta_G&space;&plus;&space;\alpha&space;\frac{\partial&space;V(G,&space;D)}{\partial&space;\theta_G}" title="\theta_G ^{(new)} = \theta_G + \alpha \frac{\partial V(G, D)}{\partial \theta_G}" /></a>

この時、
<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a>
は学習係数を示す。  

D(・)が人間による知覚評価なので、微分不可能であるため、「生成データに対して事後確率分布を出力するblack-boxシステム」とみなしている。  

以下の手順で最適化を行っている。  

1. 生成データ<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{x}_n" title="\hat{x}_n" /></a>に対し、正規分布<a href="https://www.codecogs.com/eqnedit.php?latex=N(0,&space;\sigma^2&space;I)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(0,&space;\sigma^2&space;I)" title="N(0, \sigma^2 I)" /></a>からランダムに生成した摂動<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;x_n^{(r)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;x_n^{(r)}" title="\Delta x_n^{(r)}" /></a>を付与する。
1. 摂動後の2つのデータ<a href="https://www.codecogs.com/eqnedit.php?latex=\{&space;\hat{x}_n&space;&plus;&space;\Delta&space;x_n^{(r)}&space;,&space;\hat{x}_n&space;-&space;\Delta&space;x_n-{(r)}&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{&space;\hat{x}_n&space;&plus;&space;\Delta&space;x_n^{(r)}&space;,&space;\hat{x}_n&space;-&space;\Delta&space;x_n-{(r)}&space;\}" title="\{ \hat{x}_n + \Delta x_n^{(r)} , \hat{x}_n - \Delta x_n-{(r)} \}" /></a>を評価者に提示する。
1. それらのデータの差分を解答させる。
1. 2~3をある生成データ<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{x}_n" title="\hat{x}_n" /></a>に対してR回繰り返す。

N個の生成データに対して上記の過程を繰り返す。  

評価者に解答させる差分は以下の式に表される。

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;D&space;(\hat{x}_n^{(r)})&space;\equiv&space;D(\hat{x}_n&space;&plus;&space;\Delta&space;x_n^{(r)})&space;-&space;D&space;(\hat{x}_n&space;-&space;\Delta&space;x_n^{(r)})&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;(6)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;D&space;(\hat{x}_n^{(r)})&space;\equiv&space;D(\hat{x}_n&space;&plus;&space;\Delta&space;x_n^{(r)})&space;-&space;D&space;(\hat{x}_n&space;-&space;\Delta&space;x_n^{(r)})&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;(6)" title="\Delta D (\hat{x}_n^{(r)}) \equiv D(\hat{x}_n + \Delta x_n^{(r)}) - D (\hat{x}_n - \Delta x_n^{(r)}) \ \ \ \ \ \ \ \ (6)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;D&space;(\hat{x}_n^{(r)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;D&space;(\hat{x}_n^{(r)})" title="\Delta D (\hat{x}_n^{(r)})" /></a>
は-1から1までの値をとる。  

### 3.2. 考察

人間GANは、「コンピュータによって解くことが困難な課題を、人間の処理能力を利用して解決すること」であるヒューマンコンピュテーションを利用したものであり、人間参加型(human-in-the-loop)の機械学習技術とみなされる。  

人間の知覚をDNNに組み込む人間参加型の技術はすでにいくつか存在している。  

### 3.3. 課題

- 生成モデルの初期化

人間GANの生成モデルは事後確率を最大化するように学習される。  
故に、初期の生成モデルから生成されるデータが知覚分布のmode近傍のみに分布する場合、学習を進めるとmode collapseの問題が発生してしまう。  
また、生成されるデータが知覚分布から大きく離れた領域飲みに分布する場合、通常のGANと同様にgradient vanishingによりモデルパラメータが更新されない問題も発生する。  

- 摂動の標準偏差<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /></a>への敏感性

<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /></a>
の値の大きさに応じて、生成データに加えられる摂動の大きさが変化する。  
適度に小さな値の
<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /></a>
を用いると、人間GANの識別器では知覚的な差異が発生せずに勾配が消失する。  
一方で、過度に大きな
<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /></a>
を用いると、正確な勾配推定が行われない。

- 人間への問合せ回数

人間への問合せ回数は、学習反復回数と生成データ数と摂動回数の積になる。  
この問合せ回数の増加は、人間による知覚評価にかかる様々なコストを爆増させる。  

## 4. 実験的評価

### 4.1. 実験条件

実験的評価では、提案する人間GANが音声の自然性の知覚分布を表現できるかを調査する。  
すなわち、評価者が人間GANが生成した音声を聞いて、人間の音声として許容できる音声特徴量の知覚分布を表現する。  
この評価では、通常のGANとの比較、データの次元数削減、生成モデルパラメータの初期化のために音声コーパスを利用する。  
使用するコーパスは女性199名の音声である。  

### 4.2. 実在データ分布と知覚分布の違いの確認

人間GANの必要性を実験的に確認するために、実在データ分布と知覚分布が異なることを示す。  
2次元の空間をグリッド分割し、各グリッドにおける自然性の許容度を評価する。  
評価は5段階で行われた(人間らしくない音声なら1、人間らしい音声なら5)。
1から5のそれぞれの値を事後確率
<a href="https://www.codecogs.com/eqnedit.php?latex=D(\hat{x}_n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D(\hat{x}_n)" title="D(\hat{x}_n)" /></a>
の0.00, 0.25, 0.50, 0.75, 1.00に対応させる。  
各グリッドの事後確率の値は、複数の評価値の平均である。  
次の図が結果。

<img width="600" alt="コーパスのカラーマップ" src="https://user-images.githubusercontent.com/39772824/94546207-87acf900-0288-11eb-8c41-d2376448fc26.png">

### 4.3. 生成モデル学習による生成データの変化

生成モデル学習の様子を定性的に確認するために、学習の各反復における生成データをプロットする。  
生成モデルに入力する乱数は、2次元の一様分布
<a href="https://www.codecogs.com/eqnedit.php?latex=U(-1,&space;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U(-1,&space;1)" title="U(-1, 1)" /></a>
に従うものとする。  
この乱数は、学習のはじめにランダムに生成され、学習中は固定される。  
初期の生成モデルパラメータはランダムに決定する。  
以下の値を用いて学習を行った。  

- 学習率<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha&space;=&space;0.0015" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;=&space;0.0015" title="\alpha = 0.0015" /></a>
- 音声特徴量数<a href="https://www.codecogs.com/eqnedit.php?latex=N=100" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N=100" title="N=100" /></a>
- 摂動回数<a href="https://www.codecogs.com/eqnedit.php?latex=R&space;=&space;5" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R&space;=&space;5" title="R = 5" /></a>
- 学習の反復回数 = 4

人間による評価には、クラウドソーシングサービス(Lancers)を使用する。  
評価時の質問は、2つの音声を聞いてもらい以下を問う。  
「1個目の方が人間らしいほど(1)を、2個目の方が人間らしいほど(5)を選択してください。両方同程度なら(3)を選択してください。」  
ここで得られた回答の1から5の値を、それぞれ事後確率の差分
<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;D(\hat{x}_n^{(r)}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;D(\hat{x}_n^{(r)}))" title="\Delta D(\hat{x}_n^{(r)}))" /></a>
の-1.0, -0.5, 0.0, 0.5, 1.0に対応させる。  

<img width="923" alt="HUMANGANの学習の推移" src="https://user-images.githubusercontent.com/39772824/94548758-26872480-028c-11eb-9e87-b2be82c404b6.png">

### 4.4. 生成モデル学習による事後確率の増加

学習の反復により事後確率が増加することを定量的に確認する。  
ここでは、学習に用いた乱数とは別の乱数を発生させ、新たな特徴量を生成する。  
各反復における事後確率の箱ひげ図を示す。  

<img width="600" alt="事後確率箱ひげ図" src="https://user-images.githubusercontent.com/39772824/94551605-91d2f580-0290-11eb-85f3-1730b55edf2c.png">

## 5. まとめ

人間GANでは人間の許容できるデータの範囲(知覚分布)を表現可能な生成モデル学習のために、人間による知覚評価を詐称する敵対的生成ネットワークを提案していた。  
今後は人間と計算機の共創に基づく敵対的生成ネットワークを検討するらしい。
