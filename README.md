# SRGAN-for-Doctor
## 0.引用
**SRGAN基础代码微调自 https://github.com/MathiasGruber/SRGAN-Keras  
结构性的实现有对照过https://github.com/SavaStevanovic/ESRGAN  
本项目的自然图像版本为https://github.com/fenghansen/ESRGAN-Keras**    

## 1.环境: Python 3.6 + Keras 2.2.4 + Tensorflow 1.12 + PyCharm 2018  
## 2.CT数据集上的初步试验————我们远优于未训练的各个网络
这是我们的SRGAN-D在CT数据上运行的结果对比图，传统超分辨率方法是Nearest和Bicubic，基于深度学习的超分辨率方法是SR-RRDB（本网络的第一阶段预训练）、SRGAN（DIV2K训练集）、ESRGAN（原论文提供）、ESRGAN（CT训练集微调）。  
【这里的ESRGAN（CT微调）原本用的是EnhanceNet，但是考虑到ESRGAN（原论文）的效果太过惊人无法体现其真正问题，我们使用了其CT迁移训练版，该部分由@lyc提供。实际上，参数丝毫未动地用CT图像训练后，**生成的图像在我们现在所使用的epoch之前就开始震荡了**，具体表现为：**噪点纹理一直在变化，但是轮廓无甚变化（边缘已经扭曲）**】
![CT0](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/CT0.png)   

我们郑重声明：这ESRGAN真的是官方权重跑出来的，您可以去官方GitHub提供的链接上下载他们的网络和权重（https://github.com/xinntao/ESRGAN）  
这说明了什么问题呢？说明通用数据集训练的权重不宜直接迁移运用到医学图像领域，一个领域有一个领域的特殊问题。   
  

![CT](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/CT%E7%9A%84PSNR%26SSIM%E5%AF%B9%E6%AF%94%E8%A1%A8.png)   
  
这个对比表说明了另外一个问题：PSNR/SSIM真的对基于GAN的超分辨率算法不友好。  
 
不得不说，GAN和PSNR/SSIM真的犯冲，显然这已经是两条不同的发展思路了，道不同不相为谋。PIRM-SR领域那个Perception Index倒是看起来靠谱，可是我不能拿自然图像去评判医学图像吧……要知道现在的这个模型输出自然图像都是黑白的……（我也很绝望，但是它真的学着学着就把彩图给忘了！！）  
PIRM-SR 2018就是ESRGAN得冠军的那个比赛： https://www.pirm2018.org/PIRM-SR.html 里面这张图我觉得太精辟了：  
![PIRM2018](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/PIRM2018.jpg)  
  
所以，我想说的是，虽然为了论文我贴了很多PSNR/SSIM的图，但是我觉得这个方法在“清晰”这个概念上，在GAN上，真的没什么意义。
那怎么来评判一个模型输出的好坏呢？当然是看啊！！说到底超分辨率追求的不就是视觉效果的提升么？！  
这是我们在MRI数据集上的效果（因为低剂量CT存在噪点，超分辨率的图像也有不协调点，实际应用价值略低）  
![MRI-762](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/samples/MRI-762-Epoch60000.png)  
![MRI-763](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/samples/MRI-763-Epoch60000.png)  
![MRI-764](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/samples/MRI-764-Epoch60000.png)  
![MRI-765](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/samples/MRI-765-Epoch60000.png)  

上面这四个是实验运行时用plt生成的，最好下载下来放大了看。另外名字忘记改了，应该标SRGAN-D的当时顺着上一个实验（ESRGAN-Keras）下来一直没改，那个SR-RRDB是用CT数据集与训练的所以色温不太对。  
另外，那个用MRI训练的VGG用作感知损失我个人感觉效果更好一些，或者说潜力更大一些，但是我们不懂医学，手动标注除了区分哪个是脑皮层哪个是脑内部基本已经是极限了，分的太细反而会分错的。而且这个的要点我个人感觉在轻度过拟合，因为你训练完后，最后的Dense层是不要的，也就是说你需要的只是提取语义特征的能力，从这点上来说略微过拟合后反而发挥了最大的“记忆力”，这对于同一类型的数据而言还是比较有意义的。（或许就像《Everybody Dance Now》那篇一样，严重过拟合然而效果超棒）  
![VGG](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/vgg-%E5%AF%B9%E6%AF%94%E5%9B%BE.png)  
我选择DTD主要是由于这样更稳妥一些，通用性更强，要不然超分辨率CT出来脑MRI条纹就不合适了  

另外不得不说，SRGAN的训练并不是十分稳定，我们最重要的改进其实是工程改进，增加了稳定性。逐步放大尺寸训练并不是什么新鲜想法，灵感来自英伟达的那个GAN，但是这个Dropout就纯属在下的灵光一闪了~ **我还真没见过谁往SRGAN里放Dropout（可能是他们觉得这样太蠢……）**  
从结果上来看，这个dropout很赚，稳定后还能撤掉，一点也不亏，没有它许多时候我都是训练不下来的（不得不说ESRGAN的那个权重我没看明白，用了以后直接训练崩溃，所以只能顺着SRGAN的改）  
君不见，我本来只是想做个对比试验表达一下原生SRGAN不如我们的清晰，结果它中途直接崩了，太面子了！  
【注1：下面的对比图忘记标记PSNR/SSIM了，用的都是对比图表中59000epoch的图片，srgan的PSNR还是要高于srgan-d,SSIM也差不多，但是图片质量会差很多，这就是我们改进后的实际的好处，更清晰】  
![srgan对比](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/srgan.jpg)  
![srgan对比](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/srgan-%E5%AF%B9%E6%AF%94%E5%9B%BE.png)   
从理论上讲，dropout和近年来提到的信息瓶颈也是有相似之处的，都是遗忘的哲学，去芜存菁，所以能有效也并不意外   


***最后，代码运行请注意版本问题，python3.6。要不有些util里有些方法不兼容。测试的话权重直接加载，用gan.test输出就行***
