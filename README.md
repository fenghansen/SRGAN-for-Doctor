# SRGAN-for-Doctor
***小记：我看到评委的reviews了，仨weak accept，一个weak reject，除了二号评委不太认真，其他几位说的还是有道理的。  
其实这个超分辨率模型最大的问题在于数据源——目前是简单缩放而来的配对数据（自然图像普遍也是），但是真实情况往往并不是这样的，有噪的低分辨率图像放大后噪声也会放大，这是十分危险的。然而问题在于，现在并没有的供你超分辨率的配对的医学影像，巧妇难为无米之炊。一号评委批评的很对，学习退化方法也确实是一个路子，他应该也看过ECCV2018的那个真实图像超分论文，我确实也正在琢磨这件事，感谢了！  
至于其他评委……也没批评错。其实那些问题我都考虑到过，没有表达出来并给出合适的解释是我的问题，第一次写论文确实漏洞百出，见笑了。***
## 0.引用
**SRGAN基础代码微调自 https://github.com/MathiasGruber/SRGAN-Keras  
结构性的实现有对照过https://github.com/SavaStevanovic/ESRGAN  
本项目的自然图像版本为https://github.com/fenghansen/ESRGAN-Keras （没改VGG19，自然图像用ImageNet预训练的VGG19挺好）**    

## 1.环境: Python 3.6 + Keras 2.2.4 + Tensorflow 1.12 + PyCharm 2018  
## 2.流程图  
如图所示。其中，LR为Low-Resolution低清图片，SR为Super-Resolution超分辨率图片，HR为High-Resolution高清原始图片，MSE为Mean Square Error均方误差。训练时LR为HR图像下采样并随机裁剪取得（同SRGAN），在实际应用时只需要取出单独取出生成器Generator即可通过输入任意医学图像来获得超分辨率图像，即应用时只需要从LR Image到SR Image的正向传播过程。
![流程2](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/%E6%B5%81%E7%A8%8B2.png)     
## 3.CT数据集上的横向对比试验
这是我们的SRGAN-D在CT数据上运行的结果对比图，传统超分辨率方法是Nearest和Bicubic，基于深度学习的超分辨率方法是SR-RRDB（本网络的第一阶段预训练）、SRGAN（DIV2K训练集）、ESRGAN（原论文提供）、ESRGAN（CT训练集微调）。  
【这里的ESRGAN（CT微调）原本用的是EnhanceNet，但是考虑到ESRGAN（原论文）的效果太过惊人无法体现其真正问题，我们使用了其CT迁移训练版，该部分由@lyc提供。实际上，参数丝毫未动地用CT图像训练后，**生成的图像在我们现在所使用的epoch之前就开始震荡了**，具体表现为：**噪点纹理一直在变化，但是轮廓无甚变化（边缘已经扭曲）**】
![CT00](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/CT00.png)   

我们郑重声明：这ESRGAN真的是官方权重跑出来的，您可以去官方GitHub提供的链接上下载他们的网络和权重（https://github.com/xinntao/ESRGAN）  
这说明了什么问题呢？说明通用数据集训练的权重不宜直接迁移运用到医学图像领域，一个领域有一个领域的特殊问题。   
**这其实也是我们要专门改进一下医学影像超分辨率的目的：  
这部分就是打算从超分辨率的角度来讲，说明只有针对医学图像改进的算法才是适合医学领域的超分辨率。  
下一部分是打算从医学影像超分辨率本身来讲，说明我们的方法比原来的方法好。（但是我经验太浅，实验放的不到位，论文里没写出来，可惜了）**  

![CT](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/CT%E7%9A%84PSNR%26SSIM%E5%AF%B9%E6%AF%94%E8%A1%A8.png)   
  
这个对比表说明了另外一个问题：PSNR/SSIM真的对基于GAN的超分辨率算法不友好。  
 
不得不说，GAN和PSNR/SSIM真的犯冲，显然这已经是两条不同的发展思路了，道不同不相为谋。PIRM-SR领域那个Perception Index倒是看起来靠谱，可是我不能拿自然图像去评判医学图像吧……要知道现在的这个模型输出自然图像都是黑白的……（我也很绝望，但是它真的学着学着就把彩图给忘了！！）  
PIRM-SR 2018就是ESRGAN得冠军的那个比赛： https://www.pirm2018.org/PIRM-SR.html 里面这张图我觉得太精辟了：  
![PIRM2018](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/PIRM2018.jpg)  
  
所以，我想说的是，虽然为了论文我贴了很多PSNR/SSIM的图，但是我觉得这个方法在“清晰”这个概念上，在GAN上，真的没什么意义。  
## 4.MRI数据集上的纵向对比试验
### 4.1 实际效果
那怎么来评判一个模型输出的好坏呢？当然是看啊！！说到底超分辨率追求的不就是视觉效果的提升么？！  
这是我们在MRI数据集上的效果（因为低剂量CT存在噪点，超分辨率的图像也有不协调点，实际应用价值略低）  
![MRI-762](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/samples/MRI-762-Epoch60000.png)  
![MRI-763](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/samples/MRI-763-Epoch60000.png)  
![MRI-764](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/samples/MRI-764-Epoch60000.png)  
![MRI-765](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/samples/MRI-765-Epoch60000.png)  

上面这四个是实验运行时用plt生成的，最好下载下来放大了看。另外名字忘记改了，应该标SRGAN-D的当时顺着上一个实验（ESRGAN-Keras）下来一直没改，那个SR-RRDB是用CT数据集与训练的所以色温不太对。  
### 4.2 不同VGG19对网络的影响  
![VGG](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/vgg-%E5%AF%B9%E6%AF%94%E5%9B%BE.png)  
其实，那个用MRI训练的VGG用作感知损失我个人感觉效果更好一些，或者说潜力更大一些，但是我们不懂医学，手动标注除了区分哪个是脑皮层哪个是脑内部基本已经是极限了，分的太细反而会分错的。而且这个的要点我个人感觉在轻度过拟合，因为你训练完后，最后的Dense层是不要的，也就是说你需要的只是提取语义特征的能力，从这点上来说略微过拟合后反而发挥了最大的“记忆力”，这对于同一类型的数据而言还是比较有意义的。（或许就像《Everybody Dance Now》那篇一样，严重过拟合然而效果超棒）  
【后来我想了想，应该用语义分割任务来预训练VGG的，这样肯定效果更好，不过这个MRI数据集不支持这个操作（没有语义标签），要是想更进一步的话可以考虑这个优化方案】  
我选择DTD主要是由于这样更稳妥一些，通用性更强，要不然超分辨率CT出来脑MRI条纹就不合适了，有违医学的底线。  

### 4.3 对SRGAN稳定性改进的直观效果
另外不得不说，SRGAN的训练并不是十分稳定，我们最重要的改进其实是工程改进，增加了稳定性。逐步放大尺寸训练并不是什么新鲜想法，灵感来自英伟达的那个GAN，Dropout是为了能稳定训练，要是想结果能复现，最后是要去掉的（要不输出像素值会有个缩放比例）。
从结果上来看，这个dropout很赚，稳定后还能撤掉，一点也不亏，没有它许多时候我都是训练不下来的（不得不说ESRGAN的那个权重我没看明白，用了以后直接训练崩溃，所以只能顺着SRGAN的改）。另外，不用WGAN和WGAN-GP的原因在于，5~10倍的训练时间我实在等不起，这个实验时限挺紧的。  
我本来只是想做个对比试验表达一下原生SRGAN不如我们的清晰，结果它中途直接崩了……比较复杂的图片确实很难训练就是了。
**由于SRGAN原论文的判别网络D的结构不支持我们的逐步训练方法，所以我们分别训练了三个模型：  
“原论文训练步骤的SRGAN”、  “采用了原论文的网络参数并应用我们的D网络和训练步骤的SRGAN’”  和  “SRGAN-D”。**  
 
![srgan对比](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/srgan.png)  
  
【注：下面的对比图忘记标记PSNR/SSIM了，用的都是对比图表中60000epoch的图片，srgan’的PSNR还是要高于srgan-d,SSIM也差不多，但是图片质量会差很多，这就是我们改进后的实际的好处，更清晰】   
![srgan对比](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/srgan-%E5%AF%B9%E6%AF%94%E5%9B%BE.png)   
从理论上讲，dropout和近年来提到的信息瓶颈也是有相似之处的，都是遗忘的哲学，去芜存菁，所以能有效也并不意外   

## 5.关于用我们代码测试的一些补充说明  
***最后，代码运行请注意版本问题，python3.6。要不有些util里有些方法不兼容。测试的话权重直接加载，用gan.test输出就行。***  
***如果想直接输出放大的图像，请使用util.py里的plot_bigger_images方法（改一下test函数的最后一行即可），不过这将没有PSNR和SSIM，因为没有原图做比较。  
具体案例可以参考本项目的自然图像版本https://github.com/fenghansen/ESRGAN-Keras 的后两个DIV2K案例。  
主要是没有比较大家看不出好坏，所以只能用重建图像来做比较了，不过重建图像由于有信息损失，部分图像（黄色框框）内容和原图不太一致，但其实和4x低清压缩图片的大体情况十分一致】***
