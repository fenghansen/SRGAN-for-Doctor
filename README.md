# SRGAN-for-Doctor
不管结果多么令人震惊，请看完再说，在下先上图为敬  
![CT0](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/CT0.png)   
看这个ESRGAN！惊不惊喜？意不意外？这真的是官方权重跑出来的……
![CT](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/CT%E7%9A%84PSNR%26SSIM%E5%AF%B9%E6%AF%94%E8%A1%A8.png)   
传统超分辨率方法是Nearest和Bicubic，基于深度学习的超分辨率方法是SR-RRDB（本网络的第一阶段预训练）、SRGAN（DIV2K训练集）、ESRGAN（原论文提供）、ESRGAN（CT训练集微调）。  
这说明了什么问题呢？说明通用数据集训练的权重不宜直接迁移运用到医学图像领域，一个领域有一个领域的特殊问题。  
另外，不得不说，GAN和PSNR/SSIM真的犯冲，显然这已经是两条不同的发展思路了，道不同不相为谋。PIRM-SR领域那个Perception Index倒是看起来靠谱，可是我不能拿自然图像去评判医学图像吧……要知道现在的这个模型输出自然图像都是黑白的……（我也很绝望，但是它真的学着学着就把彩图给忘了！！）  
PIRM-SR 2018就是ESRGAN得冠军的那个比赛： https://www.pirm2018.org/PIRM-SR.html 里面这张图我觉得太精辟了：  
![PIRM2018](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/pics/PIRM2018.jpg)  
所以，我想说的是，虽然为了论文我贴了很多PSNR/SSIM的图，但是我觉得这个方法在“清晰”这个概念上，在GAN上，真的没什么意义。
那怎么来评判一个模型输出的好坏呢？当然是看啊！！说到底超分辨率追求的不就是视觉效果的提升么？！
所以直接上图吧。
![MRI-762](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/samples/MRI-762-Epoch60000.png)  
![MRI-763](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/samples/MRI-763-Epoch60000.png)  
![MRI-764](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/samples/MRI-764-Epoch60000.png)  
![MRI-765](https://github.com/fenghansen/SRGAN-for-Doctor/blob/master/samples/MRI-765-Epoch60000.png)  
