# AIF-LFNet
## AIF-LFNet: All-in-Focus Light Field Super-Resolution Method Considering the Depth-Varying Defocus
### The dataset
We have constructed the dataset obtaining 150 LF images with training data and 15 LF images with vadilation and testing data by applying the Blender add-on Setup.
The dataset is shared in [Baidu Drive](https://pan.baidu.com/s/1xzNhYjPm8G31kyyqNwHvdQ?pwd=DHUU), the code is $\textbf{DHUU}$, and OneDrive(https://1drv.ms/u/s!ArC6QM5-HSjkhl1tsJn3Wx-EoqSf?e=tWasI1).
The typical images are shown below.
![fig1](https://github.com/qingpu1988/AllfocusNet/blob/main/Fig3.png)
### The algorithm
The network is shown below, $\textbf{the code has been released，which is heavily borrowed from the LF-InterNet, many thanks!,For the feature extraction layer, maybe the convolutional layer in distgdisp will be more efficient, by reducing 5-branches to 3-brabches. However,if you choose that, maybe the performance in occlude regions will be weakened}$.
![fig2](https://github.com/qingpu1988/AllfocusNet/blob/main/Fig4.png)
### The results
The MSE and SSIM results for 15 testing images are listed below.
![fig3](https://github.com/qingpu1988/AllfocusNet/blob/main/fig-result.png)
