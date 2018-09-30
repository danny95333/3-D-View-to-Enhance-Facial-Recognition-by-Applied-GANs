# Use GANs model to Generate frontal human faces based on profile faces
All the input data are from MultiPIE dataset, 9 different identity faces each time with different poses and illumination.
![input](https://github.com/danny95333/3-D-View-to-Enhance-Facial-Recognition-by-Applied-GANs/blob/master/input_samples_iteration_200.png)

## 1st version
*siamese_GAN_1st.py* is the main training file
<br>1st version used siamese network as G-Net, and added a classic Discriminator Network as D-Net, and use random image as the ‘real data’ to train D-net. In this way, I didn't a good enough result to make the synthesis frontal image.
![image](https://github.com/danny95333/siamese-GAN-model-to-generate-frontal-face/blob/master/1st_output/fake_samples_iteration_45800.png)
## 2nd version
*siamese_GAN_19_2.py* is the main training file
<br>2nd version was based on the 1st version with a little modify. I use the frontal Ground Truth image as 'real data' to train D-net to make the supervison stronger. Standard light condition and fixed frontal pose make the synthesis pic more clear and real.
<br>![image](https://github.com/danny95333/siamese-GAN-model-to-generate-frontal-face/blob/master/2nd_output/result_19_2(2).png)
## 3rd version
*GAN_train_3rd.py/GAN_dataset_3rd.py/GAN_model_3rd.py* are the main training/dataload/model files
<br>3rd version was create base on the idea of add **random noise** in the G-net and give G-net some supervision information about **generate specific several pose face**(random pose code), "Face De-Occlusion and De-Profiling via a Generative Adversarial Network" paper. At the same time, I got some GANs training tricks, which shared by others on Chinese "ZHIHU". 
<br>**"If you have supervise information, like ground truth, give it to D-net and let D-net do more things more than judge whether the pic is real or fake"**
<br>So, I give up the idea that use Siamese, a dual channel network, as G-Net and give such supervise mission to D-net, G-net only have 1 loss to minimize. The 'pose/id/light' supervise information pass to D-net and D-net are responsible for minimizing these losses.
![image](https://github.com/danny95333/siamese-GAN-model-to-generate-frontal-face/blob/master/3rd_output/synthesis_result_iteration_35500.png)
<br>
<br>**NOTICE**: netD_70000.pth and netG_70000.pth are the weights files of 1st model. The **2nd&3rd model's weights files** are too big to upload on the github. If your condition is not allowed to train this model and want to run the pretained model, you can contact me and leave your email address.
## experiment record is in record.txt file
