
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

#the below code displays the Blur images, DoG images , maxima/minima images, key points plot 
#please navigate to the main() function in order provide the file path

def kernel_flip(kernel):
  
    kernel_copy = kernel.copy()
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
              kernel_copy[i,j] = kernel[kernel.shape[0]-i-1,kernel.shape[1]-j-1]
  
    return kernel_copy
  
    return kernel_copy
def conv(image,kernel):
    #print("test1")
    #flipped_kernel = kernel_flip(kernel)
    flipped_kernel = kernel.copy()
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    kernel_height = flipped_kernel.shape[0]
    kernel_width = flipped_kernel.shape[1]
    
    h= kernel_height//2
    w= kernel_width//2
    conv_image = np.zeros(image.shape)
    
    for i in range(3, image_height-3):
        for j in range(3,image_width-3):
            sum1 = 0
            for m in range(-3,+4):
                for n in range(-3,+4):
                    
                    sum1 += flipped_kernel[m,n]*image[i+m,j+n]
                    
            conv_image[i,j] = sum1
    return conv_image

def gaussian_kernel_gen(sigma):
    #x , y  = np.meshgrid(np.linspace(-1,1,5), np.linspace(-1,1,5))
    gaussian_kernel = np.zeros((7,7))
    gaussian_height = gaussian_kernel.shape[0]
    gaussian_width = gaussian_kernel.shape[1]
    
    dummy = np.zeros((7,7))
    height_dummy = dummy.shape[0]
    width_dummy = dummy.shape[1]
    
    
    list1 = np.array([3,2,1,0,1,2,3])
    list2 = np.array([3,2,1,0,1,2,3])
    for i in range(height_dummy):
        for j in range(width_dummy):
            dummy[i,j] = list1[i]*list1[i] + list2[j]*list2[j]
    
    
    sigma_square = sigma*sigma
    denominator = 2*(math.pi)*sigma_square
    
    for k in range(gaussian_height):
        for l in range(gaussian_width):
            gaussian_kernel[k,l] = (1/denominator)* np.exp(-(dummy[k,l]/(2*sigma_square)))
    
    #normalizing the gaussian kernel
    normalizing_factor = sum(sum(gaussian_kernel))
   
    gaussian_kernel = gaussian_kernel * (1/normalizing_factor)
    
    return gaussian_kernel
                             
def blurr_image_gen(sigma,image):
    list_of_images = []
    blurr_image = image.copy()
    for i in np.nditer(sigma):
        gaussian_kernel = gaussian_kernel_gen(i)
        #print(sum(sum(gaussian_kernel)))
        blurr_image = conv(blurr_image,gaussian_kernel)
        list_of_images.append(blurr_image)
    return list_of_images

def DoG(octave1_image):
    octave1_image_copy = octave1_image.copy()
    Dog_images = []
    length = len(octave1_image_copy)
    for i in range(length-1):
        Dog_images.append(octave1_image_copy[i]-octave1_image_copy[i+1])
    return Dog_images
              
def calculating_maxima(post_DoG_image,current_DoG_image,next_DoG_image):
    Maxima_image = np.zeros(current_DoG_image.shape)
    
    DoG_height = current_DoG_image.shape[0]
    DoG_width = current_DoG_image.shape[1]

    for i in range(1, DoG_height-1):
        for j in range(1, DoG_width-1):
            list1 = []
            list2 = []
            list3 = []
            list4 = []
            for x in range(i-1, i+2):
                for y in range(j-1, j+2):
                    list1.append(post_DoG_image[x,y])
                    list2.append(current_DoG_image[x,y])
                    list3.append(next_DoG_image[x,y])
            list4 = list1 + list2 + list3
            
            list4.sort()
            length = len(list4)
            if current_DoG_image[i,j] == list4[0] or current_DoG_image[i,j] == list4[length-1]:
                Maxima_image[i,j] = 255
    return Maxima_image

def locate_key_points(image,maxima_image,factor):
    
    image_copy = image.copy()
    maxima_image_height = maxima_image.shape[0]
    maxima_image_width = maxima_image.shape[1]
    
    for i in range(maxima_image_height):
        for j in range(maxima_image_width):
            if maxima_image[i,j] == 255:
                #cv2.circle(image,(i,j),2,(0,0,0),-1)
                image_copy[i*factor,j*factor] = 255
                
    return image_copy
                
def sampling(source_image,factor_value):
   
    source_image_height = source_image.shape[0]
    source_image_width = source_image.shape[1]
    
    if source_image_height%factor_value == 0 and source_image_width%factor_value == 0:
        x = source_image_height//factor_value
        y = source_image_width//factor_value
    elif source_image_height%factor_value == 0 and source_image_width%factor_value != 0:
        x = source_image_height//factor_value
        y = (source_image_width//factor_value)+1
    elif source_image_height%factor_value != 0 and source_image_width%factor_value == 0:
        x = (source_image_height//factor_value)+1
        y = (source_image_width//factor_value)
    elif source_image_height%factor_value != 0 and source_image_width%factor_value != 0:
        x = (source_image_height//factor_value)+1
        y = (source_image_width//factor_value)+1
    
    sampled_image = np.empty((x, y))
    
    for i in range(x):
        for j in range(y):
            sampled_image[i,j] = source_image[i*factor_value,j*factor_value]
    return sampled_image
   
    
    
def main():
    #reading the input image-provide the file path in the below lies
    img = cv2.imread("/Users/vidyach/Desktop/cvip/proj1/test_images/task2.jpg",0)
    output_image = cv2.imread("/Users/vidyach/Desktop/cvip/proj1/test_images/task2.jpg",1)
    
    cv2.imshow('Original image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #passing the sigma values
    oct1_sigma = np.array([1/(math.sqrt(2)),1,math.sqrt(2),2,2*(math.sqrt(2))])
    oct2_sigma = np.array([math.sqrt(2),2,2*(math.sqrt(2)),4,4*(math.sqrt(2))])
    oct3_sigma = np.array([2*(np.sqrt(2)),4,4*(np.sqrt(2)),8,8*(np.sqrt(2))])
    oct4_sigma = np.array([4*(np.sqrt(2)),8,8*(np.sqrt(2)),16,16*(np.sqrt(2))])
    
    #declaring an empty arrays
    octave1_image = []
    octave2_image = []
    octave3_image = []
    octave4_image = []
    
    #sampling the image for octave 2
    sampled_image1 = sampling(img,2)
    sample2 = np.array(sampled_image1,dtype=float)/float(255)
    print(sampled_image1.shape)
    cv2.imshow('Sampled image-2',sample2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #sampling the image for octave 3
    sampled_image2 = sampling(sampled_image1,2)
    print(sampled_image2.shape)
    sample3 = np.array(sampled_image2,dtype=float)/float(255)
    cv2.imshow('Sampled image-3',sample3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    #sampling the image for octave 4
    sampled_image3 = sampling(sampled_image2,2)
    sample4 = np.array(sampled_image3,dtype=float)/float(255)
    cv2.imshow('Sampled image-3',sample4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
    #generating the Blur images for the Octaves
    octave1_image = blurr_image_gen(oct1_sigma,img)
    octave2_image = blurr_image_gen(oct2_sigma,sampled_image1)
    octave3_image = blurr_image_gen(oct3_sigma,sampled_image2)
    octave4_image = blurr_image_gen(oct4_sigma,sampled_image3)
    
    #generatin gthe DoGs form the Blur images
    oct1_Dog = DoG(octave1_image)
    oct2_Dog = DoG(octave2_image)
    oct3_Dog = DoG(octave3_image)
    oct4_Dog = DoG(octave4_image)
    
    #plotting the generated Ocatves using imshow()
    octave1_image1 = np.array(octave1_image[0],dtype=float)/float(255)
    cv2.imshow('Octave-1 Blurr 1',octave1_image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave1_image2 = np.array(octave1_image[1],dtype=float)/float(255)
    cv2.imshow('Octave-1 Blurr 2',octave1_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave1_image3 = np.array(octave1_image[2],dtype=float)/float(255)
    cv2.imshow('Octave-1 Blurr 3',octave1_image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave1_image4 = np.array(octave1_image[3],dtype=float)/float(255)
    cv2.imshow('Octave-1 Blurr 4',octave1_image4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave1_image5 = np.array(octave1_image[4],dtype=float)/float(255)
    cv2.imshow('Octave-1 Blurr 5',octave1_image5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #plotting the generated Ocatves using imshow()
    octave2_image1 = np.array(octave2_image[0],dtype=float)/float(255)
    cv2.imshow('Octave-2 Blurr 1',octave2_image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave2_image2 = np.array(octave2_image[1],dtype=float)/float(255)
    cv2.imshow('Octave-2 Blurr 2',octave2_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave2_image3 = np.array(octave2_image[2],dtype=float)/float(255)
    cv2.imshow('Octave-2 Blurr 3',octave2_image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave2_image4 = np.array(octave2_image[3],dtype=float)/float(255)
    cv2.imshow('Octave-2 Blurr 4',octave2_image4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave2_image5 = np.array(octave2_image[4],dtype=float)/float(255)
    cv2.imshow('Octave-2 Blurr 5',octave2_image5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave3_image1 = np.array(octave3_image[0],dtype=float)/float(255)
    cv2.imshow('Octave-3 Blurr 1',octave3_image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave3_image2 = np.array(octave3_image[1],dtype=float)/float(255)
    cv2.imshow('Octave-3 Blurr 2',octave3_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave3_image3 = np.array(octave3_image[2],dtype=float)/float(255)
    cv2.imshow('Octave-3 Blurr 3',octave3_image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave3_image4 = np.array(octave3_image[3],dtype=float)/float(255)
    cv2.imshow('Octave-3 Blurr 4',octave3_image4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave3_image5 = np.array(octave3_image[4],dtype=float)/float(255)
    cv2.imshow('Octave-3 Blurr 5',octave3_image5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave4_image1 = np.array(octave4_image[0],dtype=float)/float(255)
    cv2.imshow('Octave-4 Blurr 1',octave3_image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave4_image2 = np.array(octave4_image[1],dtype=float)/float(255)
    cv2.imshow('Octave-4 Blurr 2',octave3_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave4_image3 = np.array(octave4_image[2],dtype=float)/float(255)
    cv2.imshow('Octave-4 Blurr 3',octave3_image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave4_image4 = np.array(octave4_image[3],dtype=float)/float(255)
    cv2.imshow('Octave-4 Blurr 4',octave3_image4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    octave4_image5 = np.array(octave4_image[4],dtype=float)/float(255)
    cv2.imshow('Octave-4 Blurr 5',octave3_image5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
     #plotting the DoGs 
    cv2.imshow('Octave 1 DoG 1',oct1_Dog[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 1 DoG 2',oct1_Dog[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 1 DoG 3',oct1_Dog[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 1 DoG 4',oct1_Dog[3])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #plotting the DoGs 
    cv2.imshow('Octave 2 DoG 1',oct2_Dog[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 2 DoG 2',oct2_Dog[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 2 DoG 3',oct2_Dog[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 2 DoG 4',oct2_Dog[3])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 3 DoG 1',oct3_Dog[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 3 DoG 2',oct3_Dog[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 3 DoG 3',oct3_Dog[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 3 DoG 4',oct3_Dog[3])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 4 DoG 1',oct4_Dog[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 4 DoG 2',oct4_Dog[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 4 DoG 3',oct4_Dog[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Octave 4 DoG 4',oct4_Dog[3])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    #claculating maxima/minima octave 1
    maxima_image = calculating_maxima(oct1_Dog[0],oct1_Dog[1],oct1_Dog[2])
    maxima_image0 = calculating_maxima(oct1_Dog[1],oct1_Dog[2],oct1_Dog[3])
 
    #claculating maxima/minima octave 2
    maxima_image1 = calculating_maxima(oct2_Dog[0],oct2_Dog[1],oct2_Dog[2])
    maxima_image2 = calculating_maxima(oct2_Dog[1],oct2_Dog[2],oct2_Dog[3])
 
    
    #claculating maxima/minima octave 3
    maxima_image3 = calculating_maxima(oct3_Dog[0],oct3_Dog[1],oct3_Dog[2])
    maxima_image4 = calculating_maxima(oct3_Dog[1],oct3_Dog[2],oct3_Dog[3])

    #claculating maxima/minima octave 4
    maxima_image5 = calculating_maxima(oct4_Dog[0],oct4_Dog[1],oct4_Dog[2])
    maxima_image6 = calculating_maxima(oct4_Dog[1],oct4_Dog[2],oct4_Dog[3])
    
    cv2.imshow('maxima image1',maxima_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('maxima image2',maxima_image0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('maxima image1',maxima_image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('maxima image2',maxima_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('maxima image3',maxima_image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('maxima image4',maxima_image4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('maxima image3',maxima_image5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('maxima image4',maxima_image6)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #plotting key points for octave 2
    plotted_image = locate_key_points(img,maxima_image,1)
    plotted_image0 = locate_key_points(plotted_image,maxima_image0,1)
    
    #plotting key points for octave 2
    plotted_image1 = locate_key_points(plotted_image0,maxima_image1,2)
    plotted_image2 = locate_key_points(plotted_image1,maxima_image2,2)
    
        
    #plotting key points for octave 3
    plotted_image3 = locate_key_points(plotted_image2,maxima_image3,4)
    plotted_image4 = locate_key_points(plotted_image3,maxima_image4,4)
    
    #plotting key points for octave 2
    plotted_image5 = locate_key_points(plotted_image4,maxima_image5,8)
    plotted_image6 = locate_key_points(plotted_image5,maxima_image6,8) 
        
    cv2.imshow('key points plotted image octave 3',plotted_image6)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
main()

