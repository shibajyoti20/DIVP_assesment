from scipy import fftpack
from abc import ABC, abstractmethod
from typing import Dict

import cv2
import matplotlib.pylab as pylab
import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy import pi, r_, sin, zeros

import utils


class ImageProcessor(ABC):

    @classmethod
    @abstractmethod
    def process(cls):
        pass



class ImageCompressor(ImageProcessor):
    """
        param : {

            jpg_quality: <quality>

            png_quality: <quality>

        }
    """

    @classmethod
    def compress_normal(cls, filepath, quality):

        img = cv2.imread(filepath)

        path = utils.format_filename(filepath, "_compressed")

        ext = utils.get_file_extension(filepath)

        if ext == "not supported":
            print('File format not supported')
            return

        if ext == 'jpg' or ext == 'jpeg': 

            cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        elif ext == 'png':

            cv2.imwrite(path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), quality])


    @classmethod
    def dct2(cls, a):
        return fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

    @classmethod
    def idct2(cls, a):
        return fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')


    @classmethod
    def dct_compress(cls, filepath):

        img = cv2.imread(filepath).astype(float)

        imsize = img.shape
        dct = np.zeros(imsize)

        # Do 8x8 DCT on image (in-place)
        for i in r_[:imsize[0]:8]:
            for j in r_[:imsize[1]:8]:
                dct[i:(i+8),j:(j+8)] = cls.dct2( img[i:(i+8),j:(j+8)] )


        # Threshold
        thresh = 0.012
        dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))

        percent_nonzeros = np.sum( dct_thresh != 0.0 ) / (imsize[0]*imsize[1]*1.0)

        im_dct = np.zeros(imsize)

        for i in r_[:imsize[0]:8]:
            for j in r_[:imsize[1]:8]:
                im_dct[i:(i+8),j:(j+8)] = cls.idct2( dct_thresh[i:(i+8),j:(j+8)] )


        hstack = np.hstack((img, abs(im_dct))) 

        hstack_filename = utils.format_filename(filepath, "_dct_hstack")

        output_filename = utils.format_filename(filepath, "_dct")

        cv2.imwrite(output_filename, im_dct)

        cv2.imwrite(hstack_filename, hstack)


    @classmethod
    def dft_compress(cls, filepath):

        im = cv2.imread(filepath).astype(float)

        imsize = im.shape

        dft = zeros(imsize,dtype='complex');
        im_dft = zeros(imsize,dtype='complex');

        # 8x8 DFT
        for i in r_[:imsize[0]:8]:
            for j in r_[:imsize[1]:8]:
                dft[i:(i+8),j:(j+8)] = np.fft.fft2( im[i:(i+8),j:(j+8)] )

        # Thresh
        thresh = 0.013
        dft_thresh = dft * (abs(dft) > (thresh*np.max(abs(dft))))


        percent_nonzeros_dft = np.sum( dft_thresh != 0.0 ) / (imsize[0]*imsize[1]*1.0)

        # 8x8 iDFT
        for i in r_[:imsize[0]:8]:
            for j in r_[:imsize[1]:8]:
                im_dft[i:(i+8),j:(j+8)] = np.fft.ifft2( dft_thresh[i:(i+8),j:(j+8)] )

        hstack = np.hstack((im, abs(im_dft))) 

        hstack_filename = utils.format_filename(filepath, "_dft_hstack")

        output_filename = utils.format_filename(filepath, "_dft")

        cv2.imwrite(output_filename, abs(im_dft))

        cv2.imwrite(hstack_filename, hstack)



class ImageEnhancer(ImageProcessor):

    @classmethod
    def histogram_eq(cls, filepath):
        
        img = cv2.imread(filepath,0)

        equ = cv2.equalizeHist(img)

        res = np.hstack((img, equ)) 

        output_filename = utils.format_filename(filepath, "_enhanced_histeq")

        cv2.imwrite(output_filename, res)

    
    @classmethod
    def clahe(cls, filepath):

        img = cv2.imread(filepath,0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        cl1 = clahe.apply(img)

        res = np.hstack((img, cl1)) 


        output_filename = utils.format_filename(filepath, "_enhanced_clahe")

        cv2.imwrite(output_filename, res)



class ImageSegmentor(ImageProcessor):

    @classmethod
    def k_means(cls, filepath, k):

        img = cv2.imread(filepath,0)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        twoDimage = img.reshape((-1,3))

        twoDimage = np.float32(twoDimage)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        K = k

        attempts=10

        ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((img.shape))

        hstack = np.hstack((img, result_image)) 

        hstack_filename = utils.format_filename(filepath, "_k_means_hstack")

        output_filename = utils.format_filename(filepath, "_k_means")

        cv2.imwrite(output_filename, result_image)

        cv2.imwrite(hstack_filename, hstack)


    @classmethod
    def contour_detection(cls, filepath):
        img = cv2.imread(filepath,0)

        img = cv2.resize(img,(256,256))

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
        mask = np.zeros((256,256), np.uint8)
        masked = cv2.drawContours(mask, [cnt],-1, 255, -1)

        dst = cv2.bitwise_and(img, img, mask=mask)
        result_image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        output_filename = utils.format_filename(filepath, "_contour")

        cv2.imwrite(output_filename, result_image)