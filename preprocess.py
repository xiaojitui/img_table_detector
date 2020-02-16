#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.misc import imresize

# pip install PyMuPDF
import fitz
from PIL import Image
import tempfile


# In[ ]:





# In[ ]:

    
def pdf_to_img(pdffile, outpath, zoom = 1):
  
    doc = fitz.open(pdffile) 

    for page_n in range(len(doc)):
        page = doc.loadPage(page_n) #number of page
        mat = fitz.Matrix(zoom, zoom)
        pix = page.getPixmap(matrix = mat, alpha = False)
        imgname = 'page_' + str(page_n) + '.jpg'
        pix.writePNG(outpath + imgname)
        #pix.writeImage(outpath + imgname)
    doc.close()
    

def pdf_to_tmp_img(pdffile, zoom = 1, page_num = 0):
  
    doc = fitz.open(pdffile) 

    page = doc.loadPage(page_num) #number of page

    mat = fitz.Matrix(zoom, zoom)
    pix = page.getPixmap(matrix = mat, alpha = False)

    with tempfile.NamedTemporaryFile() as fp:
        pix.writePNG(fp.name)
        fp.seek(0)
        img = Image.open(fp.name)
        img_array = np.array(img)
        img_size = img_array.shape
        img.close()
    ######imgname = path + '/scanned.png'
    ######pix.writePNG(imgname)
    doc.close()
    return img_array, img_size


# In[ ]:


def get_data(imgsamples, labels):
    data = {}

    for image in imgsamples:
        data[image] = []
        for label in labels:
            if image == label.split(',')[0].split('/')[-1]:
                data[image].append([int(label.split(',')[1]), int(label.split(',')[2]), 
                                    int(label.split(',')[3]), int(label.split(',')[4])])
    return data


# In[ ]:


def showbox(img, boxes, h_r=1, w_r=1):
    t = img.copy()
    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, x2 = x1//w_r, x2//w_r
        y1, y2 = y1//h_r, y2//h_r
        rect = Rectangle((x1,y1),x2-x1,y2-y1, fill=None, linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    if len(t.shape) == 3:
        ax.imshow(t[:, :, 0], cmap = 'gray')
    else:
        ax.imshow(t, cmap = 'gray')


# In[ ]:


def resize_img(img, output_h = 800, output_w = 600, output_c = 1):
    img_resized = imresize(img, (output_h, output_w)) 

    if output_c == 1 and len(img_resized.shape) == 3:
        img_resized = img_resized.mean(axis=2).astype(np.float32)
        img_resized = img_resized[:, :, np.newaxis]
        
    return img_resized

# trying to get ~105 rows (8 pixel high), ~20 columns (32 pixel wide)


# In[ ]:


def cut_img(img, h_sep = 8):
    img_bag = []
    
    for i in range(0, img.shape[0]//h_sep):
        if len(img.shape) == 3:
            img_piece = img[i*h_sep:(i+1)*h_sep, :, :]
        else:
            img_piece = img[i*h_sep:(i+1)*h_sep, :]
            img_piece = img_piece[:, :, np.newaxis]
        img_bag.append(img_piece)
        
    return img_bag


# In[ ]:





# In[ ]:





# In[ ]:




