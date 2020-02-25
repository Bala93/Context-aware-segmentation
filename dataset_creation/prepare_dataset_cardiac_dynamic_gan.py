import h5py
import numpy as np
import glob
import os
from tqdm import tqdm 
from matplotlib import pyplot as plt
from skimage import measure
from img_mask_transform import RandomVerticalFlip,RandomHorizontalFlip,RandomRotation
from PIL import Image
from torchvision import transforms


def get_file_dim_list(file_path):

    file_path = file_path + '*.h5'
    files = glob.glob(file_path)
    
    file_list = []
    dim_list = []
    
    for file in tqdm(files):
        hf = h5py.File(file)
        fname = os.path.basename(file)
        img = np.array(hf['img'].value)
        mask = np.uint8(hf['mask'].value)
        img = np.pad(img,pad_width=((5,5),(5,5)),mode='constant')
        mask = np.pad(mask,pad_width=((5,5),(5,5)),mode='constant')
        #mask = mask * 85
        mask1 = mask > 0
    
        region_props = measure.regionprops(mask1.astype('int'))
        #print (len(region_props))
    
        bbox = region_props[0].bbox
    
        x1,y1 = bbox[0],bbox[1]
        x2,y2 = bbox[2],bbox[3]
    
        w = x2 - x1
        h = y2 - y1
    
        if w - h > 0:
            h = w
        if h - w > 0:
            w = h 
        
        dim_list.append(w)
        file_list.append(fname)

    return dim_list,file_list


h5_dir = '/media/htic/NewVolume3/Balamurali/cardiac_mri_acdc_dataset/train/'
dim_list,file_list = get_file_dim_list(h5_dir)
print (len(dim_list))

dim_np = np.array(dim_list)
files_np = np.array(file_list)
batch_size =16
tform = transforms.Compose([RandomHorizontalFlip(),RandomVerticalFlip(),RandomRotation(5)])

h5_save_dir = '/media/htic/NewVolume3/Balamurali/cardiac_mri_acdc_dataset/train_dyn_16/'
h5_count = 1
dim = 160
step = 10

for kk in tqdm(range(10,160,step)):
    files = files_np[np.logical_and(dim_np >= kk,dim_np < kk + step)]
    len_files = len(files)
    if not len_files:
        continue
    
    # Files convert to numpy stack
    img_np  = np.empty([160,160,0])
    mask_np = np.empty([160,160,0])
    for h5_name in files:
        h5_path = h5_dir + h5_name
        with h5py.File(h5_path,'r') as hf:
            img = hf['img'].value
            mask = hf['mask'].value
            img = np.pad(img,pad_width=((5,5),(5,5)),mode='constant')
            mask = np.pad(mask,pad_width=((5,5),(5,5)),mode='constant')
            #print (img.shape,mask.shape)
            img_np = np.dstack([img_np,img])
            mask_np = np.dstack([mask_np,mask])

    # Find the remaining files
    rem = len_files % batch_size
    len_rem = batch_size - rem 
    
    img_np_aug = np.empty([160,160,0])
    mask_np_aug = np.empty([160,160,0])
    
    # Iterate and get the new stack
    for jj in  range(len_rem):
        #print (len_files)
        rand_int = np.random.randint(0,len_files,1)[0]
        img = img_np[:,:,rand_int]
        mask = mask_np[:,:,rand_int]

        img_mask = np.concatenate([img,mask],axis=1)
        tform_img_mask = tform(img_mask)
        
        tform_img  = tform_img_mask[:,:160]
        tform_mask = tform_img_mask[:,160:]
        
        img_np_aug = np.dstack([img_np_aug,tform_img])
        mask_np_aug = np.dstack([mask_np_aug,tform_mask])

    #print (img_np.shape,mask_np.shape,img_np_aug.shape,mask_np_aug.shape)
    img_np_upd  = np.dstack([img_np,img_np_aug]) 
    mask_np_upd = np.dstack([mask_np,mask_np_aug])

    # If possible randomize the image and mask together
    
    batch_split = int(img_np_upd.shape[-1] / batch_size)

    print ("{}:{}---{}-->{}".format(kk,kk+step,len_files,batch_split))

    for jj in range(batch_split):

        h5_save_path = os.path.join(h5_save_dir,'{}.h5'.format(h5_count))
        img  = img_np_upd[:,:,batch_size * jj: (jj + 1) * batch_size]
        mask = mask_np_upd[:,:,batch_size * jj: (jj + 1) * batch_size].astype('int')
        centroid = []
        for ii in range(batch_size):
            mask_ = mask[:,:,ii]>0
            mask_ = mask_.astype('int')
            rp = measure.regionprops(mask_)[0]

            cy,cx = rp.centroid
            cy,cx = int(cy),int(cx)

            centroid.append([cx,cy])

        centroid = np.array(centroid)

        with h5py.File(h5_save_path,'w') as f:
            f.create_dataset('img',data=img)
            f.create_dataset('mask',data=mask)
            f.create_dataset('coord',data=centroid)
            f.create_dataset('bbox',data=kk)
        #print (img.shape,mask.shape,centroid.shape,kk)

        #print (img.shape)
        #print (mask.shape)

        h5_count += 1     
 
    #break
