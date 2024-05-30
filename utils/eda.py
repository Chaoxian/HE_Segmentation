import os
import numpy as np
from tqdm import tqdm
import cv2

def PixelWiseCnt(masks_path="/dssg/home/acct-zhaochaoxian/zhaochaoxian-user1/SemanticSegmentation/ExternalData/sem_masks"):
    """
        Return cnt(dict) and sum(int)
        cnt: a dict where key is class index and value is pixel numbers
        sum: total pixel numbers
    """
    masks_list=os.listdir(masks_path)
    cnt={0:1e-8,1:1e-8} #modified by jidong at 2024/4/11
    for i in masks_list:
        mask = cv2.imread(os.path.join(masks_path,i),0)
        mask[mask==255] = 1 #modified by jidong at 2024/4/11
        classes=np.unique(mask)
        for idx in classes:
            cnt[idx]+=np.sum(mask==idx)

    sum=int(cnt[0]+cnt[1]) #modified by jidong at 2024/4/11
    return cnt,sum