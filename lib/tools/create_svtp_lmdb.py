import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tqdm import tqdm
import six
from PIL import Image
import scipy.io as sio
from tqdm import tqdm
import re

def checkImageIsValid(imageBin):
  if imageBin is None:
    return False
  imageBuf = np.fromstring(imageBin, dtype=np.uint8)
  img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
  imgH, imgW = img.shape[0], img.shape[1]
  if imgH * imgW == 0:
    return False
  return True


def writeCache(env, cache):
  with env.begin(write=True) as txn:
    for k, v in cache.items():
      txn.put(k.encode(), v)


def _is_difficult(word):
  assert isinstance(word, str)
  return not re.match('^[\w]+$', word)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
  """
  Create LMDB dataset for CRNN training.
  ARGS:
      outputPath    : LMDB output path
      imagePathList : list of image path
      labelList     : list of corresponding groundtruth texts
      lexiconList   : (optional) list of lexicon lists
      checkValid    : if true, check the validity of every image
  """
  assert(len(imagePathList) == len(labelList))
  nSamples = len(imagePathList)
  env = lmdb.open(outputPath, map_size=2147483648)#1099511627776)所需要的磁盘空间的最小值，之前是1T，我改成了2g，否则会报磁盘空间不足，这个数字是字节
  cache = {}
  cnt = 1
  for i in range(nSamples):
    imagePath = imagePathList[i]
    label = labelList[i]
    if len(label) == 0:
      continue
    if not os.path.exists(imagePath):
      print('%s does not exist' % imagePath)
      continue
    with open(imagePath, 'rb') as f:
      imageBin = f.read()
    if checkValid:
      if not checkImageIsValid(imageBin):
        print('%s is not a valid image' % imagePath)
        continue

    imageKey = 'image-%09d' % cnt
    labelKey = 'label-%09d' % cnt
    cache[imageKey] = imageBin
    cache[labelKey] = label.encode()#######################3ygx GBK#encoding='UTF-8' gb18030
    if lexiconList:
      lexiconKey = 'lexicon-%09d' % cnt
      cache[lexiconKey] = ' '.join(lexiconList[i])
    if cnt % 1000 == 0:
      writeCache(env, cache)
      cache = {}
      print('Written %d / %d' % (cnt, nSamples))
    cnt += 1
  nSamples = cnt-1
  cache['num-samples'] = str(nSamples).encode()
  writeCache(env, cache)
  print('Created dataset with %d samples' % nSamples)

if __name__ == "__main__":
  data_dir = '/media/zj/新加卷/ubuntu/projects/HWNet/datas/recognition/true_data_resize_36/train_recognition/'#'/data/mkyang/datasets/English/benchmark/svtp/'
  lmdb_output_path = '/media/zj/新加卷/ubuntu/projects/aster.pytorch/data/true_data/train'
  gt_file = '/media/zj/新加卷/ubuntu/projects/aster.pytorch/data/train_gt.txt'#os.path.join(data_dir, 'gt.txt')
  image_dir = data_dir
  with open(gt_file, 'r') as f:
    lines = [line.strip('\n') for line in f.readlines()]

  imagePathList, labelList = [], []
  for i, line in enumerate(lines):
    splits = line.split(' ')
    image_name = splits[0]
    gt_text = splits[1]
    # print(image_name, gt_text)
    imagePathList.append(os.path.join(image_dir, image_name+'__'+gt_text+'.jpg'))
    labelList.append(gt_text)

  createDataset(lmdb_output_path, imagePathList, labelList)