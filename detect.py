from __future__ import absolute_import
import argparse
from sys import platform
import os
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
#====================EAST===================================================
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import EAST,East_res,EAST_tcnn,EAST_acb,EAST_PVANet,EAST_EfficientNet

import os
from dataset import get_rotate_mat
import numpy as np
import lanms
#======================ASTER================================================
import sys
sys.path.append('./')
import argparse
import os
import os.path as osp
import numpy as np
import math
import time
from PIL import Image, ImageFile
from glob import glob
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from config import get_args
from lib import datasets, evaluation_metrics, models
from lib.models.model_builder import ModelBuilder
from lib.datasets.dataset import LmdbDataset, AlignCollate
from lib.loss import SequenceCrossEntropyLoss
from lib.trainers import Trainer
from lib.evaluators import Evaluator
from lib.utils.logging import Logger, TFLogger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.osutils import make_symlink_if_not_exists
from lib.evaluation_metrics.metrics import get_str_list
from lib.utils.labelmaps import get_vocabulary, labels2strs

global_args = get_args(sys.argv[1:])

#====================EAST====================================================
def crop_img(img, length):
    h, w = img.height, img.width
	# confirm the shortest side of image >= length
    if h >= w :
        img = img.resize((length, int(h * length / w) if int(h * length / w)%32==0 else int(h * length / (w*32))*32), Image.BILINEAR)
    elif h < w :
        img = img.resize(( int(w * length / h) if int(w * length / h)%32==0 else int(w * length / (h*32))*32, length), Image.BILINEAR)

    ratio_w = img.width / w
    ratio_h = img.height / h

    return img, ratio_h, ratio_w

def load_pil(img):
	'''convert PIL Image to torch.Tensor
	#(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)
	'''
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])#
	return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
	'''check if the poly in image scope
	Input:
		res        : restored poly in original image
		score_shape: score map shape
		scale      : feature map -> image
	Output:
		True if valid
	'''
	cnt = 0
	for i in range(res.shape[1]):
		if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
	'''restore polys from feature maps in given positions
	Input:
		valid_pos  : potential text positions <numpy.ndarray, (n,2)>
		valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
		score_shape: shape of score map
		scale      : image / feature map
	Output:
		restored polys <numpy.ndarray, (n,8)>, index
	'''
	polys = []
	index = []
	valid_pos *= scale
	d = valid_geo[:4, :] # 4 x N
	angle = valid_geo[4, :] # N,

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		y_min = y - d[0, i]
		y_max = y + d[1, i]
		x_min = x - d[2, i]
		x_max = x + d[3, i]
		rotate_mat = get_rotate_mat(-angle[i])
		
		temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
		temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
		coordidates = np.concatenate((temp_x, temp_y), axis=0)
		res = np.dot(rotate_mat, coordidates)
		res[0,:] += x
		res[1,:] += y
		
		if is_valid_poly(res, score_shape, scale):
			index.append(i)
			polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
	return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):# score_thresh=0.9
	'''get boxes from feature map
	Input:
		score       : score map from model <numpy.ndarray, (1,row,col)>
		geo         : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	score = score[0,:,:]
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape,4) ##################### 4
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
	boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
	return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
	'''refine boxes
	Input:
		boxes  : detected polys <numpy.ndarray, (n,9)>
		ratio_w: ratio of width
		ratio_h: ratio of height
	Output:
		refined boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	return np.around(boxes)
	
	
def detect_TSEAST(img, model, device):
	'''detect text regions of img using model
	Input:
		img   : PIL Image
		model : detection model
		device: gpu if gpu is available
	Output:
		detected polys
	'''
	img, ratio_h, ratio_w = crop_img(img, 160)

	with torch.no_grad():
		score, geo = model(load_pil(img).to(device))
	boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
	return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, xyxy, boxes):

    if  xyxy is None:
        return img

    draw = ImageDraw.Draw(img)
    draw.rectangle(xyxy,outline=(205, 38, 38),width=3)
    if boxes is not None:#if len(boxes)!=0:
        for box in boxes:
            draw.polygon([box[0]+xyxy[0], box[1]+xyxy[1], box[2]+xyxy[0], box[3]+xyxy[1], box[4]+xyxy[0], box[5]+xyxy[1], box[6]+xyxy[0], box[7]+xyxy[1]], outline=(255, 0, 0))
    return img


#======================ASTER================================================

def image_process(image_cv2, imgH=36, imgW=128, keep_ratio=False, min_ratio=1):
    # img = Image.open(image_cv2).convert('RGB')
    img = Image.fromarray(cv2.cvtColor(image_cv2,cv2.COLOR_BGR2RGB))
    if keep_ratio:
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    img = img.resize((imgW, imgH), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)

    return img


class DataInfo(object):
  def __init__(self, voc_type):
    super(DataInfo, self).__init__()
    self.voc_type = voc_type

    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS','Traffic_Sign']
    self.EOS = 'EOS'
    self.PADDING = 'PADDING'
    self.UNKNOWN = 'UNKNOWN'
    self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
    self.char2id = dict(zip(self.voc, range(len(self.voc))))
    self.id2char = dict(zip(range(len(self.voc)), self.voc))

    self.rec_num_classes = len(self.voc)

#===================================================

def detect_NSyolov3(save_txt=False, save_img=True):
    img_size = (960, 960) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img,save_img,save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img,opt.save_img,opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)
    print('Load NSYOLOv3 Model ...')
    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)
    # Eval mode
    model.to(device).eval()
    print('NSYOLOv3 加载成功!')
    model_TSEAST = EAST_PVANet(inception_mid = False,inception_end = True,version=1,conv1_5=False,acb_block = False,dcn =False,with_modulated_dcn=True).to(device)
    print('Load  TSEAST Model ...')
    model_TSEAST.load_state_dict(torch.load('pths/TSEAST.pth'))
    model_TSEAST.to(device).eval()
    print('TSEAST 加载成功!')

    np.random.seed(1001)
    torch.manual_seed(1001)
    torch.cuda.manual_seed(1001)
    torch.cuda.manual_seed_all(1001)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    dataset_info = DataInfo('Traffic_Sign')
    print('Load  ASTER Model ...')
    # Create model
    model_ASTER = ModelBuilder(arch='ResNet_ASTER', rec_num_classes=dataset_info.rec_num_classes,
                        sDim=512, attDim=512, max_len_labels=22,
                        eos=dataset_info.char2id[dataset_info.EOS], STN_ON=True)
    model_ASTER.load_state_dict(torch.load('pths/ASTER.pth'))
    device = torch.device("cuda")
    model_ASTER = model_ASTER.to(device)
    model_ASTER = nn.DataParallel(model_ASTER)
    model_ASTER.eval()
    print('ASTER 加载成功!')
    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'pths/export.onnx', verbose=True)
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = opt.save_img
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = ['Text-Based Traffic Sign']#load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred, _ = model(img)

        if opt.half:
            pred = pred.float()
     
        for i, det in enumerate(non_max_suppression(pred, opt.conf_thres, opt.nms_thres)):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s
            print(s)
            image_ori_PIL = Image.fromarray(cv2.cvtColor(im0,cv2.COLOR_BGR2RGB))
            plot_img = image_ori_PIL
            save_path = str(Path(out) / Path(p).name)
            # s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '检测到 %g %s' % (n, '个文字类型交通标志')  # add to string
                print(s)
                # Write results
                for *xyxy, conf, _, cls in det:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    img_east = image_ori_PIL.crop(list(map(int,xyxy)))
                    boxes = detect_TSEAST(img_east, model_TSEAST, device)
                    # if boxes is None:
                    #     # print('图片中 部分交通牌上 未检测 到文字 ! ', end = ' ')
                    #     continue
                    plot_img = plot_boxes(plot_img,xyxy,boxes)############画图
                    if boxes is not None and xyxy is not None:
                        for i,box in enumerate (boxes):
                            pts1 = np.float32([[box[0]+xyxy[0], box[1]+xyxy[1]], [box[2]+xyxy[0], box[3]+xyxy[1]], [box[4]+xyxy[0], box[5]+xyxy[1]], [box[6]+xyxy[0], box[7]+xyxy[1]]])
                            w1 = np.sqrt(np.sum((box[2]-box[0])**2))
                            w2 = np.sqrt(np.sum((box[6]-box[4])**2))
                            h1 = np.sqrt(np.sum((box[7]-box[1])**2))
                            h2 = np.sqrt(np.sum((box[5]-box[3])**2))
                            w = int((w1+w2)//2)
                            h = int((h1+h2)//2)
                            pts2 = np.float32(([0,0],[w,0],[w,h],[0,h]))
                            M = cv2.getPerspectiveTransform(pts1,pts2)
                            dst = cv2.warpPerspective(im0,M,(w,h))
                            img = image_process(dst)
                            # cv2.imwrite('/home/zj/OCR/projects/EAST/ICDAR_2015/temp/'+str(i)+'.jpg',dst)
                            with torch.no_grad():
                                img = img.cuda()
                            input_dict = {}
                            input_dict['images'] = img.unsqueeze(0)
                            rec_targets = torch.IntTensor(1, 22).fill_(1)
                            rec_targets[:,22-1] = dataset_info.char2id[dataset_info.EOS]
                            input_dict['rec_targets'] = rec_targets
                            input_dict['rec_lengths'] = [22]
                            output_dict = model_ASTER(input_dict)
                            pred_rec = output_dict['output']['pred_rec']
                            pred_str, _ = get_str_list(pred_rec, input_dict['rec_targets'], dataset=dataset_info)
                            print('Recognition result: {0} '.format(pred_str[0]),end=' ') 
                            box =list(map(int,[box[0]+xyxy[0], box[1]+xyxy[1], box[2]+xyxy[0], box[3]+xyxy[1], box[4]+xyxy[0], box[5]+xyxy[1], box[6]+xyxy[0], box[7]+xyxy[1]]))
                            print(box,sep=',')
                            if save_txt:  # Write to file
                                
                                with open(str(Path(out))+'/'  + 'results.txt', 'a') as file:
                                    file.write(('%s %s %g %g %g %g %g %g %g %g '  + '\n') % (path,pred_str[0]  ,*box))
                if save_img:
                    plot_img.save(save_path)
            else:
                print('图片中 未检测 到文字型交通标志 !', end = ' ')

            print('Done. (%.3fs)' % (time.time() - t))

            # Stream results
            # if view_img:
            #     cv2.imshow(p, im0)

            # # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'images':
            #         cv2.imwrite(save_path, im0)
            #     else:
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer

            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
            #         vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('All Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='cfg/prune_0.9_yolov3-TTK.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/TTK.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='pths/prune_0.9_yolov3-TTK.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='./results', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='目标检测的置信度')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='目标检测')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save_img', type=bool, default=False)
    parser.add_argument('--save_txt', type=bool, default=True)

    opt = parser.parse_args()
    print('')
    print('-------------------------------------------------------------自然场景下交通标志文本识别系统--------------------------------------------------------------')
    print('')
    print(opt)

    with torch.no_grad():
        detect_NSyolov3()
