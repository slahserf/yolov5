# coding=gbk
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from wfutils.uploadfile import save_to_obs
from models.experimental import attempt_load
from pathlib import Path
from wfutils.wfdatasets import LoadStreams, LoadWebImages, LoadDurationStreams
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from wfutils.general import rotate_non_max_suppression, scale_labels
from wfutils.plots import plot_one_box_chinese, plot_one_rotated_box, plot_one_rotated_box_chinese
from utils.torch_utils import select_device
from wfutils.savepredtrash import make_img
from wfutils.requestget import change_task_static, report, get_dict, get_host_ip
from PIL import Image
import numpy as np
import datetime
from wfutils.entity import get_put_stream_logger
import json

flag2 = False


def _init():
    global flag2
    flag2 = False


def set_value1(value):
    global flag2
    flag2 = value


begin_time = None


def detect(s='', w='', i=640, tid='', mid='', mn='', tt='', vp='', inp='', ext='', no_probability=True):
    source, weights, imgsz, model_name, taskType, videoPaths, imageNamePath, extend, task_id, model_id = s, w, i, mn, tt, vp, inp, ext, tid, mid
    print(source)
    print(taskType)
    print(model_name)
    label_dict = get_dict(model_id)
    if extend == '':
        conf = 0.20
    else:
        extend = json.loads(extend)
        conf = extend['conf']
    # print(label_dict)
    playUrl = [True if x['playUrl'] != "" else False for x in source]
    videoSrc = [True if x['videoSrc'] != "" else False for x in source]
    channel_code = [x['channelCode'] if x['channelCode'] != "" else "" for x in source]
    print(channel_code)

    device = '0'
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # 1 - liveVideo  2 - pastVideo 3 - img

    if taskType == '1':
        cudnn.benchmark = True  # set True to speed up constant image size inference
        try:
            dataset = LoadStreams(source, img_size=imgsz, tid=task_id)
        except Exception:
            raise StopIteration
    else:
        task_static = []
        dataset = LoadWebImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    augment, conf_thres, iou_thres, classes, agnostic_nms, save_conf = False, conf, 0.15, None, False, False
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once\
    if_duplicate = False
    current_pred = None
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference

        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # check if this pred is duplicate with last pred
        t1 = time.time()
        if taskType != '3':
            if current_pred is not None:
                if torch.equal(torch.Tensor([[*xyxy, cls] for *xyxy, conf, cls in pred[0]]),
                               torch.Tensor([[*xyxy, cls] for *xyxy, conf, cls in current_pred])):
                    if_duplicate = True
                else:
                    if_duplicate = False
            else:
                current_pred = pred[0].clone()
                if_duplicate = False

        # t2 = time.time()
        # print('first')
        # print(t2-t1)
        for i, det in enumerate(pred):  # detections per image
            if taskType == '1':  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path
            if taskType == '2':
                save_path = source[i]["videoSrc"]

            # elif taskType == '3':
            #     save_path = imageNamePath  # img.jpg
            #     task_static.append(save_path)

            # t3 = time.time()
            garbage_we_want = 0
            if len(det) and if_duplicate is False:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # check if exist garbages
                garbageResultDict = {}
                # t5 = time.time()
                for c in det[:, -1].unique():
                    label = f'{names[int(c)]}'
                    # print(label)
                    # print(label)
                    # if pred contains the garbage we want
                    if label in label_dict.keys():
                        n = (det[:, -1] == c).sum()  # detections per class
                        nums = int(f'{n}')
                        garbageResultDict[label] = {
                            "checkType": names[int(c)],
                            "checkNum": str(nums),
                            "garbageDirection": "",
                            "garbageSize": "",
                            "garbageDensity": "",
                        }
                        garbage_we_want += 1
                # print(garbageResultDict)
                # t6 = time.time()
                # print('second')
                # print(t6 - t5)
                # check if exist garbage
                if garbage_we_want > 0:
                    # Write results
                    # t7 = time.time()
                    im0 = Image.fromarray(im0)
                    area_dic = {}
                    # t8 = time.time()
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]}'
                        if label in label_dict.keys():
                            if no_probability:
                                chinese_label = label_dict[label]
                            else:
                                chinese_label = label_dict[label] + " " + f'{conf:.2f}'
                            plot_one_box_chinese(xyxy, im0, label=chinese_label, color=colors[int(cls)],
                                                 line_thickness=3)
                            wh = (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))  # w*h
                            if area_dic.get(label) is None:
                                area_dic[label] = wh
                            else:
                                area_dic[label] = area_dic.get(label) + wh
                        # plot_one_box(xyxy, im0, color=colors[int(cls)], line_thickness=3)
                    # t9 = time.time()
                    # print('draw_in')
                    # print(t9 - t8)
                    im0 = np.array(im0)
                    # t10 = time.time()
                    # print('draw_out')
                    # print(t10 - t7)
                    # Print results
                    if taskType == '3':
                        # cv2.imwrite(save_path, im0)
                        # print(save_path)
                        # report_path = save_path
                        report(task_id, taskType, model_id, channel_code[i], garbageResultDict, area_dic, im0, '')
                    elif taskType == '2':
                        # report_path = make_img(im0, imageNamePath)
                        report(task_id, taskType, model_id, channel_code[i], garbageResultDict, area_dic, im0, '')
                        # print('report finish')
                    else:
                        # report_path = make_img(im0, imageNamePath)
                        dataset.report_queue[i].put(
                            [task_id, taskType, model_id, channel_code[i], garbageResultDict, area_dic,
                             im0,
                             ''])
                    # t11 = time.time()
                    # print('report')
                    # print(t11 - t10)
                # t7 = time.time()
                # print('third')
                # print(t7 - t6)
            # t4 = time.time()
            # print('final')
            # print(t4-t3)
            # Stream results
            if taskType == '1':
                # w, h, n = im0.shape
                # im0 = cv2.resize(im0, (int(h / 3), int(w / 3)), interpolation=cv2.INTER_CUBIC)
                psl = get_put_stream_logger()
                if psl[str(task_id) + channel_code[i]]["play"]:
                    # print("det142")
                    dataset.frame_queue[i].put(im0)
                if videoSrc[i]:
                    dataset.queue_list[i].put(im0)
            # elif taskType == '3' and (not len(det) or garbage_we_want == 0):
            #     cv2.imwrite(save_path, im0)
            elif taskType == '2':
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                # t14 = time.time()
                vid_writer.write(im0)
                # t13 = time.time()
                # print('write')
                # print(t13-t14)
                # print('write_in')
                # print(t13-t1)

            if flag2:
                raise StopIteration

    # t17 = time.time()
    # print('total')
    # print(t17-t0)
    if channel_code[0] == '':
        if taskType == '2':
            change_task_static(task_id, '2')

        elif taskType == '3':
            change_task_static(task_id, '2')


def img_detect(s='', w='', i=640, mid='', mn='', inp='', ext='', no_probability=True,
               imageUrlPort=''):
    source, weights, imgsz, model_name, imageNamePath, extend, model_id = s, w, i, mn, inp, ext, mid
    print(source)
    label_dict = get_dict(model_id)
    images_url = [x['url'] for x in source]

    device = '0'
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # 1 - liveVideo  2 - pastVideo 3 - img

    task_static = []
    dataset = LoadWebImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    augment, conf_thres, iou_thres, classes, agnostic_nms, save_conf = False, 0.20, 0.15, None, False, False
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once\
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference

        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # check if this pred is duplicate with last pred
        for i, det in enumerate(pred):  # detections per image

            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # check if exist garbages
                garbageResultDict = {}
                garbage_we_want = 0
                t5 = time.time()
                for c in det[:, -1].unique():
                    label = f'{names[int(c)]}'
                    # print(label)
                    # if pred contains the garbage we want
                    if label in label_dict.keys():
                        n = (det[:, -1] == c).sum()  # detections per class
                        nums = int(f'{n}')
                        garbageResultDict[label] = {
                            "checkType": names[int(c)],
                            "checkNum": str(nums),
                            "garbageDirection": "",
                            "garbageSize": "",
                            "garbageDensity": "",
                        }
                        garbage_we_want += 1
                # check if exist garbage
                if garbage_we_want > 0:
                    # Write results
                    im0 = Image.fromarray(im0)
                    area_dic = {}
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]}'
                        if label in label_dict.keys():
                            if no_probability:
                                chinese_label = label_dict[label]
                            else:
                                chinese_label = label_dict[label] + " " + f'{conf:.2f}'
                            plot_one_box_chinese(xyxy, im0, label=chinese_label, color=colors[int(cls)],
                                                 line_thickness=3)
                            wh = (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))  # w*h
                            if area_dic.get(label) is None:
                                area_dic[label] = wh
                            else:
                                area_dic[label] = area_dic.get(label) + wh
                    im0 = np.array(im0)
                    # Print results
                    imgPath = save_to_obs(im0)
                    imgSize = im0.shape[0] * im0.shape[1]
                    for k in garbageResultDict:
                        garbageSize = int(area_dic[k]) / int(imgSize)
                        garbageResultDict[k]["garbageSize"] = garbageSize if garbageSize <= 1 else 1
                        garbageResultDict[k]["garbageDensity"] = garbageSize if garbageSize <= 1 else 1
                    data = {
                        'garbageResultList': list(garbageResultDict.values()),
                        'imgPath': imgPath,
                    }
                    param = json.dumps(data)
                    return 1, param
                else:
                    result = {'imgPath': images_url[i]}
                    param = json.dumps(result)
                    return 0, param
            else:
                result = {'imgPath': images_url[i]}
                param = json.dumps(result)
                return 0, param


def img_detect_rotate(s='', w='', i=640, mid='', mn='', inp='', ext='', no_probability=True,
                      imageUrlPort=''):
    source, weights, imgsz, model_name, imageNamePath, extend, model_id = s, w, i, mn, inp, ext, mid
    print(source)
    label_dict = get_dict(model_id)
    images_url = [x['url'] for x in source]

    device = '0'
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # 1 - liveVideo  2 - pastVideo 3 - img

    task_static = []
    dataset = LoadWebImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    augment, conf_thres, iou_thres, classes, agnostic_nms, save_conf = False, 0.10, 0.15, None, False, False
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once\
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference

        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = rotate_non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # check if this pred is duplicate with last pred
        for i, det in enumerate(pred):  # detections per image

            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :5] = scale_labels(img.shape[2:], det[:, :5], im0.shape).round()
                # check if exist garbages
                garbageResultDict = {}
                garbage_we_want = 0
                t5 = time.time()
                for c in det[:, -1].unique():
                    label = f'{names[int(c)]}'
                    # print(label)
                    # if pred contains the garbage we want
                    if label in label_dict.keys():
                        n = (det[:, -1] == c).sum()  # detections per class
                        nums = int(f'{n}')
                        garbageResultDict[label] = {
                            "checkType": names[int(c)],
                            "checkNum": str(nums),
                            "garbageDirection": "",
                            "garbageSize": "",
                            "garbageDensity": "",
                        }
                        garbage_we_want += 1
                # check if exist garbage
                if garbage_we_want > 0:
                    # Write results
                    im0 = Image.fromarray(im0)
                    area_dic = {}
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]}'
                        if label in label_dict.keys():
                            if no_probability:
                                chinese_label = label_dict[label]
                            else:
                                chinese_label = label_dict[label] + " " + f'{conf:.2f}'
                        plot_one_rotated_box_chinese(xyxy, im0, label=chinese_label, color=colors[int(cls)],
                                                     line_thickness=3)
                        wh = (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))  # w*h
                        if area_dic.get(label) is None:
                            area_dic[label] = wh
                        else:
                            area_dic[label] = area_dic.get(label) + wh
                    im0 = np.array(im0)
                    # Print results
                    imgPath = save_to_obs(im0)
                    imgSize = im0.shape[0] * im0.shape[1]
                    for k in garbageResultDict:
                        garbageSize = int(area_dic[k]) / int(imgSize)
                        garbageResultDict[k]["garbageSize"] = garbageSize if garbageSize <= 1 else 1
                        garbageResultDict[k]["garbageDensity"] = garbageSize if garbageSize <= 1 else 1
                    data = {
                        'garbageResultList': list(garbageResultDict.values()),
                        'imgPath': imgPath,
                    }
                    param = json.dumps(data)
                    return 1, param
                else:
                    result = {'imgPath': images_url[i]}
                    param = json.dumps(result)
                    return 0, param
            else:
                result = {'imgPath': images_url[i]}
                param = json.dumps(result)
                return 0, param


def detect_for_stream(s='', w='', i=640, tid='', mid='', mn='', tt='', dt='', cc='', ext='', no_probability=True):
    source, weights, imgsz, model_name, taskType, task_id, model_id, duration_time, channel_code, extend = s, w, i, mn, tt, tid, mid, dt, cc, ext
    label_dict = get_dict(model_id)
    extend = json.loads(extend)

    device = '0'
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    try:
        dataset = LoadDurationStreams(source, img_size=imgsz, dt=duration_time)
    except Exception:
        raise StopIteration

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    augment, conf_thres, iou_thres, classes, agnostic_nms, save_conf = False, extend['conf'], 0.15, None, False, False
    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once\
    best_img = None
    best_pred = None
    best_num = 0
    best_garbageResultDict = None
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference

        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        for i, det in enumerate(pred):  # detections per image

            p, s, im0, frame = path, '', im0s, dataset.count

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # check if exist garbages
                garbageResultDict = {}
                temp_num = 0
                for c in det[:, -1].unique():
                    label = f'{names[int(c)]}'
                    # if pred contains the garbage we want
                    if label in label_dict.keys():
                        n = (det[:, -1] == c).sum()  # detections per class
                        nums = int(f'{n}')
                        temp_num += nums
                        garbageResultDict[label] = {
                            "checkType": names[int(c)],
                            "checkNum": str(nums),
                            "garbageDirection": "",
                            "garbageSize": "",
                            "garbageDensity": "",
                        }

                if temp_num > best_num:
                    best_num = temp_num
                    best_pred = det
                    best_img = im0
                    best_garbageResultDict = garbageResultDict
    print("upload report")

    area_dic = {}
    if best_img is None:
        return 201, 'no result'
    best_img = Image.fromarray(best_img)
    for *xyxy, conf, cls in reversed(best_pred):
        label = f'{names[int(cls)]}'
        if label in label_dict.keys():
            if no_probability:
                chinese_label = label_dict[label]
            else:
                chinese_label = label_dict[label] + " " + f'{conf:.2f}'
            plot_one_box_chinese(xyxy, best_img, label=chinese_label, color=colors[int(cls)],
                                 line_thickness=3)
            wh = (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))  # w*h
            if area_dic.get(label) is None:
                area_dic[label] = wh
            else:
                area_dic[label] = area_dic.get(label) + wh

    best_img = np.array(best_img)
    # report_path = make_img(best_img, str(datetime.datetime.now().strftime("%Y-%m-%d")))

    report(task_id, taskType, model_id, channel_code, best_garbageResultDict, area_dic, best_img,
           '')

    return 200, 'detection finished'


def detect_rotate(s='', w='', i=640, tid='', mid='', mn='', tt='', vp='', inp='', ext='', no_probability=True):
    source, weights, imgsz, model_name, taskType, videoPaths, imageNamePath, extend, task_id, model_id = s, w, i, mn, tt, vp, inp, ext, tid, mid
    print(source)
    label_dict = get_dict(model_id)
    extend = json.loads(extend)
    print(label_dict)
    playUrl = [True if x['playUrl'] != "" else False for x in source]
    videoSrc = [True if x['videoSrc'] != "" else False for x in source]
    channel_code = [x['channelCode'] if x['channelCode'] != "" else "" for x in source]

    device = '0'
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # 1 - liveVideo  2 - pastVideo 3 - img

    if taskType == '1':
        cudnn.benchmark = True  # set True to speed up constant image size inference
        try:
            dataset = LoadStreams(source, img_size=imgsz, tid=task_id)
        except Exception:
            raise StopIteration
    else:
        task_static = []
        dataset = LoadWebImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    augment, conf_thres, iou_thres, classes, agnostic_nms, save_conf = False, extend['conf'], 0.45, None, False, False
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once\
    if_duplicate = False
    current_pred = None
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference

        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = rotate_non_max_suppression(pred, conf_thres, iou_thres, classes=classes,
                                          agnostic=agnostic_nms)

        # check if this pred is duplicate with last pred
        t1 = time.time()
        if taskType != '3':
            if current_pred is not None:
                if torch.equal(torch.Tensor([[*xyxy, cls] for *xyxy, conf, cls in pred[0]]),
                               torch.Tensor([[*xyxy, cls] for *xyxy, conf, cls in current_pred])):
                    if_duplicate = True
                else:
                    if_duplicate = False
            else:
                current_pred = pred[0].clone()
                if_duplicate = False

        # t2 = time.time()
        # print('first')
        # print(t2-t1)
        for i, det in enumerate(pred):  # detections per image
            if taskType == '1':  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path
            if taskType == '2':
                save_path = source[i]["videoSrc"]

            elif taskType == '3':
                save_path = imageNamePath  # img.jpg
                task_static.append(save_path)

            # t3 = time.time()
            garbage_we_want = 0
            if len(det) and if_duplicate is False:
                # Rescale boxes from img_size to im0 size
                det[:, :5] = scale_labels(img.shape[2:], det[:, :5], im0.shape).round()
                # check if exist garbages
                garbageResultDict = {}
                # t5 = time.time()
                for c in det[:, -1].unique():
                    label = f'{names[int(c)]}'
                    # print(label)
                    # print(label)
                    # if pred contains the garbage we want
                    if label in label_dict.keys():
                        n = (det[:, -1] == c).sum()  # detections per class
                        nums = int(f'{n}')
                        garbageResultDict[label] = {
                            "checkType": names[int(c)],
                            "checkNum": str(nums),
                            "garbageDirection": "",
                            "garbageSize": "",
                            "garbageDensity": "",
                        }
                        garbage_we_want += 1
                # print(garbageResultDict)
                # t6 = time.time()
                # print('second')
                # print(t6 - t5)
                # check if exist garbage
                if garbage_we_want > 0:
                    # Write results
                    # t7 = time.time()
                    im0 = Image.fromarray(im0)
                    area_dic = {}
                    # t8 = time.time()
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]}'
                        if label in label_dict.keys():
                            if no_probability:
                                chinese_label = label_dict[label]
                            else:
                                chinese_label = label_dict[label] + " " + f'{conf:.2f}'
                            plot_one_rotated_box(xyxy, im0, label=chinese_label, color=colors[int(cls)],
                                                 line_thickness=2)
                            wh = (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))  # w*h
                            if area_dic.get(label) is None:
                                area_dic[label] = wh
                            else:
                                area_dic[label] = area_dic.get(label) + wh
                        # plot_one_box(xyxy, im0, color=colors[int(cls)], line_thickness=3)
                    # t9 = time.time()
                    # print('draw_in')
                    # print(t9 - t8)
                    im0 = np.array(im0)
                    # t10 = time.time()
                    # print('draw_out')
                    # print(t10 - t7)
                    # Print results
                    if taskType == '3':
                        # cv2.imwrite(save_path, im0)
                        # print(save_path)
                        # report_path = save_path
                        report(task_id, taskType, model_id, channel_code[i], garbageResultDict, area_dic, im0, '')
                    elif taskType == '2':
                        # report_path = make_img(im0, imageNamePath)
                        report(task_id, taskType, model_id, channel_code[i], garbageResultDict, area_dic, im0, '')
                    else:
                        # report_path = make_img(im0, imageNamePath)
                        dataset.report_queue[i].put(
                            [task_id, taskType, model_id, channel_code[i], garbageResultDict, area_dic,
                             im0,
                             ''])
                    # t11 = time.time()
                    # print('report')
                    # print(t11 - t10)
                # t7 = time.time()
                # print('third')
                # print(t7 - t6)
            # t4 = time.time()
            # print('final')
            # print(t4-t3)
            # Stream results
            if taskType == '1':
                # w, h, n = im0.shape
                # im0 = cv2.resize(im0, (int(h / 3), int(w / 3)), interpolation=cv2.INTER_CUBIC)
                psl = get_put_stream_logger()
                if psl[str(task_id) + channel_code[i]]["play"]:
                    # print("det142")
                    dataset.frame_queue[i].put(im0)
                if videoSrc[i]:
                    dataset.queue_list[i].put(im0)
            elif taskType == '3' and (not len(det) or garbage_we_want == 0):
                cv2.imwrite(save_path, im0)
            elif taskType == '2':
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                # t14 = time.time()
                vid_writer.write(im0)
                # t13 = time.time()
                # print('write')
                # print(t13-t14)
                # print('write_in')
                # print(t13-t1)

            if flag2:
                raise StopIteration

    # t17 = time.time()
    # print('total')
    # print(t17-t0)
    if channel_code[i] == '':
        if taskType == '2':
            task_static = [x['videoSrc'] for x in source]
            path = task_static[0]
            if len(task_static) > 1:
                path = task_static[0]
                for p in task_static[1:]:
                    path += ',' + p

            change_task_static(task_id, '2', path)

        elif taskType == '3':
            path = task_static[0]
            if len(task_static) > 1:
                path = task_static[0]
                for p in task_static[1:]:
                    path += ',' + p

            change_task_static(task_id, '2', path)
