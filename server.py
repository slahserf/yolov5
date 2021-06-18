import threading
import time
from configparser import ConfigParser

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import WFdetect
from wfutils.requestget import get_host_ip
from wfutils.controlthread import stop_thread
from wfutils.entity import set_put_stream_logger, get_put_stream_logger, set_task_thread_status, get_task_thread_status, \
    remove_task_thread_status, remove_put_stream_logger

import uvicorn

cfg = ConfigParser()
# print(os.path.abspath('.'))
cfg.read('./config/server_info.ini')
# print(cfg.sections())
port = cfg.get('server', 'port')
imageUrlPort = cfg.get('server', 'imageUrlPort')

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


class EchoArgs(BaseModel):
    taskId: int
    modelId: int
    modelName: str
    taskType: str
    taskUrlInfoList: list
    imageNamePath: str
    extend: str


class TaskArgs(BaseModel):
    taskId: str
    channelCodeList: list


class PushArgs(BaseModel):
    taskId: str
    channelCode: str
    status: str
    playUrl: str


class ImgDetectArgs(BaseModel):
    modelId: int
    modelName: str
    taskUrlInfoList: list
    imageNamePath: str
    extend: str


class DurationDetectionArgs(BaseModel):
    taskId: int
    modelId: int
    modelName: str
    taskType: str
    channelCode: str
    videoUrl: str
    durationTime: int
    durationUnit: str
    extend: str


@app.post("/startTask/")
async def stream_detection(args: EchoArgs):
    try:
        thread_01 = threading.Thread(target=detection, args=(
            args.taskType, args.taskUrlInfoList, args.imageNamePath, args.extend, args.modelName,
            args.taskId, args.modelId),
                                     daemon=True)
        thread_01.start()
        set_task_thread_status(args.taskId, thread_01.ident)
        # detection(args.taskType, args.taskUrlInfoList, args.imageNamePath, args.extend, args.modelName,
        #           args.taskId, args.modelId)
    except Exception:
        print('stream_detection exception')
        return {"code": 500, "msg": "error"}
    return {"code": 200, "msg": "success"}


@app.post("/imgTask/")
async def img_detection(args: ImgDetectArgs):
    try:
        weight = './model/' + args.modelName + '.pt'
        if args.modelName == 'rotatebest':
            status, param = WFdetect.img_detect_rotate(s=args.taskUrlInfoList, w=weight, mn=args.modelName,
                                                       inp=args.imageNamePath, ext=args.extend,
                                                       mid=args.modelId, imageUrlPort=imageUrlPort)
        else:
            status, param = WFdetect.img_detect(s=args.taskUrlInfoList, w=weight, mn=args.modelName,
                                                inp=args.imageNamePath, ext=args.extend,
                                                mid=args.modelId, imageUrlPort=imageUrlPort)
    except Exception as e:
        print('stream_detection exception')
        print(e)
        return {"code": 500, "msg": e}
    return {"code": 200, "status": status, "msg": param}


def detection(taskType, taskUrlInfoList, imageNamePath, extend, modelName, taskId, modelId):
    WFdetect.set_value1(False)
    weight = './model/' + modelName + '.pt'
    if modelName == 'rotatebest':
        WFdetect.detect_rotate(s=taskUrlInfoList, w=weight, mn=modelName, tt=taskType, inp=imageNamePath, ext=extend,
                               tid=taskId,
                               mid=modelId)
    else:
        WFdetect.detect(s=taskUrlInfoList, w=weight, mn=modelName, tt=taskType, inp=imageNamePath, ext=extend,
                        tid=taskId,
                        mid=modelId)


@app.post("/StreamDurationDetection/")
async def stream_duration_detection(args: DurationDetectionArgs):
    try:
        # print(args)
        thread_stream_duration = threading.Thread(target=SD_detection, args=(args.videoUrl, args.modelName,
                                                                             args.durationUnit, args.durationTime,
                                                                             args.taskType, args.taskId,
                                                                             args.modelId, args.channelCode,
                                                                             args.extend), daemon=True)
        thread_stream_duration.start()
        # print(args.taskId)
        # weight = './model/' + args.modelName + '.pt'
        # if args.durationUnit == 's':
        #     durationTime = args.durationTime
        # elif args.durationUnit == 'm':
        #     durationTime = args.durationTime * 60
        # elif args.durationUnit == 'h':
        #     durationTime = args.durationTime * 60 * 60
        # code, msg = WFdetect.detect_for_stream(s=args.videoUrl, w=weight, mn=args.modelName,
        #                                        dt=durationTime, tt=args.taskType, tid=args.taskId,
        #                                        mid=args.modelId, cc=args.channelCode, ext=args.extend)
    except Exception as e:
        print('stream_detection exception')
        print(e)
        return {"code": 500, "msg": e}

    # print('stream_detection finish')
    return {"code": 200, "msg": 'Ok'}


def SD_detection(videoUrl, modelName,
                 durationUnit, durationTime,
                 taskType, taskId,
                 modelId, channelCode,
                 extend):
    weight = './model/' + modelName + '.pt'
    if durationUnit == 's':
        dT = durationTime
    elif durationUnit == 'm':
        dT = durationTime * 60
    elif durationUnit == 'h':
        dT = durationTime * 60 * 60
    code, msg = WFdetect.detect_for_stream(s=videoUrl, w=weight, mn=modelName,
                                           dt=dT, tt=taskType, tid=taskId,
                                           mid=modelId, cc=channelCode, ext=extend)
    print(code, msg)


@app.post("/closeTask/")
async def close_task(args: TaskArgs):
    try:
        for i in get_task_thread_status(args.taskId):
            print(i)
            stop_thread(i)
        remove_task_thread_status(args.taskId)
        for c in args.channelCodeList:
            remove_put_stream_logger(args.taskId, c)
    except Exception:
        print(Exception)
    return {"msg": "success"}


@app.post("/changePushStatus/")
async def change_push_status(args: PushArgs):
    if args.taskId + args.channelCode in get_put_stream_logger().keys():
        set_put_stream_logger(args.taskId, args.channelCode, args.playUrl, bool(int(args.status)))
        print(get_put_stream_logger())
    else:
        print("No Task")
        return {"code": 500, "msg": "No Task"}
    time.sleep(3)
    return {"code": 200, "msg": "success"}


if __name__ == '__main__':
    uvicorn.run(app, host=get_host_ip(), port=int(port))
