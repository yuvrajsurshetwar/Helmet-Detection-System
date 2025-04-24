from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.plots import plot_one_box
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.datasets import LoadStreams, LoadImages
from models.experimental import attempt_load
from numpy import random
import torch.backends.cudnn as cudnn
import torch
import cv2
import argparse
import time
from pathlib import Path
from prometheus_client import Counter
import requests
import random
import os
import glob
from sqlalchemy import false
regions = ['mx', 'in']  # Change to your country


files = glob.glob('output/*')
for f in files:
    os.remove(f)


import yagmail
import threading
yag = yagmail.SMTP("autoemailsender2@gmail.com", "tczewxnxfrpviped")

global helmet, plateNumber, noOfPassengers,counter

counter = 0

def sendChalan():

    global helmet, plateNumber, noOfPassengers, counter

    try:

        amount = 0
        challanBreaked = ''
        if helmet=="Not Wearing":
            amount = amount + 500
            challanBreaked = challanBreaked + '<br><br><b>Not Wearing Helmet (500 Rs)</b>'

        if noOfPassengers!='':
            if int(noOfPassengers)>=3:
                amount = amount + 1000
                challanBreaked = challanBreaked + '<br><br><b>There were 3 Passengers in your vehicle (1000 Rs)</b>'
        
        if challanBreaked!='':
            message = "Hi, User<br><br>You have break the following traffic rules "+challanBreaked+"<br><br>Now you have to pay <b>"+str(amount)+"Rs.</b> as Challan."
            attachments='output/Det_'+str(counter)+'.png'
            yag.send(to='Helmetdetection2@gmail.com', subject="Traffic Rule Challan", contents=message,attachments=attachments)
            print('Challan Send','Challan mail send Successfully',message)
    except Exception as e:
        print('Error','Mail not send due to ' + str(e))

def detect(opt):

    global helmet, plateNumber, noOfPassengers,counter

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    cudnn.benchmark = True

    # if source:
        # Opens the inbuilt camera of laptop to capture video.
    cap = cv2.VideoCapture(source)
    framerate = 20
    counter = 0
    imgCounter = 0
    existingOutputs = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        counter += 1

        if counter == framerate:
            print(counter)
            counter = 0

            # This condition prevents from infinite looping
            # incase video ends.
            if ret == False:
                break

            try:

                cv2.imwrite('frames/Frame.jpg', frame)
                source = 'frames/Frame.jpg'

                dataset = LoadImages(source, img_size=imgsz, stride=stride)

                # Get names and colors
                names = model.module.names if hasattr(
                    model, 'module') else model.names
                colors = [[random.randint(0, 255)
                            for _ in range(3)] for _ in names]

                print(names)

                # Run inference
                if device.type != 'cpu':
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                        next(model.parameters())))  # run once
                t0 = time.time()

                for path, img, im0s, vid_cap in dataset:
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=opt.augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(
                        pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                    t2 = time_synchronized()

                    # Apply Classifier
                    if classify:
                        pred = apply_classifier(pred, modelc, img, im0s)

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        s += '%gx%g ' % img.shape[2:]  # print string
                        # normalization gain whwh
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                        if len(det):

                            print(det)

                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(
                                img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                # detections per class
                                n = (det[:, -1] == c).sum()
                                # add to string
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                            # Write results
                            for *xyxy, conf, cls in det:

                                c = int(cls)  # integer class
                                label = f'{names[c]} {conf:.2f}'
                                plot_one_box(
                                    xyxy, im0, label=label, color=colors[c], line_thickness=3)

                                x1, y1, x2, y2 = int(
                                    xyxy[0])-10, int(xyxy[1])-10, int(xyxy[2])+10, int(xyxy[3])+10
                                rider_helmet_status = True
                                
                                # print(names[c])
                                if names[c] == 'Rider':
                                    print(
                                        '\n\nProcessing for rider # ', xyxy)
                                    rider_helmet_status = None
                                    rider_lp_number = None
                                    rider_lp_status = None
                                    no_of_passengers = 0

                                    try:
                                        roi = im0s[y1:y2, x1:x2]
                                        cv2.imwrite('rider.png', roi)
                                    except Exception as e:
                                        x1, y1, x2, y2 = int(xyxy[0]), int(
                                            xyxy[1]), int(xyxy[2]), int(xyxy[3])
                                        roi = im0s[y1:y2, x1:x2]
                                        cv2.imwrite('rider.png', roi)

                                    rid_dataset = LoadImages(
                                        'rider.png', img_size=imgsz, stride=stride)
                                    rid_t0 = time.time()
                                    for rid_path, rid_img, rid_im0s, rid_vid_cap in rid_dataset:
                                        rid_img = torch.from_numpy(
                                            rid_img).to(device)
                                        rid_img = rid_img.half() if half else rid_img.float()  # uint8 to fp16/32
                                        rid_img /= 255.0  # 0 - 255 to 0.0 - 1.0
                                        if rid_img.ndimension() == 3:
                                            rid_img = rid_img.unsqueeze(0)

                                        rid_pred = model(
                                            rid_img, augment=opt.augment)[0]

                                        # Apply NMS
                                        rid_pred = non_max_suppression(
                                            rid_pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

                                        # Apply Classifier
                                        if classify:
                                            rid_pred = apply_classifier(
                                                rid_pred, modelc, rid_img, rid_im0s)

                                        # Process detections
                                        # detections per image
                                        for rid_i, rid_det in enumerate(rid_pred):
                                            rid_p, rid_s, rid_im0, rid_frame = rid_path, '', rid_im0s.copy(), getattr(
                                                rid_dataset, 'frame', 0)

                                            rid_p = Path(rid_p)  # to Path
                                            # print string
                                            rid_s += '%gx%g ' % rid_img.shape[2:]
                                            if len(rid_det):

                                                # print(rid_det)

                                                # Rescale boxes from img_size to im0 size
                                                rid_det[:, :4] = scale_coords(
                                                    rid_img.shape[2:], rid_det[:, :4], rid_im0.shape).round()

                                                # Print results
                                                for rid_c in rid_det[:, -1].unique():
                                                    # detections per class
                                                    rid_n = (
                                                        rid_det[:, -1] == rid_c).sum()
                                                    # add to string
                                                    rid_s += f"{rid_n} {names[int(rid_c)]}{'s' * (rid_n > 1)}, "

                                                # Write results
                                                for *xyxy, rid_conf, cls in rid_det:

                                                    # integer class
                                                    rid_c = int(cls)
                                                    rid_label = f'{names[rid_c]} {rid_conf:.2f}'
                                                    plot_one_box(
                                                        xyxy, rid_im0, label=rid_label, color=colors[rid_c], line_thickness=3)

                                                    rider_helmet_status = False

                                                    if names[rid_c] == "Helmet":
                                                        rider_helmet_status = True
                                                        no_of_passengers = no_of_passengers + 1

                                                    if names[rid_c] == "No Helmet":
                                                        rider_helmet_status = False
                                                        no_of_passengers = no_of_passengers + 1


                                if rider_helmet_status:
                                    helmet = 'Wearing'
                                    
                                    print('\n\nRider wearing Helmet')
                                else:
                                    helmet = 'Not Wearing'

                                    print(
                                        '\n\nRider not wearing Helmet')

                                if rider_lp_status:
                                    print('\nRider having LP')
                                else:
                                    print('\nRider not having LP\n\n')

                                print('\nPlate Number : ',
                                        rider_lp_number)
                                print('\nNo. of passengers : ',
                                        no_of_passengers)

                                noOfPassengers =no_of_passengers



                                if rider_helmet_status == False or no_of_passengers>=3:
                                    print('Voilence found')
                                    cv2.imwrite(
                                            'output/Det_'+str(counter)+'.png', rid_im0)

                                    counter +=1
                        

                                somethingThread = threading.Thread(target=sendChalan)
                                somethingThread.start()

                        cv2.imshow('Output', im0)
                        k = cv2.waitKey(2000) & 0xFF
                        if k == 27:
                            cv2.destroyAllWindows()
                            cap.release()
                            break
            except Exception as e:
                print(e)
        else:
            pass

    cap.release()
    cv2.destroyAllWindows()


def start_detecttion(file=None):

    files = glob.glob('output/*')
    for f in files:
        os.remove(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='./runs/train/finalModel/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='dataset/t_image/', help='source')
    parser.add_argument('--img-size', type=int, default=448,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true',
                        help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    if file != None:
        opt.source = file

    detect(opt)


start_detecttion(0)
