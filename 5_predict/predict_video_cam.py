import albumentations as alb
import albumentations.pytorch as albp
import copy
import cv2
import json
import glob
import numpy as np
import math
import torch
#from torchvision.ops import box_iou

from ultralytics import YOLOv10
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from fix_model_state_dict import fix_model_state_dict

IMSHOW = True
SAVE_IMAGE = True

def intersect(bb0, bb1):
    x_min = max(bb0[0], bb1[0])
    y_min = max(bb0[1], bb1[1])
    x_max = min(bb0[2], bb1[2])
    y_max = min(bb0[3], bb1[3])
    w = max(0, x_max - x_min + 0.1)
    h = max(0, y_max - y_min + 0.1)
    intersect = w * h
    
    return intersect

def box_dist(bb0, bb1):
    cx0 = (bb0[0] + bb0[2]) / 2
    cy0 = (bb0[1] + bb0[3]) / 2
    cx1 = (bb1[0] + bb1[2]) / 2
    cy1 = (bb1[1] + bb1[3]) / 2
    
    dist = math.sqrt((cx0 - cx1) ** 2 + (cy0 - cy1) ** 2)
    
    return dist


if __name__ == "__main__":
    INPUT_PATH = '/input'
    OUTPUT_PATH = '/output'
 
#    input_video = glob.glob(f'{INPUT_PATH}/*.mp4')[0]
    input_video = './vid_1_short.mp4'
#    input_video = '/data/MICCAI2024_SurgVU/videos/case_000/case_000_video_part_001.mp4'
#    input_video = '/data/MICCAI2024_SurgVU/videos/case_001/case_001_video_part_001.mp4'
#    input_video = '/data/MICCAI2024_SurgVU/videos/case_002/case_002_video_part_001.mp4'
#    input_video = '/data/MICCAI2024_SurgVU/videos/case_003/case_003_video_part_001.mp4'
#    input_video = '/data/MICCAI2024_SurgVU/videos/case_004/case_004_video_part_001.mp4'
#    input_video = '/data/MICCAI2024_SurgVU/videos/case_005/case_005_video_part_001.mp4'
    
    tool_names = ['force_bipolar',  # 0 (cls)
                  'needle_driver',  # 1 (cls), 0 (det)
                  'monopolar_curved_scissors',  # 2 (cls), 1 (det)
                  'clip_applier',  # 3 (cls), 2 (det)
                  'bipolar_forceps',   # 4 (cls), 3 (det)
                  'vessel_sealer',   # 5 (cls), 4 (det)
                  'prograsp_forceps',   # 6 (cls), 5 (det)
                  'grasping_retractor',   # 7 (cls), 6 (det)
                  "cadiere_forceps",   # 8 (cls)
                  "permanent_cautery_hook_spatula", # 9 (cls)
                  "stapler", # 10 (cls)
                  "tip_up_fenestrated_grasper", # 11 (cls)
                  'unknown'] # 12 (cls)
                  
    json_dict = {}
    json_dict["type"] = "Multiple 2D bounding boxes"
    json_dict["version"] = {"major": 1, "minor": 0} 
    box_list = []

    # model for parts detection (with loading pretrained weights)
    pretrained_det_p = './yolov10b_parts_best.0831.pt'
    model_det_p = YOLOv10(pretrained_det_p)
    
    # model for classification
    from timm_model import TimmModel
    params_cls = {'model_name': 'convnext_small.fb_in22k_ft_in1k_384',
                  'pretrained': False,
                  'in_channels': 3,
                  'num_classes': 13}
    model_cls = TimmModel(**params_cls)

    # Load pretrained model weights
    pretrained_cls = '240827_082856_convnext_v1-LR0.0001-small_freeze_stem-epoch=07-valid_MultilabelF1Score=0.66.ckpt'
    checkpoint = torch.load(pretrained_cls, weights_only=True, map_location='cpu')
    state_dict = fix_model_state_dict(checkpoint['state_dict'])
    model_cls.load_state_dict(state_dict)
    model_cls.cuda()
    model_cls.eval()

    # Construct the CAM object once, and then re-use it on many images.
    cam = ScoreCAM(model=model_cls, target_layers=[model_cls.model.stages[2]])
    
    # preprocess for classification
    resize_width = 384
    resize_height = 384
#    roi1 = [0, 0, 898, 56]
#    roi2 = [0, 671, 898, 720]
#    pad_size = [8, 0, 8, 0]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    max_pixel_value = 255.0
    crop_bottom = 40
    
    transform_cls = alb.Compose([alb.Resize(height=resize_height,
                                            width=resize_width,
                                            interpolation=cv2.INTER_CUBIC,
                                            p=1.0),
                                 alb.Normalize(mean=mean,
                                               std=std,
                                               max_pixel_value=max_pixel_value),
                                 albp.ToTensorV2()])
    
    # read an image from the input video
    cap = cv2.VideoCapture(input_video)
    ret, image = cap.read()
    frame_number = 0
    
    org_height, org_width, _ = image.shape 
#    org_width -= (192 * 2)
    
    while ret:
        # center crop
#        image = image[:, 192:-192, :]
        
        if SAVE_IMAGE:
            save_image_filename = f"./output/{frame_number:04d}_input.jpg"
            cv2.imwrite(save_image_filename, image)
        
        # detection
        result_p = model_det_p.track(image, persist=True, show=False, conf=0.3, iou=0.5)

        # classification
        image_cls = copy.copy(image)
        image_cls[-crop_bottom:, :, :] = 0  # black
        image_cls = transform_cls(image=image[:, :, ::-1])['image']
        image_cls = image_cls.unsqueeze(0).cuda()

        with torch.no_grad():
            pred_cls = model_cls(image_cls)

        # CAM
        pred_cls = pred_cls[0].cpu().numpy()
        pred_cls_binary = (pred_cls >= 0.0)
        print("pred_cls:", pred_cls_binary.astype(np.uint8))
        cam_bbs = []
        for idx, clas in enumerate(pred_cls_binary):
            if clas:
                targets = [ClassifierOutputTarget(idx)]
#                print("tool:", tool_names[idx])
                
                # CAM (can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing)
                gs_cam = cam(input_tensor=image_cls, targets=targets, aug_smooth=False)
                gs_cam = gs_cam[0, :]
                gs_cam_resize = cv2.resize(gs_cam, (org_width, org_height))
                
                maxval = gs_cam_resize.max()
#                print("grayscale max:", maxval)
                
                # get maximum response
                gs_binary = (gs_cam_resize > maxval * 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(gs_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if IMSHOW:
                   image_copy = copy.copy(image)
                   
                for _, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    cam_bb = {'class_id': idx,
                              'xyxy': [x, y, x+w, y+h],
                              'cam_value': gs_cam_resize[y:y+h, x:x+w].max()}
                    cam_bbs.append(cam_bb) 
#                    cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 255, 255), 1)

                if IMSHOW:
#                    cv2.imshow("cam_gray", gs_cam_resize)
#                    cv2.imshow("image_cls", image_copy[:, :, ::-1])

                    # 
#                    image_to_show = np.clip((image_cls[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0, 0, 1)
                    image_to_show = cv2.resize(image_copy, (resize_width, resize_height))
                    image_to_show = image_to_show[:, :, ::-1] / 255.0
                    visualization = show_cam_on_image(image_to_show, gs_cam, use_rgb=True)
                    # You can also get the model outputs without having to redo inference
#                    model_outputs = cam.outputs
                    # imshow
                    cam_image = cv2.resize(visualization, (org_width, org_height))
                    cv2.imshow("cam", cam_image[:, :, ::-1])
                    cv2.waitKey(10)
                    
                    if SAVE_IMAGE:
                        save_image_filename = f"./output/{frame_number:04d}_cam_{idx:02d}.jpg"
                        cv2.imwrite(save_image_filename, cam_image[:, :, ::-1])

        boxes = result_p[0].boxes.cpu()
        print("boxes.cls:", boxes.cls)
#        print("boxes.xyxy:", boxes.xyxy)

        # prepare for grouping
        if len(boxes.cls) == 0:
            classes = []
            groups = []
        else:
            classes = boxes.cls.numpy()
            groups = []

            if IMSHOW:
                image_copy1 = copy.copy(image)
                image_copy2 = copy.copy(image)
                for box in boxes.xyxy:
                    box = box.numpy()
                    cv2.rectangle(image_copy1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                                  color=(255,255,255), thickness=1)
                if SAVE_IMAGE:
                    save_image_filename = f"./output/{frame_number:04d}_parts.jpg"
                    cv2.imwrite(save_image_filename, image_copy1)

        # grouping
        num_parts = len(classes)
        group_ids = [-1] * num_parts
        # 1. find wrists
        group_num = 0
        for idx in range(len(classes)):
            if classes[idx] == 1:
                # wrist -> make a new group
                groups.append([idx])
                group_ids[idx] = group_num
                group_num += 1
            
        # 2. grouping other parts
        for idx0 in range(len(classes)):
            if group_ids[idx0] < 0:
                # check the intersect
                max_isect = -1
                argmax_isect = None
                for idx1 in range(len(classes)):
                    if idx1 != idx0 and group_ids[idx1] >= 0:
                        isect = intersect(boxes.xyxy[idx0], boxes.xyxy[idx1])
                        if isect > 0 and isect > max_isect:
                            max_isect = isect
                            argmax_isect = idx1
                # grouping
                if argmax_isect is None:
                    # make a new group
                    groups.append([idx0])
                    group_ids[idx0] = group_num
                    group_num += 1
                else:
                    # add to a group
                    group_id = group_ids[argmax_isect]
                    groups[group_id].append(idx0)
                    
        print("groups:", groups)

        # for each group
        group_class_id = [-1] * len(groups)
        for group_id, group in enumerate(groups):
            idx = group[0]
            # identify representative box            
            # check (wrist -> clasper -> shaft)
            for parts_id in [1, 2, 0]:
                found_flag = False
                for idx in group:
                    if classes[idx] == parts_id:
                        bb = boxes.xyxy[idx].numpy()
                        parts = parts_id
                        prob = boxes.conf[idx].item()
                        found_flag = True
                        break

                if found_flag == True:
                    break

            # identify tool
            max_cam_value = -1
            argmax_tool_id = -1
            for idx in group:
                for cam_bb in cam_bbs:
                    isect = intersect(boxes.xyxy[idx],
                                      cam_bb['xyxy'])
                    if isect > 0:
                        cam_value = cam_bb['cam_value']
                        if cam_value > max_cam_value:
                            max_cam_value = cam_value
                            argmax_tool_id = cam_bb['class_id']
            
            # when not identified
            if argmax_tool_id == -1 and frame_number > 0:
                # find closest bbox from the previous frame
                min_total_dist = 1e10
                argmin_prev_group_id = -1
                for prev_group_id, prev_group in enumerate(prev_groups):
                    total_dist = 0
                    total_dist_cnt = 0
                    for idx0 in group:
                        for idx1 in prev_group:
                            if boxes.cls[idx0] == prev_boxes.cls[idx1]:
                                total_dist += box_dist(boxes.xyxy[idx0],
                                                       prev_boxes.xyxy[idx1])
                                total_dist_cnt += 1
                    if total_dist_cnt == 0:
                        continue
                    total_dist /= total_dist_cnt
                    if total_dist < min_total_dist:
                        min_total_dist = total_dist
                        argmin_prev_group_id = prev_group_id

                if argmin_prev_group_id >= 0:
                    argmax_tool_id = prev_group_class_id[argmin_prev_group_id]
                    print("Unknown ->", tool_names[argmax_tool_id])
            
            group_class_id[group_id] = argmax_tool_id
            
            # convert to tool name
            tool_name = tool_names[argmax_tool_id]

            box = {}
            lx = float(bb[0])
            ty = float(bb[1])
            rx = float(bb[2])
            by = float(bb[3])
            corners = [[lx, ty, 0.5],
                       [rx, ty, 0.5],
                       [rx, by, 0.5]                           ,
                       [lx, by, 0.5]]
            box["corners"] = corners
            box["name"] = f'slice_nr_{frame_number}_{tool_name}'
            box["probability"] = float(prob)
                    
            box_list.append(box)
        
            if IMSHOW:
                cv2.rectangle(image_copy1, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), 
                              color=(0,255,0), thickness=2)
#                cv2.putText(image_copy1, tool_name, (int(bb[0]), int(bb[1]) - 5),
                cv2.putText(image_copy1, tool_name, (int(bb[0]), int(bb[3] + 15)),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
                cv2.rectangle(image_copy2, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), 
                              color=(0,255,0), thickness=2)
#                cv2.putText(image_copy2, tool_name, (int(bb[0]), int(bb[1]) - 5),
                cv2.putText(image_copy2, tool_name, (int(bb[0]), int(bb[3] + 15)),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

        # copy info for the next frame
        prev_groups = groups
        prev_group_class_id = group_class_id
        prev_boxes = boxes
        
        if IMSHOW:
            # imshow
            cv2.imshow("result", image_copy1)
            cv2.waitKey(100)
            
            if SAVE_IMAGE:
                save_image_filename = f"./output/{frame_number:04d}_results1.jpg"
                cv2.imwrite(save_image_filename, image_copy1)
                save_image_filename = f"./output/{frame_number:04d}_results2.jpg"
                cv2.imwrite(save_image_filename, image_copy2)

        frame_number += 1
        ret, image = cap.read()
        
    # write json file
    json_dict["boxes"] = box_list
    
    print("json:", json_dict)
    
    json_filename = f'./surgical-tools.json'
    with open(json_filename, "w") as f:
        json.dump(json_dict, f, indent=4)  
        