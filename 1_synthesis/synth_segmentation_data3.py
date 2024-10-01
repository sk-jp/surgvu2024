import copy
import cv2
import glob
import numpy as np
import os
import random


"""
def merge_image(fg, bg, fg_mask):
    _, mask = cv2.threshold(fg_mask, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    fg_image = cv2.bitwise_and(fg, fg, mask=mask)
    bg_image = cv2.bitwise_and(bg, bg, mask=mask_inv)
    image = cv2.add(bg_image, fg_image)
    return image
"""
def merge_image(fg, bg, fg_mask, mask_id):
    height, width = fg_mask.shape
    mask = (fg_mask == mask_id).reshape(height, width).astype(np.uint8) * 255
#    _, mask = cv2.threshold(fg_mask, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    fg_image = cv2.bitwise_and(fg, fg, mask=mask)
    bg_image = cv2.bitwise_and(bg, bg, mask=mask_inv)
    image = cv2.add(bg_image, fg_image)
    return image

"""
def merge_mask(cur_mask, add_mask):
    _, mask = cv2.threshold(add_mask, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    add_mask = cv2.bitwise_and(add_mask, add_mask, mask=mask)
    cur_mask = cv2.bitwise_and(cur_mask, cur_mask, mask=mask_inv)
    merged_mask = cv2.add(cur_mask, add_mask)
    return merged_mask
"""

def merge_mask(cur_type_mask, add_type_mask, cur_parts_mask, add_parts_mask, type_id):
    height, width = add_type_mask.shape
    mask = (add_type_mask == type_id).reshape(height, width).astype(np.uint8) * 255
#    _, mask = cv2.threshold(add_mask, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    add_type_mask = cv2.bitwise_and(add_type_mask, add_type_mask, mask=mask)
    cur_type_mask = cv2.bitwise_and(cur_type_mask, cur_type_mask, mask=mask_inv)
    merged_type_mask = cv2.add(cur_type_mask, add_type_mask)
    add_parts_mask = cv2.bitwise_and(add_parts_mask, add_parts_mask, mask=mask)
    cur_parts_mask = cv2.bitwise_and(cur_parts_mask, cur_parts_mask, mask=mask_inv)
    merged_parts_mask = cv2.add(cur_parts_mask, add_parts_mask)   
    
    return merged_type_mask, merged_parts_mask, add_type_mask, add_parts_mask

def merge_bbox(bbox0, bbox1):
    sx0, sy0, w0, h0 = bbox0
    sx1, sy1, w1, h1 = bbox1
    ex0 = sx0 + w0
    ey0 = sy0 + h0
    ex1 = sx1 + w1
    ey1 = sy1 + h1
    
    sx = min(sx0, sx1)
    ex = max(ex0, ex1)
    sy = min(sy0, sy1)
    ey = max(ey0, ey1)
    
    return [sx, sy, ex-sx, ey-sy]


def get_type_label_val(type_mask, bbox):
    x, y, w, h = bbox
    types = type_mask[y:y+h, x:x+w]
    types, counts = np.unique(types, return_counts=True)
    type_count = {}
    for i in range(len(types)):
        type_count[types[i]] = counts[i]
    type_count = sorted(type_count.items(), reverse=True)
#    print("types:", types, counts, type_count)

    type_label_val = 0
    for i in range(len(type_count)):
        if type_count[i][0] != 0:
            type_label_val = type_count[i][0]
            break

    return type_label_val

    
def get_parts_bbox(parts_mask, type_mask):
    parts_label_vals = [1, 2, 3]
    bboxes = []        
    for parts_label_val in parts_label_vals:
        bbox = []
        parts_label_img = (parts_mask==parts_label_val).astype(np.uint8)
        contours, _ = cv2.findContours(parts_label_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for _, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if len(bbox) == 0:
                bbox = [x, y, w, h]
            else:
                bbox = merge_bbox(bbox, [x, y, w, h])
        
        if len(bbox) != 0:
            type_label_val = get_type_label_val(type_mask, bbox)
#            bbox.insert(0, parts_label_val-1)
            label_val = (type_label_val - 1) * 3 + (parts_label_val - 1)
            bbox.insert(0, label_val)
            bboxes.append(bbox)
    return bboxes

                    
if __name__ == '__main__':
    bg_dir = '/data/semi-synthetic/background_surgvu'
    fg_topdirs = ['/data/EndoVisSub2017-RoboticInstrumentSegmentation/crop_image_mask',
                  '/data/EndoVisSub2018-RoboticSceneSegmentation/crop_image_mask']
    dst_topdir = '/data/MICCAI2024_SurgVU/synthetic3'

#    num_images_to_create = 30000
    num_images_per_bg_image = 20    # total number is about (num_images_per_bg_image*50*50)

    # foreground images
    fg_images = []
    fg_images.extend(sorted(glob.glob(f'{fg_topdirs[0]}/images/*/*.png')))
    fg_images.extend(sorted(glob.glob(f'{fg_topdirs[1]}/images/*/*.png')))

    # categoraize foreground images
    fg_image_files = dict()
    fg_count = dict()
    for t in [1, 2, 3, 4, 5, 6]:
        for p in ['l', 'r', 'b']:
            fg_image_files[f'{t}{p}'] = []
            fg_count[f'{t}{p}'] = 0
        
    for fg_image in fg_images:
        basename = os.path.basename(fg_image)
        tp = basename.split('_')[3][1] + basename.split('_')[4][0]
        fg_image_files[tp].append(fg_image)
    #    fg_count[tp] += 1

    # count
    for tp, files in fg_image_files.items():
        fg_count[tp] = len(files)
    print("fg_count:", fg_count)

    """
    # find max
    max_fg_count = 0
    for val in fg_count.values():
        if val > max_fg_count:
            max_fg_count = val
    """

    zero_fg_count = 5
    min_fg_count = 100

    """
    # increase the number of data (by copy)
    for tp, count in fg_count.items():
        num_copies = max_fg_count // count - 1
        copy_list = copy.copy(fg_image_files[tp])
        for _ in range(num_copies):
            fg_image_files[tp].extend(copy_list)
    """
    # increase the number of data (by copy)
    for tp, count in fg_count.items():
        if count < zero_fg_count:
            # remove the tool
            fg_image_files[tp] = []
        elif count < min_fg_count:
            num_copies = int(min_fg_count / count) - 1
            copy_list = copy.copy(fg_image_files[tp])
            for _ in range(num_copies):
                fg_image_files[tp].extend(copy_list)
        """
        elif count > max_fg_count:
            fg_image_files[tp] = random.sample(fg_image_files[tp], max_fg_count)
        elif count < max_fg_count / 2:
            num_copies = int(max_fg_count / 2 / count) - 1
            copy_list = copy.copy(fg_image_files[tp])
            for _ in range(num_copies):
                fg_image_files[tp].extend(copy_list)
        """
        
    # count again
    for tp, files in fg_image_files.items():
        fg_count[tp] = len(files)
    print("fg_count:", fg_count)

    # merge to a list
    fg_image_files_p = {}
    num_fg_image_files_p = {}
    for p in ['l', 'r', 'b']:
        fg_image_files_p[p] = []
        for t in [1, 2, 3, 4, 5, 6]:
            fg_image_files_p[p].extend(fg_image_files[f'{t}{p}'])
        num_fg_image_files_p[p] = len(fg_image_files_p[p])

    # backgroud images
    bg_subdirs = sorted(glob.glob(f'{bg_dir}/*'))
    bg_types = []
    for bg_subdir in bg_subdirs:
        bg_types.append(bg_subdir.replace(bg_dir + '/', ''))

    bg_image_files = {}
    for bg_type in bg_types:        
        bg_image_files[bg_type] = []
        bg_files = sorted(glob.glob(f'{bg_dir}/{bg_type}/*.jpg'))
        for bg_file in bg_files:
            bg_image_files[bg_type].append(bg_file)

    # synthsize images
    width = 896
    height = 720
    black_y = 671

    for bg_type in bg_types:
        print("bg:", bg_type)
        
        # select BG images
        bg_sel_files = bg_image_files[bg_type]
            
        # for each selected BG image
        for bg_sel_file in bg_sel_files:
                
            cnt_images = 0
            while cnt_images < num_images_per_bg_image:
                
                dst_filename = f"{bg_type}_{os.path.basename(bg_sel_file).split('.')[0]}"
                
                bg_sel_image = cv2.imread(bg_sel_file)
                bg_sel_image[black_y:, :, :] = 0    # paint the bottom black

                type_mask = np.zeros((height, width), dtype=np.uint8)
                parts_mask = np.zeros((height, width), dtype=np.uint8)
                parts_bboxes = []

                # number of foreground tools
                p = random.random()
                if p < 0.4:
                    tool_pos = ['l', 'r']
                elif p < 0.8:
                    tool_pos = ['r', 'l']
                elif p < 0.9:
                    tool_pos = ['l', 'r', 'b']
                else:
                    tool_pos = ['r', 'l', 'b']
                    
                # select tool
                for tp in tool_pos:
                    id = random.randint(0, num_fg_image_files_p[tp]-1)
#                    print(num_fg_image_files_p[tp], id)
                    fg_image_file = fg_image_files_p[tp][id]
                    fg_image = cv2.imread(fg_image_file)
                    
                    type_id = int(os.path.basename(fg_image_file).split('_')[3][1])

                    fg_type_file = fg_image_file.replace('images', 'type_masks')
                    fg_type_mask = cv2.imread(fg_type_file)
                    fg_type_mask = fg_type_mask[:, :, 0]

                    fg_parts_file = fg_image_file.replace('images', 'parts_masks')
                    fg_parts_mask = cv2.imread(fg_parts_file)
                    fg_parts_mask = fg_parts_mask[:, :, 0]
                    
                    ratio = random.uniform(0.5, 1.5)
                    fg_image = cv2.resize(fg_image, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
                    fg_type_mask = cv2.resize(fg_type_mask, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
                    fg_parts_mask = cv2.resize(fg_parts_mask, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

                    fg_height, fg_width, _ = fg_image.shape
                    if fg_width >= width or fg_height >= black_y:
                        continue

                    if tp == 'l':
                        # horizontal position
                        if fg_width >= width / 2:
                            sfx = random.randint(0, fg_width // 4)
                        else:
                            sfx = 0
                        # vertical position
                        fy = random.randint(0, black_y - fg_height)
                        
                        # merge images/masks
                        roi_image = bg_sel_image[fy:fy+fg_height, :fg_width-sfx]
                        roi_image = merge_image(fg_image[:, sfx:, :], roi_image,
                                                fg_type_mask[:, sfx:], type_id)
                        bg_sel_image[fy:fy+fg_height, :fg_width-sfx] = roi_image
                        
                        roi_type_mask = type_mask[fy:fy+fg_height, :fg_width-sfx]
                        roi_parts_mask = parts_mask[fy:fy+fg_height, :fg_width-sfx]
                        
                        roi_type_mask, roi_parts_mask, fg_type_mask, fg_parts_mask = merge_mask(
                            roi_type_mask, fg_type_mask[:, sfx:],
                            roi_parts_mask, fg_parts_mask[:, sfx:], type_id)
                        
                        type_mask[fy:fy+fg_height, :fg_width-sfx] = roi_type_mask
                        parts_mask[fy:fy+fg_height, :fg_width-sfx] = roi_parts_mask
                        
                        # bounding boxes
                        parts_bbox = get_parts_bbox(fg_parts_mask, fg_type_mask)
                        for i, bb in enumerate(parts_bbox):
                            bb[2] += fy
                            parts_bbox[i] = bb

                    elif tp == 'r':
                        # horizontal position
                        if fg_width >= width / 2:
                            sfx = random.randint(0, fg_width // 4)
                        else:
                            sfx = 0
                        # vertical position
                        fy = random.randint(0, black_y - fg_height)

                        # merge images/masks
        #                bg_sel_image[fy:fy+fg_height, -fg_width:] = fg_image
                        roi_image = bg_sel_image[fy:fy+fg_height, -(fg_width-sfx):]
                        roi_image = merge_image(fg_image[:, :fg_width-sfx, :], roi_image,
                                                fg_type_mask[:, :fg_width-sfx], type_id)
                        bg_sel_image[fy:fy+fg_height, -(fg_width-sfx):] = roi_image
                        
                        roi_type_mask = type_mask[fy:fy+fg_height, -(fg_width-sfx):]
                        roi_parts_mask = parts_mask[fy:fy+fg_height, -(fg_width-sfx):]
                        
                        roi_type_mask, roi_parts_mask, fg_type_mask, fg_parts_mask = merge_mask(
                            roi_type_mask, fg_type_mask[:, :fg_width-sfx], 
                            roi_parts_mask, fg_parts_mask[:, :fg_width-sfx], type_id)
                        
                        type_mask[fy:fy+fg_height, -(fg_width-sfx):] = roi_type_mask
                        parts_mask[fy:fy+fg_height, -(fg_width-sfx):] = roi_parts_mask

                        # bounding boxes
                        parts_bbox = get_parts_bbox(fg_parts_mask, fg_type_mask)
                        for i, bb in enumerate(parts_bbox):
                            bb[1] += (width - fg_width + sfx)
                            bb[2] += fy
                            parts_bbox[i] = bb
                        
                    elif tp == 'b':
                        fx = random.randint(0, width - fg_width)
                        roi_image = bg_sel_image[-fg_height:, fx:fx+fg_width]
                        roi_image = merge_image(fg_image, roi_image, fg_type_mask, type_id)
                        bg_sel_image[-fg_height:, fx:fx+fg_width] = roi_image
                        
                        roi_type_mask = type_mask[-fg_height:, fx:fx+fg_width]
                        roi_parts_mask = parts_mask[-fg_height:, fx:fx+fg_width]
                        roi_type_mask, roi_parts_mask, fg_type_mask, fg_parts_mask = merge_mask(
                            roi_type_mask, fg_type_mask, roi_parts_mask, fg_parts_mask, type_id)
                        type_mask[-fg_height:, fx:fx+fg_width] = roi_type_mask
                        parts_mask[-fg_height:, fx:fx+fg_width] = roi_parts_mask
                        
                        parts_bbox = get_parts_bbox(fg_parts_mask, fg_type_mask)
                        for i, bb in enumerate(parts_bbox):
                            bb[1] += fx
                            bb[2] += (height - fg_height)
                            parts_bbox[i] = bb

                    parts_bboxes.extend(parts_bbox)

                    dst_filename += f"__{os.path.basename(fg_image_file).split('.')[0]}"
                
                """
                # imshow
                color = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
                for bb in parts_bboxes:
                    parts_label = bb[0] % 3
                    type_label = bb[0] // 3
                    cv2.rectangle(bg_sel_image, (bb[1], bb[2]), 
                                  (bb[1]+bb[3]-1, bb[2]+bb[4]-1), color[parts_label], 2)
                    cv2.putText(bg_sel_image, f"{type_label}",
                                (int(bb[1]), int(bb[2]) - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
                    
                cv2.imshow('synth', bg_sel_image)
                cv2.imshow('type', type_mask*35)
                cv2.imshow('parts', parts_mask*80)
                cv2.waitKey(0)
                """
                
                # change to yolo format            
                for i, bb in enumerate(parts_bboxes):
                    bb[1] = (bb[1] + bb[3] / 2) / width
                    bb[2] = (bb[2] + bb[4] / 2) / height
                    bb[3] = bb[3] / width
                    bb[4] = bb[4] / height
                    parts_bboxes[i] = bb

                dst_filename = dst_filename.replace('video', '').replace('frame', '').replace('train', '')
                print("dst:", dst_filename)

                # write
                output_image_file = f'{dst_topdir}/images/{dst_filename}.png'
                cv2.imwrite(output_image_file, bg_sel_image)
                """
                output_type_file = f'{dst_topdir}/type_masks/{dst_filename}.png'
                cv2.imwrite(output_type_file, type_mask)
                output_parts_file = f'{dst_topdir}/parts_masks/{dst_filename}.png'
                cv2.imwrite(output_parts_file, parts_mask)
                """
                output_bbox_file = f'{dst_topdir}/labels/{dst_filename}.txt'
                with open(output_bbox_file, 'wt') as f:
                    for bb in parts_bboxes:
                        f.write(f'{bb[0]:d} {bb[1]:.8f} {bb[2]:.8f} {bb[3]:.8f} {bb[4]:.8f}\n')
                     
                cnt_images += 1
