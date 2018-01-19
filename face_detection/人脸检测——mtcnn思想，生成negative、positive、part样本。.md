negative样本：IOU < 0.3，标签为：0 0 0 0 0
positive样本：IOU > =0.65，标签为：1 0.01 0.02 0.01 0.02
part样本：0.4 <= IOU < 0.65，标签为： -1 0.03 0.04 0.03 0.04
注意mtcnn的label加了回归框，训练时候的输出层要作修改：（回归框的作用还是很大的）
compute bbox reg label，其中x1,x2,y1,y2为真实的人脸坐标，x_left,x_right,y_top,y_bottom，width,height为预测的人脸坐标，
如果是在准备人脸和非人脸样本的时候，x_left,x_right,y_top,y_bottom，width,height就是你的滑动窗与真实人脸的IOU>0.65（根据你的定义）的滑动窗坐标。
offset_x1 = (x1 - x_left) / float(width)
offset_y1 = (y1 - y_top) / float(height)
offset_x2 = (x2 - x_right) / float(width)
offset_y2 = (y2 - y_bottom ) / float(height)

很多人可能会有一个疑问：就是训练的时候人脸样本时候回归框label的，但非人脸呢，这地方可以全给0。
代码：
```
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU

anno_file = "./wider_annotations/anno.txt"
im_dir = "/home/seanlx/Dataset/wider_face/WIDER_train/images"
neg_save_dir = "/data3/seanlx/mtcnn1/12/negative"
pos_save_dir = "/data3/seanlx/mtcnn1/12/positive"
part_save_dir = "/data3/seanlx/mtcnn1/12/part"

save_dir = "./pnet"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')

with open(anno_file, 'r') as f:
    annotations = f.readlines()

num = len(annotations)
print "%d pics in total" % num
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    bbox = map(float, annotation[1:])
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    idx += 1
    if idx % 100 == 0:
        print idx, "images done"

    height, width, channel = img.shape

    neg_num = 0
    while neg_num < 50:
        size = npr.randint(12, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny : ny + size, nx : nx + size, :]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write("12/negative/%s"%n_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1


    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        # generate negative examples that have overlap with gt
        for i in range(5):
            size = npr.randint(12,  min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny1 : ny1 + size, nx1 : nx1 + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                f2.write("12/negative/%s"%n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1

        # generate positive examples and part faces
        for i in range(20):
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            cropped_im = img[ny1 : ny2, nx1 : nx2, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write("12/positive/%s"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write("12/part/%s"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
        print "%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx)

f1.close()
f2.close()
f3.close()
```
