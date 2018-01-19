不多说了，直接代码吧：

生成AFLW_ann.txt的代码，其中包含图像名称 和 图像中人脸的位置（x,y,w,h）;

** AFLW中含有aflw.aqlite文件。

```

importsqlite3

list_annotation = list()

# Format for saving: path x y w h

ann_format ="{}/{} {} {} {} {}"

conn = sqlite3.connect('aflw.sqlite')

fidQuery ='SELECT face_id FROM Faces'

faceIDs = conn.execute(fidQuery)

foridxinfaceIDs:

fidQuery ='SELECT file_id FROM Faces WHERE face_id = {}'.format(idx[0])

imgID = conn.execute(fidQuery)

imgID = [idforidinimgID]

imgDataQuery ="SELECT db_id,filepath,width,height FROM FaceImages WHERE file_id = '{}'".format(imgID[0][0])

fileID = conn.execute(imgDataQuery)

fileID = [idforidinfileID]

db_id = fileID[0][0]

filepath = fileID[0][1]

faceRectQuery ='SELECT x,y,w,h FROM FaceRect WHERE face_id = {}'.format(idx[0])

faceRect = conn.execute(faceRectQuery)

faceRect = [idforidinfaceRect]

iflen(faceRect)==0:

continue

x,y,w,h =  faceRect[0]

list_annotation.append(ann_format.format(db_id,filepath,x,y,w,h))

with open("AFLW_ann.txt",'w') as f:

f.writelines("%s\n"% lineforlineinlist_annotation)

```

AFLW图片都整理到flickr文件下（含0,1,2三个文件），生成人脸的程序（并且对人脸进行了左右镜像）：

```

importos

fromPILimportImage

fromPILimportImageFile

# ImageFile.LOAD_TRUNCATED_IMAGES = True

importcv2

importnumpy as np

with open('AFLW_ann.txt','r') as f:

lines = f.readlines()

save_dir1 ='data_prepare/net_positive'

save_dir2 ='data_prepare/net_positive_flip'

ifos.path.exists(save_dir1)==False:

os.makedirs(save_dir1)

ifos.path.exists(save_dir2)==False:

os.makedirs(save_dir2)

foridx, lineinenumerate(lines):

s1 = line.strip().split(' ')

image_path = s1[0]

x = int(s1[1])

y = int(s1[2])

w = int(s1[3])

h = int(s1[4])

print(image_path)

# image = Image.open(image_path)

image = cv2.imread(image_path)

ifimageisNone:

continue

ifx<=0andy<=0andw<=0andh<=0:

continue

box = (x, y, x+w, y+h)

# patch = image.crop(box)

patch = image[box[1]:box[3], box[0]:box[2], :]

ifpatchisNone:

continue

patch1 = patch#.resize((51, 51))

# patch2 = patch1.transpose(Image.FLIP_LEFT_RIGHT)

h = patch.shape[0]

w = patch.shape[1]

iLR = patch.copy()

foriinrange(h):

forjinrange(w):

iLR[i,w-1-j] = patch[i,j]

patch2 = iLR

s2 = image_path.split('/')

image_name = s2[-1]

save_path1 = save_dir1+'/'+str(idx)+image_name +'.jpg'

save_path2 = save_dir2+'/'+'l'+str(idx)+image_name +'.jpg'

#patch1.save(save_path1, 'jpeg')

#patch2.save(save_path2, 'jpeg')

cv2.imwrite(save_path1, np.array(patch1))

cv2.imwrite(save_path2, np.array(patch2))

print(idx)

```
