import os
import cv2 as cv
import re
import math
import random
from tqdm import tqdm

def get_file_path(folder_path) -> list:
    tmp_list = os.listdir(folder_path)
    for itr in range(len(tmp_list)):
        tmp_list[itr] = folder_path + '/' + tmp_list[itr]
    return tmp_list

def RandomErasing(path, augmentation_probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
    for img_path in tqdm(path):
        img = cv.imread(img_path)
        if random.uniform(0, 1) > augmentation_probability:
            pass
        else:
            for attempt in range(100):
                try:
                    area = img.shape[0] * img.shape[1]

                    target_area = random.uniform(sl, sh) * area
                    aspect_ratio = random.uniform(r1, 1/r1)

                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))

                    if h < img.shape[0] and w < img.shape[1]:

                        new_file_name = re.findall('(.*)\.jpg', img_path)
                        new_file_name = new_file_name[0] + '_2.jpg'


                        x1 = random.randint(0, img.shape[0] - h)
                        y1 = random.randint(0, img.shape[1] - w)
                        if img.shape[2] == 3:
                            # img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                            # img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                            # img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                            # print(img[x1:x1+h, y1:y1+w, 0])
                            img[x1:x1+h, y1:y1+w, 0] = mean[0]
                            img[x1:x1+h, y1:y1+w, 1] = mean[1]
                            img[x1:x1+h, y1:y1+w, 2] = mean[2]
                            # img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                        else:
                            img[x1:x1+h, y1:y1+w, 0] = mean[1]
                            # img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))


                        # store the erased img to the place you want
                        cv.imwrite(new_file_name, img)
                        break


                except:
                    continue

    return img


if __name__=='__main__':
    # These two folder stored the data that you want to do random erasing
    whole_data_path = get_file_path('./PetImages_augmentation/Dog')
    RandomErasing(whole_data_path)
