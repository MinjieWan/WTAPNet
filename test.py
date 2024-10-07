from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
import shutil
import glob

def save_imgs(imgs_dir, out_dir, out_gt_dir):
    imgs = os.listdir(imgs_dir)
    for i, img in enumerate(imgs):
        img_src = Image.open(imgs_dir + "/" + img)
        suffix = img.split("_")[1].split(".")[0]
        # print(suffix)
        if suffix == '1':
            # print("save:"+ out_dir + "/" + img)
            img_src.save(out_dir + "/" + img)
        if suffix == '2':
            img_src.save(out_gt_dir + "/" + img)

def del_suffix(im_dir, save_dir):
    imgs = os.listdir(im_dir)
    for i, img in enumerate(imgs):
        img_src = Image.open(im_dir + "/" + img)
        newname = img.split("_")[0]
        img_src.save(save_dir + "/" + newname + '.png')


def file_name(ori_dir,gt_dir):
    ori_list = []
    gt_list = []
    for root, dirs, files in os.walk(ori_dir):
        for file in files:
            ori_list.append(os.path.splitext(file)[0])
    for root, dirs, files in os.walk(gt_dir):
        for file in files:
            gt_list.append(os.path.splitext(file)[0].split("_")[0])
    print(len(ori_list))
    diff = set(gt_list).difference(set(ori_list))  # 差集，在a中但不在b中的元素
    for name in diff:
        print("no ori", name + ".png")
    diff2 = set(ori_list).difference(set(gt_list))  # 差集，在b中但不在a中的元素
    print(len(diff2))
    for name in diff2:
        print("no gt", name + ".png")


def convert_square_paste(im_dir, save_dir):
    """
    convert to square with paste
    :param image:Images that need to be processed
    :return:After processing the picture
    """
    imgs = os.listdir(im_dir)
    for i, img in enumerate(imgs):
        image = Image.open(im_dir + "/" + img)
        width,height = image.size
        new_image_length = max(width,height)#获取新图边长
        if new_image_length % 4 == 0:
            new_image = Image.new("RGB", (new_image_length, new_image_length), (127, 127, 127))  # 生成一张正方形底图
            pad_len = int(abs(width - height) / 2)  # 预填充区域的大小
            box = (0, pad_len) if width > height else (pad_len, 0)


        else:
            add = 4 - new_image_length % 4
            new_image_length = new_image_length + add
            new_image = Image.new("RGB", (new_image_length, new_image_length), (127, 127, 127))  # 生成一张正方形底图
            if width > height:
                pad_len = int(abs(width - height) / 2 + add / 2)  # 预填充区域的大小
                box = (int(add / 2) , pad_len)
            else:
                pad_len = int(abs(width - height) / 2 + add / 2)  # 预填充区域的大小
                box = (pad_len, int(add / 2))

        new_image.paste(image,box)
        new_image.save(save_dir + "/" + img)

def resizer(im_dir, save_dir, new_size):
    size = new_size
    imgs = os.listdir(im_dir)
    for i, img in enumerate(imgs):
        image = Image.open(im_dir + "/" + img)
        new_image = image.resize((size,size))
        new_image.save(save_dir + "/" + img)


def crop_1024(im_dir, save_dir):

    imgs = os.listdir(im_dir)
    for i, img in enumerate(imgs):
        image = Image.open(im_dir + "/" + img)
        # 前两个坐标点是左上角坐标
        # 后两个坐标点是右下角坐标
        # width在前， height在后
        box = (0, 0, 1024, 1024)
        region = image.crop(box)
        region.save(save_dir + "/" + img)

def save_L(imgs_dir, out_dir):
    imgs = os.listdir(imgs_dir)
    for i, img in enumerate(imgs):
        img_src = Image.open(imgs_dir + "/" + img)
        img2 = img_src.convert("L")
        img2.save(out_dir + "/" + img)

def save_2(imgs_dir, out_dir):
    imgs = os.listdir(imgs_dir)
    for i, img in enumerate(imgs):
        img_src = Image.open(imgs_dir + "/" + img)
        img1 = np.array(img_src)
        img2 = np.where(img1[..., :] >0 , 255 , 0)
        # img3 = 255 - img2
        img_save = Image.fromarray(img2)
        img_save.convert('L').save(out_dir + "/" + img)






if __name__ == "__main__":



    # file_object = open('G:/mynet/data/IRSTD-1k/test_128/81.txt')
    # try:
    #     for line in file_object:
    #         # print(line)
    #         shutil.move('../data/IRSTD-1k/test_128/org/' + line.rstrip('\n') ,
    #                     "../data/IRSTD-1k/test_128/org_81/")
    #         # shutil.move('G:/mynet/data/IRSTD-1k/IRSTD1k_Label/' + line.rstrip('\n') + '.png',
    #         #             "G:/mynet/data/IRSTD-1k/test/gt/")
    # finally:
    #     file_object.close()


  #   im_dir = 'E:/Infrared_Small_Target_Detection/F-Domain/MDvsFA_cGAN-master/data/training/'
  #   save_dir = 'E:/Infrared_Small_Target_Detection/F-Domain/MDvsFA_cGAN-master/data/train_org/'
  #   save_gt_dir = 'E:/Infrared_Small_Target_Detection/F-Domain/MDvsFA_cGAN-master/data/train_gt/'
  #   save_nosuffix = 'E:/Infrared_Small_Target_Detection/F-Domain/MDvsFA_cGAN-master/data/train_org_nosuffix/'
  #
  # #  save_imgs(im_dir, save_dir, save_gt_dir)
  #   train_org = './data/MDvsFA/train_org_nosuffix/'
  #   train_gt = './data/MDvsFA/train_gt/'
  #
  #   test_org = './data/MDvsFA/test_org/'
  #   test_org_sq = './data/MDvsFA/test_org_sq/'
  #
  #   test_gt = './data/MDvsFA/org_test_0/'
  #   test_gt_save = './data/MDvsFA/org_test/'
    # del_suffix(save_dir, save_nosuffix)

    # file_name(train_org, train_gt)

    # resizer(test_gt, test_gt_save, 512)

    # bomb = 'E:/Infrared_Small_Target_Detection/F-Domain/data/bomb/'
    # bomb_1024 = './data/bomb/'
    # crop_1024(bomb, bomb_1024)
  # in_dir = 'G:/mynet/data/NCHU-Seg/train/org'
  # out_dir = 'G:/mynet/data/NCHU-Seg/train/gt'

  save_L(in_dir, out_dir)
  # file_name(in_dir, out_dir)

# -*- coding:utf-8 -*-








