from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
import shutil
import random

def crop_center(mask_dir, im_dir, mask_save, img_save):

    imgs = os.listdir(im_dir)
    for i, img in enumerate(imgs):
        mask = Image.open(mask_dir + "/" + img)
        image = Image.open(im_dir + "/" + img)
        img_arr = np.asarray(mask)
        t_x, t_y = np.where(img_arr > 0)
        # (left, top, right, bottom)
        l = t_x[0] - 64
        t = t_y[0] - 64
        r = t_x[0] + 64
        b = t_y[0] + 64
        box = (l, t, r, b)
        region_img = image.crop(box)
        region_img.save(img_save + "/" + img)
        region_mask = mask.crop(box)
        region_mask.save(mask_save + "/" + img)

def crop_lt(mask_dir, im_dir, mask_save, img_save):
    imgs = os.listdir(im_dir)
    for i, img in enumerate(imgs):
        mask = Image.open(mask_dir + "/" + img)
        image = Image.open(im_dir + "/" + img)
        img_arr = np.asarray(mask)
        t_x, t_y = np.where(img_arr > 0)
        # (left, top, right, bottom)
        l = t_x[0] - 32
        t = t_y[0] - 32
        r = t_x[0] + 96
        b = t_y[0] + 96
        box = (l, t, r, b)
        region_img = image.crop(box)
        region_img.save(img_save + "/" + img)
        region_mask = mask.crop(box)
        region_mask.save(mask_save + "/" + img)

def crop_rt(mask_dir, im_dir, mask_save, img_save):
    imgs = os.listdir(im_dir)
    for i, img in enumerate(imgs):
        mask = Image.open(mask_dir + "/" + img)
        image = Image.open(im_dir + "/" + img)
        img_arr = np.asarray(mask)
        t_x, t_y = np.where(img_arr > 0)
        # (left, top, right, bottom)
        l = t_x[0] - 96
        t = t_y[0] - 32
        r = t_x[0] + 32
        b = t_y[0] + 96
        box = (l, t, r, b)
        region_img = image.crop(box)
        region_img.save(img_save + "/" + img)
        region_mask = mask.crop(box)
        region_mask.save(mask_save + "/" + img)

def crop_lb(mask_dir, im_dir, mask_save, img_save):
    imgs = os.listdir(im_dir)
    for i, img in enumerate(imgs):
        mask = Image.open(mask_dir + "/" + img)
        image = Image.open(im_dir + "/" + img)
        img_arr = np.asarray(mask)
        t_x, t_y = np.where(img_arr > 0)
        # (left, top, right, bottom)
        l = t_x[0] - 32
        t = t_y[0] - 96
        r = t_x[0] + 96
        b = t_y[0] + 32
        box = (l, t, r, b)
        region_img = image.crop(box)
        region_img.save(img_save + "/" + img)
        region_mask = mask.crop(box)
        region_mask.save(mask_save + "/" + img)

def crop_rb(mask_dir, im_dir, mask_save, img_save):
    imgs = os.listdir(im_dir)
    for i, img in enumerate(imgs):
        mask = Image.open(mask_dir + "/" + img)
        image = Image.open(im_dir + "/" + img)
        img_arr = np.asarray(mask)
        t_x, t_y = np.where(img_arr > 0)
        # (left, top, right, bottom)
        l = t_x[0] - 96
        t = t_y[0] - 96
        r = t_x[0] + 32
        b = t_y[0] + 32
        box = (l, t, r, b)
        region_img = image.crop(box)
        region_img.save(img_save + "/" + img)
        region_mask = mask.crop(box)
        region_mask.save(mask_save + "/" + img)

def crop_sq(mask_dir, im_dir, mask_save, img_save):
    imgs = os.listdir(mask_dir)
    for i, img in enumerate(imgs):
        mask = Image.open(mask_dir + "/" + img)
        image = Image.open(im_dir + "/" + img)
        img_arr = np.asarray(mask)
        t_y, t_x = np.where(img_arr > 0)
        w, h = image.size
        if w > h:
            cut = h // 2
        else:
            cut = w // 2

        l = t_x[0] - cut
        r = t_x[0] + cut
        t = t_y[0] - cut
        b = t_y[0] + cut
        if l < 3 :
            l = 0
            r = cut * 2
        if r > w :
            l = w - 2 * cut
            r = w
        if t < 3:
            t = 0
            b =  2 * cut
        if b > h:
            t = h - 2 * cut
            b = h

        box = (l, t, r, b)
        region_img = image.crop(box)
        region_img = region_img.resize((128, 128), Image.BICUBIC)
        region_img.save(img_save + "/" + img)
        region_mask = mask.crop(box)
        region_mask = region_mask.resize((128, 128), Image.BICUBIC).convert('1')
        region_mask.save(mask_save + "/" + img)

def crop_128(mask_dir, im_dir, mask_save, img_save):
    imgs = os.listdir(mask_dir)
    for i, img in enumerate(imgs):
        mask = Image.open(mask_dir + "/" + img)
        image = Image.open(im_dir + "/" + img)
        img_arr = np.asarray(mask)
        t_y, t_x = np.where(img_arr > 0)
        w, h = image.size
        if w<128 or h<128:
            image_128 = image.resize((128, 128), resample=Image.BICUBIC)
            image_128.save(img_save + "/" + img)
            mask_128 = mask.resize((128, 128), resample=Image.BICUBIC)
            mask_128 = mask_128.convert('1')
            mask_128.save(mask_save + "/" + img)


        else:
            cut = 64
            l = t_x[0] - cut
            r = t_x[0] + cut
            t = t_y[0] - cut
            b = t_y[0] + cut
            if l < 3:
                l = 0
                r = cut * 2
            if r > w:
                l = w - 2 * cut
                r = w
            if t < 3:
                t = 0
                b = 2 * cut
            if b > h:
                t = h - 2 * cut
                b = h

            box = (l, t, r, b)
            region_img = image.crop(box)
            # region_img = region_img.resize((128, 128), Image.BICUBIC)
            region_img.save(img_save + "/" + img)
            region_mask = mask.crop(box)
            region_mask = region_mask.convert('1')
            region_mask.save(mask_save + "/" + img)



def tar_size(in_dir, out_dir):
    imgs = os.listdir(in_dir)

    for i, img in enumerate(imgs):
        image = Image.open(in_dir + "/" + img).convert('L')
        img_arr = np.asarray(image)
        shape = img_arr.shape
        tar = 0
        for h in range(0, shape[0]):
            for w in range(0, shape[1]):
                value = img_arr[h, w]
                # print("",value)
                if value != 0:
                    tar += 1
        if tar > 81:
            print(img , tar)
            shutil.move(in_dir + "/" + img,
                        out_dir)
            shutil.move('G:/mynet/data/NUDT-SIRST/128/images/' + img,
                        'G:/mynet/data/NUDT-SIRST/128/images_81/')



def split_test(data_path, label_path):
    # data_path = "文件夹地址"
    # sub_path = ['各存放子文件的文件夹名称', '...']
    fileNames = os.listdir(data_path)  # 获取当前路径下的文件名，返回List
    num_plt = len(fileNames)

    test_path = 'G:/mynet/data/sirst_128/train/test/'
    # train_path =  'G:/mynet/data/sirst_128/train/'

    # num_test = num_plt // 10 * 3  # 计算测试集的总数
    num_test = 84  # 计算测试集的总数
    index_test = random.sample(range(num_plt), num_test)  # 从num_plt个文件中随机选出num_test个作为索引,random.sample用于生成不重复的随机数
    index_test.sort(reverse=True)  # 对索引进行排序，使其从后往前删除
    # print(index_test)
    for i in index_test:
        plt_test = fileNames.pop(i)
        shutil.move(data_path + '/' + plt_test, test_path + 'images')
        shutil.move(label_path + '/' + plt_test, test_path + 'masks')

    # for i in fileNames:
    #     shutil.copy(data_path + '/' + i, train_path + 'images')
    #     shutil.copy(label_path + '/' + i, train_path + 'masks')






if __name__ == "__main__":
    im_dir = 'G:/mynet/data/mdfa_NUST/test/'
    save_dir ='G:/mynet/data/mdfa_NUST/test_128/'
    im_gt_dir = 'G:/mynet/data/mdfa_NUST/test_gt/'
    save_gt_dir = 'G:/mynet/data/mdfa_NUST/test_gt_128/'

    out_128 =  'G:/mynet/data/IRSTD-1k/train_128/gt_81/'
    org_128 = 'G:/mynet/data/IRSTD-1k/train_128/org_81/'


    # test dir
    # im_gt_dir ='C:/Users/admin/Desktop/Auto DL Code/Dice_0.82/code/test_data/mask'
    # im_dir = 'C:/Users/admin/Desktop/Auto DL Code/Dice_0.82/code/test_data/'
    # save_gt_dir ='C:/Users/admin/Desktop/Auto DL Code/Dice_0.82/code/test_data/save/mask/'
    # save_dir = 'C:/Users/admin/Desktop/Auto DL Code/Dice_0.82/code/test_data/save/'


    # crop_128(im_gt_dir, im_dir, save_gt_dir, save_dir)

    # tar_size(im_gt_dir, save_gt_dir)
    data_path = 'G:/mynet/data/sirst_128/train/images'
    label_path = 'G:/mynet/data/sirst_128/train/masks'

    split_test(data_path, label_path)

