import time
import cv2
import numpy as np
import win32con
import win32gui
import win32ui
import win32api
import win32process
import os

def show_all_hwnd():
    """
    显示所有桌面级窗口的句柄
    """
    hwnd_title = dict()

    def get_all_hwnd(hwnd, mouse):
        if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
            hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})

    win32gui.EnumWindows(get_all_hwnd, 0)

    for h, t in hwnd_title.items():
        if t is not "":
            print(h, t)

class GameEnv(object):
    def __init__(self):
        pass

    def whether_has_continued(self, img=None, threshold=0.8):
        """
        判断是否是结束画面
        :param img: 当前的游戏画面
        :param threshold: 判断阈值
        :return: 结果True or Flase
        """
        img_continue = cv2.imread("continue.jpg", cv2.IMREAD_GRAYSCALE)

        res = cv2.matchTemplate(img, img_continue, cv2.TM_CCOEFF_NORMED)

        w, h = img_continue.shape[::-1]
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
        cv2.imshow('imshow', img)

        if (res >= threshold).any():
            return True
        else:
            return False

    def grab_screen(self, handle=None, width=None, height=None):
        """
        对指定窗口截屏
        :param handle: 需要截屏的窗口句柄
        :param width: 窗口的宽度
        :param height: 窗口的高度
        :return: 截取屏幕的灰度图像
        """
        # 获取窗口DC
        hdDC = win32gui.GetDC(handle)
        # 根据句柄创建一个DC
        newhdDC = win32ui.CreateDCFromHandle(hdDC)
        # 创建一个兼容设备内存的DC
        saveDC = newhdDC.CreateCompatibleDC()
        # 创建bitmap保存图片
        saveBitmap = win32ui.CreateBitmap()

        if width is None and height is None:
            # 获取窗口的位置信息
            left, top, right, bottom = win32gui.GetWindowRect(handle)
            # 窗口长宽
            width = right - left
            height = bottom - top
        # bitmap初始化
        saveBitmap.CreateCompatibleBitmap(newhdDC, width, height)
        saveDC.SelectObject(saveBitmap)
        saveDC.BitBlt((0, 0), (width, height), newhdDC, (0, 0), win32con.SRCCOPY)

        signedIntsArray = saveBitmap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        newhdDC.DeleteDC()
        saveDC.DeleteDC()
        win32gui.ReleaseDC(handle, hdDC)
        win32gui.DeleteObject(saveBitmap.GetHandle())

        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    def cal_total_reward(self, img=None, threshold=0.5):
        """
        判断当前画面的分数
        :param img: 当前的游戏画面
        :param threshold: 判断阈值
        :return: 结果True or Flase
        """
        num_postion = [(59 - i * 6, 54 - i * 6) for i in range(8)]
        reward_current = 0
        postion_num = [0,0,0,0,0,0,0,0]
        for i in reversed(range(8)):
            "读取0到8个区域的图片"
            img_postion = img[8:15, num_postion[i][1]:num_postion[i][0]]
            reslist = []
            for j in range(9):
                "读取0到9的数字"
                img_nums = cv2.imread(("data/"+str(j)+".jpg"), cv2.IMREAD_GRAYSCALE)
                res = cv2.matchTemplate(img_postion, img_nums, cv2.TM_CCOEFF_NORMED)
                reslist.append(res)

            max_res = max(reslist)
            if max_res > threshold:
                postion_num[i] = reslist.index(max_res)
                reward_current = reward_current * 10 + reslist.index(max_res)
            else:
                reward_current = reward_current * 10
                postion_num[i] = 0

            del reslist

        print(postion_num)
        return reward_current




if __name__ == '__main__':
    game_env = GameEnv()

    # 获取窗口句柄
    handle = win32gui.FindWindow(None, 'Raiden III @ 2005-2014 MOSS LTD & H2 INTERACTIVE CO LTD ALL RIGHTS RESERVED')
    while True:
        show_all_hwnd()
        start_time = time.time()
        img = game_env.grab_screen(handle, width=None, height=None)
        #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        #game_is_over = game_env.whether_has_continued(img)
        #reward_current = game_env.cal_total_reward(img)
        #cv2.putText(img, str(reward_current), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 5)
        img = img[15:30, 0:150]
        #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #thresh = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        # for i in range(0, len(contours)):
        #     x, y, w, h = cv2.boundingRect(contours[i])
        #     cv2.rectangle(thresh, (x, y), (x + w, y + h), (153, 153, 0), 5)
        #
        #     newimage = thresh[y :y + h , x:x + w]  # 先用y确定高，再用x确定宽
        #     string = "data/number" + str(i) + ".jpg"
        #     cv2.imwrite(string, newimage)
        #     print(i)
        end_time = time.time()
        #print(f"{end_time - start_time: 0.4f}")
        cv2.imshow('img_gameover1', img)
        if cv2.waitKey(25):
            pass
            #cv2.destroyAllWindows()

#img_number1 = img[8:15, 54:59]
#img_number2 = img[8:15, 48:53]
#img_number3 = img[8:15, 42:47]
#img_number3 = img[8:15, 36:41]
#img_number5 = img[8:15, 30:35]
# def cut_continue_img():
#     img_gameover = img_gameover[78:92, 25:95]       #high and width
#     cv2.imwrite("gameover.jpg",img_gameover)
#     cv2.rectangle(img_gameover, (25,78), (95,92), 255, 1)
#     cv2.imshow('img_gameover1', img_gameover)
#     cv2.waitKey(0)

# def write_number_to_jpg():
#     num_postion = [(59 - i * 6, 54 - i * 6) for i in range(8)]
#
#     for i in range(8):
#         img_number = img[8:15, num_postion[i][1]:num_postion[i][0]]
#         string = "data/number" + str(i) + ".jpg"
#         cv2.imwrite(string, img_number)
 #cv2.rectangle(img, (num_postion[i][1],8), (num_postion[i][0],15), 255, 1)