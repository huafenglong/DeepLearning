import time
import win32gui
import win32process
import ctypes
import win32api
import win32con

class Window():
    def __init__(self, string=None):
        self.window_text = string
        self.window_handle = win32gui.FindWindow(None, self.window_text)

    def switch_hwnds(self):
        """
        切换到当前句柄，需先用按键触发
        :return:
        """
        win32gui.SetForegroundWindow(self.window_handle)

    # def get_all_hwnds_from_text(self):
    #     """
    #     读取所有标题是string的窗口句柄
    #     """
    #     string = self.handle_text
    #     def get_all_hwnd(hwnd, hwnd_title):
    #         if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
    #             if win32gui.GetWindowText(hwnd) == string:
    #                 self.hwnd_title.append(hwnd)
    #
    #     win32gui.EnumWindows(get_all_hwnd, self.hwnd_title)


class WrapRaiden3Env():
    def __init__(self):
        self.action_list = [38, 40, 37, 39, 90, 88]  #13

    def step(self, action):
        win32api.keybd_event(self.action_list[action], 0, 0, 0)

    def observation(self, observation):
        pass

    def reset(self):
        pass




class GrabMemoryReward():
    def __init__(self):
        pass

    def grab_reward_from_memory(self, window_handle=None, memory_address=0x0053B644):
        """
        根据窗口的句柄，和进程的内存地址，读出地址的值
        :param window_handle: 窗口的句柄
        :param memory_address: 内存地址
        :return: 地址的值
        """
        # 获取窗口进程（线程[0]和进程[1]）
        process_id = win32process.GetWindowThreadProcessId(window_handle)[1]
        #print(f"窗口句柄是{window_handle}, 窗口的进程pid是{process_id}")
        # 获取进程的句柄,在0x1F0FFF处循环遍历，False句柄不被子进程继承
        process_handle = win32api.OpenProcess(0x1F0FFF, False, process_id)
        # 读出内核
        kernel32 = ctypes.windll.LoadLibrary(r"C:\Windows\System32\kernel32.dll")
        # 转换数据类型为C long,ctypes.byref作为缓存区处理不兼容问题，4个字节长度，（None数据尺寸？）
        memory_data = ctypes.c_long()
        kernel32.ReadProcessMemory(int(process_handle), memory_address, ctypes.byref(memory_data), 4, None)
        print(f"当前的得分是{memory_data.value}")

        return memory_data.value

    def write_reward_to_memory(self, window_handle = None, value = 0, memory_address=0x0053B644):
        # 获取窗口进程（线程【0】和进程【1】）
        process_id = win32process.GetWindowThreadProcessId(window_handle)[1]
        print(f"窗口句柄是{window_handle}, 窗口的进程pid是{process_id}")
        # 获取进程的句柄,在0x1F0FFF处循环遍历，句柄不被子进程继承
        process_handle = win32api.OpenProcess(0x1F0FFF, False, process_id)
        # 读出内核
        kernel32 = ctypes.windll.LoadLibrary(r"C:\Windows\System32\kernel32.dll")
        # 转换数据类型为C long,ctypes.byref作为缓存区处理不兼容问题，4个字节长度，（None数据尺寸？）
        kernel32.WriteProcessMemory(int(process_handle), memory_address,
                                    ctypes.byref(ctypes.c_long(int(value))), 4, None)


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

if __name__ == '__main__':
    all_memory_address = {"mutil":0x0061BD9C, "signal":0x0053B644}
    game_window_text = "[#] 棆揹嘨 (C)2005-2006 MOSS LTD ALL RIGHTS RESERVED. [#]"
    win32_raiden3 = Window(game_window_text)
    win32_raiden3.get_all_hwnds_from_text()

    process_id = win32process.GetWindowThreadProcessId(win32_raiden3.hwnd_title[0])[1]
    process_handle = win32api.OpenProcess(0x1F0FFF, False, process_id)
    win32process.SetProcessAffinityMask(process_handle, 0x0001)

    process_id = win32process.GetWindowThreadProcessId(win32_raiden3.hwnd_title[1])[1]
    process_handle = win32api.OpenProcess(0x1F0FFF, False, process_id)
    win32process.SetProcessAffinityMask(process_handle, 0x0002)

    proc1 = mp.Process(target=proces1, args=(win32_raiden3.hwnd_title[0],))
    proc2 = mp.Process(target=proces1, args=(win32_raiden3.hwnd_title[1],))

    proc1.start()
    proc2.start()

    process_handle1 = win32api.OpenProcess(0x1F0FFF, False, proc1.pid)
    win32process.SetProcessAffinityMask(process_handle1, 0x0004)
    process_handle2 = win32api.OpenProcess(0x1F0FFF, False, proc2.pid)
    win32process.SetProcessAffinityMask(process_handle2, 0x000f)

    proc1.join()



