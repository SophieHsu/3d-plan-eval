import cv2
import numpy as np
import os
import threading
import time
import win32gui
import win32ui
import win32con


__current_dir_path = os.path.dirname(os.path.abspath(__file__))

__SCREEN_RECORDING_CONFIG = {
    'VR': {
        'window_title': 'VR',
        'output_path': os.path.join(__current_dir_path, 'recordings', 'vr_fov_aware_planner.avi'),
        'fps': 30
    },
    'PLANNER': {
        'window_title': 'PLANNER',
        'output_path': os.path.join(__current_dir_path, 'recordings', 'vr_fov_aware_planner.avi'),
        'fps': 30
    },
}

class WindowClosedException(Exception):
    pass

class Recorder:
    def __init__(self, window_title=None, output_path=None, fps=30, **kwargs):
        self.window_title = window_title
        self.window_handle = None
        self.output_path = output_path or f"recording_{int(time.time())}.avi"
        self.fps = fps
        self.is_recording = False
        self.video_writer = None
        self.record_thread = None

        if self.window_title:
            self.set_window_handle()

    def set_window_handle(self):
        self.window_handle = win32gui.FindWindow(None, self.window_title)
        if not self.window_handle:
            raise ValueError(f"Window with title '{self.window_title}' not found")

    def get_window_dimensions(self):
        if not self.window_handle:
            raise ValueError("Window handle not set")
        rect = win32gui.GetWindowRect(self.window_handle)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        return width, height

    def window_exists(self):
        try:
            win32gui.GetWindowPlacement(self.window_handle)
            return True
        except Exception:
            return False

    def capture_window(self):
        if not self.window_handle:
            raise ValueError("Window handle not set")

        if not self.window_exists():
            raise WindowClosedException("Target window no longer exists")

        width, height = self.get_window_dimensions()

        window_dc = win32gui.GetWindowDC(self.window_handle)
        dc_obj = win32ui.CreateDCFromHandle(window_dc)
        compatible_dc = dc_obj.CreateCompatibleDC()

        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(dc_obj, width, height)
        compatible_dc.SelectObject(bitmap)

        compatible_dc.BitBlt((0, 0), (width, height), dc_obj, (0, 0), win32con.SRCCOPY)

        bitmap_info = bitmap.GetInfo()
        bitmap_bits = bitmap.GetBitmapBits(True)
        img = np.frombuffer(bitmap_bits, dtype=np.uint8)
        img.shape = (height, width, 4)  # RGBA

        win32gui.DeleteObject(bitmap.GetHandle())
        compatible_dc.DeleteDC()
        dc_obj.DeleteDC()
        win32gui.ReleaseDC(self.window_handle, window_dc)

        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    def record(self):
        width, height = self.get_window_dimensions()

        if not self.video_writer:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                str(self.output_path), fourcc, self.fps, (width, height)
            )

        while self.is_recording:
            try:
                if not self.window_exists():
                    print(f"Window '{self.window_title}' was closed, stopping recording")
                    self.is_recording = False
                    break

                frame = self.capture_window()
                self.video_writer.write(frame)
                time.sleep(1 / self.fps)
            except WindowClosedException:
                print(f"Window '{self.window_title}' was closed, stopping recording")
                self.is_recording = False
                break
            except Exception as e:
                print(f"Error during recording: {e}")
                self.is_recording = False
                break

    def start_recording(self):
        if self.is_recording:
            print("Already recording")
            return

        self.is_recording = True
        self.record_thread = threading.Thread(target=self.record)
        self.record_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.record_thread:
            self.record_thread.join()

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None


if __name__ == "__main__":
    recorders = []
    try:
        recorders = [
            Recorder(**__SCREEN_RECORDING_CONFIG['VR']),
            Recorder(**__SCREEN_RECORDING_CONFIG['PLANNER'])
        ]

        for recorder in recorders:
            recorder.start_recording()

        while any(map(lambda r: r.is_recording, recorders)):
            time.sleep(0.1)

    finally:
        for recorder in recorders:
            recorder.stop_recording()
