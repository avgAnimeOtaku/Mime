import virtualvideo
import cv2


class MyVideoSource(virtualvideo.VideoSource):
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        _, img = self.cam.read()
        size = img.shape
        #opencv's shape is y,x,channels
        self._size = (size[1],size[0])

    def img_size(self):
        return self._size

    def fps(self):
        return 30

    def generator(self):
        while True:
            _, img = self.cam.read()
            yield img


vidsrc = MyVideoSource()
fvd = virtualvideo.FakeVideoDevice()
fvd.init_input(vidsrc)
fvd.init_output(2, 640, 480, pix_fmt='yuyv422')
fvd.run()