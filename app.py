#Import library
import tkinter as Tk
import cv2
import PIL.Image, PIL.ImageTk
import time
import threading
import argparse
import numpy as np
# Cai dat tham so doc weight, config va class name
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', default='model/yolov3-tiny.cfg',
                help='path to yolo config file')
ap.add_argument('-w', '--weights', default='model/yolov3-tiny_best.weights',
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default='model/yolov3-tiny.names',
                help='path to text file containing class names')
args = ap.parse_args()
 
# Ham tra ve output layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
 
# Ham ve cac hinh chu nhat va ten class
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(args.weights, args.config)
"""
    >> ClassName: MyCamera
    >> Mô tả    : Các thông số cần thiết cho 1 camera
    >> Thuoc tinh:
        - video_src: ip of camera
        - width,heigh,fps
        - vid   : read capture of video => set fps,with,height
        >>using for start
        - ret,frame: read from video
        - convert_color, convert_pillow
        >>using for recording
        - recording,recording_filename,recording_writer
        >> start thread
        - running,thread
    >> Phuong thuc:
        - start_recording(file_name):
        - stop_recording(file_name)
        - record(frame)
        - process
        - get_frame
"""

class MyCamera:
    def __init__(self,video_src=0,width=None,height=None,fps=None):
        self.video_src  = video_src
        self.width      = width
        self.height     = height
        self.fps        = fps

        #Open the video source
        self.vid        = cv2.VideoCapture(video_src)
        if not self.vid.isOpened():
            raise ValueError("[MyCamera] Unable to open video source",video_src)
        #Get video source width and height and fps
        if not self.width:
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))    # convert float to int
        if not self.height:
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  # convert float to int
        if not self.fps:
            self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))  # convert float to int

        #default value at start
        self.ret = False
        self.frame = None

        self.convert_color = cv2.COLOR_BGR2RGB
        #self.convert_color = cv2.COLOR_BGR2GRAY
        self.convert_pillow = True
        
        # default values for recording        
        self.recording = False
        self.recording_filename = 'output.mp4'
        self.recording_writer = None
        
        # start thread
        self.running = True
        self.thread = threading.Thread(target=self.process, args=(video_src,))
        self.thread.start()

    def start_recording(self, filename=None):
        if self.recording:
            print('[MyCamera] already recording:', self.recording_filename)
        else:
            if filename:
                self.recording_filename = filename
            else:
                self.recording_filename = time.strftime("%Y.%m.%d %H.%M.%S", time.localtime()) + ".avi"
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'MP42')
            self.recording_writer = cv2.VideoWriter(self.recording_filename, fourcc, self.fps, (self.width, self.height))
            self.recording = True
            print('[MyCamera] started recording:', self.recording_filename)
    
    def stop_recording(self):
        if not self.recording:
            print('[MyCamera] not recording')
        else:
            self.recording = False
            self.recording_writer.release()  #Stop record
            print('[MyCamera] stop recording:', self.recording_filename)
        
    def record(self,frame):
        # write frame to file         
        if self.recording_writer and self.recording_writer.isOpened():
            self.recording_writer.write(frame)
    
    def process(self,a):
        while self.running:
            start = time.time()
            ret,frame = self.vid.read()
            if ret:
                frame = cv2.resize(frame,(self.width,self.height))
                if self.recording:
                    self.record(frame)

                if self.convert_pillow:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = PIL.Image.fromarray(frame)
                    
            else:
                print('[Camera] stream end:', self.video_src)
                self.running = False
                if self.recording:
                    self.stop_recording()
                break
            
            #Assign new frame
            self.ret = ret
            self.frame = frame

            #sleep for next frame
            time.sleep(1/self.fps)
            print(a,time.time()-start)
    def process1(self,a):
        #count =0
        while self.running:
            ret,frame = self.vid.read()
            #count +=1
            if ret:
                frame = cv2.resize(frame,(self.width,self.height))
                if self.recording:
                    self.record(frame)

                Width = frame.shape[1]
                Height = frame.shape[0]
                scale = 0.00392
                blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(get_output_layers(net))
        
                # Loc cac object trong khung hinh
                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.4
        
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if (confidence > 0.5):
                            center_x = int(detection[0] * Width)
                            center_y = int(detection[1] * Height)
                            w = int(detection[2] * Width)
                            h = int(detection[3] * Height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])
        
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
                # Ve cac khung chu nhat quanh doi tuong
                for i in indices:
                    i = i[0]
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
        
                    draw_prediction(frame, class_ids[i], round(x), round(y), round(x + w), round(y + h))


                if self.convert_pillow:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = PIL.Image.fromarray(frame)
                    
            else:
                print('[Camera] stream end:', self.video_src)
                self.running = False
                if self.recording:
                    self.stop_recording()
                break
            
            #Assign new frame
            self.ret = ret
            self.frame = frame

            #sleep for next frame
            time.sleep(1/self.fps)
    def get_frame(self):
        return self.ret,self.frame

    def stop(self):
        if self.running:
            print("[Mycamera] stop camera",self.video_src)
            self.running = False
            self.thread.join()
        else:
            print("[Mycamera] already stop",self.video_src)
        #Stop stream
        if self.vid.isOpened():
            self.vid.release()
        self.stop_recording()
    def start(self):
        if not self.running:
            print("[Mycamera] start camera",self.video_src)
            self.vid        = cv2.VideoCapture(self.video_src)
            if not self.vid.isOpened():
                raise ValueError("[MyCamera] Unable to open video source",video_src)
            #Get video source width and height and fps
            if not self.width:
                self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))    # convert float to int
            if not self.height:
                self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  # convert float to int
            if not self.fps:
                self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))  # convert float to int

            #default value at start
            self.ret = False
            self.frame = None

            self.convert_color = cv2.COLOR_BGR2RGB
            #self.convert_color = cv2.COLOR_BGR2GRAY
            self.convert_pillow = True
            
            # default values for recording        
            self.recording = False
            self.recording_filename = 'output.mp4'
            self.recording_writer = None
            self.running = True
            self.thread = threading.Thread(target=self.process, args=("name",))
            self.thread.start()
        else:
            print("[Mycamera] already start",self.video_src)

    def __del__(self):
        #stop thread
        if self.running:
            self.running = False
            self.thread.join()

        #Stop stream
        if self.vid.isOpened():
            self.vid.release()    

"""
"""
class tkCamera(Tk.Frame):
    def __init__(self, window, text="", video_source=0, width=None, height=None):
        super().__init__(window)

        self.window = window
        self.video_source = video_source
        self.camera = MyCamera(self.video_source,width,height)

        self.label = Tk.Label(self,text= text)
        self.label.pack()


        self.canvas = Tk.Canvas(self,width=self.camera.width,height=self.camera.height)
        self.canvas.pack()

        #Button that lsts the user take a Recording
        self.btn_snapshot = Tk.Button(self, text="Start Record", command=self.start_record)
        self.btn_snapshot.pack(anchor='center', side='left')
    
        self.btn_snapshot = Tk.Button(self, text="Stop Record", command=self.stop_record)
        self.btn_snapshot.pack(anchor='center', side='left')
        

        # Button that lets the user take a snapshot
        self.btn_snapshot = Tk.Button(self, text="Snapshot", command=self.snapshot)
        self.btn_snapshot.pack(anchor='center', side='left')

        self.btn_snapshot = Tk.Button(self, text="Stop", command=self.stop_camera)
        self.btn_snapshot.pack(anchor='center', side='left')

        self.btn_snapshot = Tk.Button(self, text="Start", command=self.start_camera)
        self.btn_snapshot.pack(anchor='center', side='left')


        # After it is called once, the update method will be automatically called every delay milliseconds
        # calculate delay using `FPS`
        self.delay = int(1000/self.camera.fps)

        print('[tkCamera] source:', self.video_source)
        print('[tkCamera] fps:', self.camera.fps, 'delay:', self.delay)

        self.image = None
        
        self.running = True
        self.update_frame()
    
    def start_record(self):
        self.camera.start_recording()
    def stop_record(self):
        self.camera.stop_recording()



    def stop_camera(self):
        self.camera.stop()
    def start_camera(self):
        self.camera.start()
        


    def snapshot(self):
        if self.image:
            self.image.save(time.strftime("frame-%d-%m-%Y-%H-%M-%S.jpg"))
    
    def update_frame(self):
        ret,frame = self.camera.get_frame()

        if ret:
            self.image = frame
            self.photo = PIL.ImageTk.PhotoImage(image=self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        
        if self.running:
            self.window.after(self.delay,self.update_frame)



class App:
    def __init__(self,window,window_title,video_sources):
        self.window = window
        self.window.title(window_title)

        self.vids =[]
        
        columns = 4
        for number,source in enumerate(video_sources):
            text, stream = source
            vid = tkCamera(self.window, text, stream, 400, 200)
            x = number % columns
            y = number // columns
            vid.grid(row=y, column=x,padx=10, pady=10)
            self.vids.append(vid)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
    
    def on_closing(self, event=None):
        print('[App] stoping threads')
        for source in self.vids:
            source.camera.running = False
        print('[App] exit')
        self.window.destroy()


if __name__ == '__main__':     

    sources = [
        ('Video2', 'video2.mp4'),
        ('Video3', 'video3.mp4'),
        ('Video4', 'video4.mp4'),
        #('Video5', 'video5.mp4'),
        #('Video6', 'video6.mp4'),
        #('Video7', 'video7.mp4'),
        #('Video8', 'video8.mp4'),
        #('Video9', 'video9.mp4'),
        
    ]     
    # Create a window and pass it to the Application object
    App(Tk.Tk(), "Monitor camera App", sources)



