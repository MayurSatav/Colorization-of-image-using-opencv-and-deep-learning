from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2


        
class Root(Tk):

    def __init__(self):
        super(Root, self).__init__()
        self.title("Tkinter Dialog Widget")
        self.minsize(300, 500)
        
        #self.wm_iconbitmap('icon.ico')

        self.labelFrame = ttk.LabelFrame(self, text ='Input Image')
        self.labelFrame.grid(column = 0, row = 1, padx = 5, pady = 5)
        
        self.labelFrame1 = ttk.LabelFrame(self, text ='Path')
        self.labelFrame1.grid(column = 0, row = 3, padx = 5, pady = 5)

        self.labelFrame2 = ttk.LabelFrame(self, text ='Image')
        self.labelFrame2.grid(column = 0, row = 4, padx = 5, pady = 5)

        self.labelFrame3 = ttk.LabelFrame(self, text ='Run')
        self.labelFrame3.grid(column = 0, row = 5, padx = 5, pady = 5)

        self.button()
        
    def button(self):
        self.button = ttk.Button(self.labelFrame, text = 'Browse File', width=50,command = self.fileDialog)
        self.button.grid(column = 0, row = 1)
        
        self.button1 = ttk.Button(self.labelFrame3, text = 'Run Program', width=50,command = self.RunPro)
        self.button1.grid(column = 0, row = 1)

    def RunPro(self):
    
        myimage = self.path

        # load our serialized black and white colorizer model and cluster
        # center points from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe("model/colorization_deploy_v2.prototxt", "model/colorization_release_v2.caffemodel")
        pts = np.load("model/pts_in_hull.npy")

        # add the cluster centers as 1x1 convolutions to the model
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

        # it will load the image from disk and scale the intensities of each pixel to the range [0, 1]
        image = cv2.imread(myimage)

        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        resized = cv2.resize(lab, (224, 224))# resize the Lab image to 224x224 
        L = cv2.split(resized)[0]
        L -= 50

        # pass the L channel through the network which will *predict* the 'a'
        # and 'b' channel values
        print("[INFO] colorizing image...")
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))#resize the predicted 'ab' volume

        # grab the 'L' channel from the *original* image
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR) # convert the output image to RGB
        colorized = np.clip(colorized, 0, 1)

        colorized = (255 * colorized).astype("uint8")
        colorized = cv2.resize(colorized, (300, 350))

        cv2.imshow("Colorized", colorized)
        cv2.waitKey(0)

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir= '/',title = 'select file', filetype = (('jpeg','*.jpg'),('All Files','*.*')))

        self.e1 = ttk.Entry(self.labelFrame1, width = 50)
        self.e1.insert(0, self.filename)
        self.e1.grid(row=2, column=0, columnspan=50)
        
        Root.OpenImage(self.filename)
        #place image
        
        newpath=self.filename
        self.path = newpath.replace('/','\\\\')
        print (self.path)
        
        im = Image.open(self.path)
        resized = im.resize((300, 300),Image.ANTIALIAS)
        tkimage = ImageTk.PhotoImage(resized)
        myvar=ttk.Label(self.labelFrame2,image = tkimage)
        myvar.image = tkimage
        myvar.grid(column=0, row=4)

    def OpenImage(self):
        pass

if __name__ == '__main__':
    
    root = Root()
    
    root.mainloop()
