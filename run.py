import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageTk
from ultralytics.yolo.utils.plotting import Annotator
import tensorflow as tf 
import math 
import tkinter as tk


categories = [
   'Construction Helmet', 
   'No Construction Helmet',
   'Not Construction Helmet',
]

COLORS = {
   'No Construction Helmet': (255, 0, 0),
   'Construction Helmet': (0,155,0),
   'Not Construction Helmet': (226,221,68)
}

IMG_SIZE = 224

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CHS Monitoring ")
        self.root.geometry("900x750")
        self.root.config(bg="#650cc3")
        
        self.camera_label = tk.Label(self.root, bg="#650cc3")
        self.camera_label.pack(padx=20, pady=20, expand=True, side=tk.TOP)
        
        self.start_button= tk.Button(self.root, text="Start Monitoring", height="2", width="20", border=0, fg='#052455', bg='#00ffce', font=("Helvetica", 12, "bold"), command=self.start_camera)
        self.start_button.pack(padx=10, pady=10, side="left")


        self.stop_button= tk.Button(self.root, text="Stop Monitoring", height="2", width="20", border=0, fg='white', bg="#2c005b", font=("Helvetica", 12, "bold"), command=self.stop_camera)
        self.stop_button.pack(padx=10, pady=10, side="left", anchor="center")

        self.count_label = tk.Label(self.root, text="Detected helmet in real-time: ", fg='white', bg="#650cc3",font=("Helvetica", 14, "bold"))
        self.count_label.pack(padx=10, pady=10, expand=True, side="left")

        
        self.model = YOLO("bestnew500.pt")
        self.new_model = tf.keras.models.load_model('ConstructionHelmetPrediction123.hdf5')
        self.vc = None
        self.counts = {category: 0 for category in categories}
        self.root.mainloop()
    
    def start_camera(self):
        self.vc = cv2.VideoCapture(0)
        self.update_camera()
    
    def stop_camera(self):
        self.root.quit()    


    def update_camera(self):
        rval, frame = self.vc.read()
        if rval:
            frame = np.asarray(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds = np.copy(frame)

            results = self.model.predict(frame)
            self.counts = {c: 0 for c in categories}
            for r in results:
                annotator = Annotator(frame)
                boxes = r.boxes
                for box in boxes:
                    b = [math.ceil(i) for i in box.xyxy[0]]
                    c = box.cls
                    crop_img = preds[ b[1]:b[1]+abs(b[3] - b[1]),b[0]:b[0]+abs(b[2]-b[0])]
                    crop_img = cv2.resize(crop_img, (IMG_SIZE,IMG_SIZE) )
                    crop_img = tf.keras.preprocessing.image.img_to_array(crop_img) 
                    crop_img /= 255. 
                    crop_img = np.expand_dims(crop_img,0)
                    crop_img = self.new_model.predict(crop_img)
                    crop_img = np.argmax(crop_img[0])
                    category = categories[crop_img]
                    self.counts[category] += 1
                    annotator.box_label(b, f'{category}', color=COLORS[category])
            frame = cv2.resize(frame, (828 , 566))
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            self.camera_label.config(image=photo)
            self.camera_label.image = photo
            
            count_text = "Detected helmet in real-time:  "
            for category in categories:
                count_text += "\n"f"{category}: {self.counts[category]}  "
            self.count_label.config(text=count_text)
        self.root.after(20, self.update_camera)
    
if __name__ == "__main__":
    app = App()