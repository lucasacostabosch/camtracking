#trackManager.py
from ultralytics import YOLO
import cv2

class TrackManager:
    def __init__(self):
        self.chosen_model = YOLO("./models/yolov8n.pt")
        self.tracker = './models/botsort.yaml'
        self.resize_width = 1024 
        self.resize_height = 720  
        self.exclusion_line_y = self.resize_height * 4//7
        self.exclusion_line_x = self.resize_width - 150

    def predict(self, img, classes=[]):
        img = cv2.resize(img, (self.resize_width, self.resize_height))
        if classes:
            results = self.chosen_model.track(img, show=True, persist=True, tracker=self.tracker)
        else:
            results = self.chosen_model.track(img, show=False, persist=True, tracker=self.tracker)
        return results

    def predict_and_detect(self, img, classes=[]):
        img = cv2.resize(img, (self.resize_width, self.resize_height))
        results = self.predict(img, classes)
        
        countFila2 = 0
        countFila1 = 0
        
        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls[0])] == "person":
                    
                    if int(box.xyxy[0][1]) < self.exclusion_line_y and int(box.xyxy[0][0]) > self.exclusion_line_x:
                        countFila2 += 1
                    
                    elif int(box.xyxy[0][0]) < self.exclusion_line_x and int(box.xyxy[0][1]) < self.exclusion_line_y:
                        countFila1 += 1
                    else: continue

                    cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                  (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
                    cv2.putText(img, f"Persona {box.conf[0]:.2f}",
                                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        
        cv2.putText(img, f"Fila 1: {countFila1}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(img, f"Fila 2: {countFila2}", (880, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        ##cv2.line(img, (0, self.exclusion_line_y), (self.resize_width, self.exclusion_line_y), (255, 0, 0), 1)
        ##cv2.line(img, (self.exclusion_line_x, 0), (self.exclusion_line_x, self.resize_height), (255, 0, 0), 1)
        
        return img

    def process_frame(self, frame):
        result_frame = self.predict_and_detect(frame)
        return result_frame
