# YOLOV8-Based-Smart-Traffic-Management-System  

## 📌 Overview  
This project is an **AI + IoT powered traffic management system** designed to reduce **urban congestion** and ensure **emergency vehicle prioritization**.  
It uses **YOLOv8 object detection** to identify vehicles, detect congestion, and recognize emergency vehicles (ambulances, fire trucks, police).  
Based on detections, the system **dynamically adjusts traffic lights** in real-time using a **Raspberry Pi and GPIO-controlled LEDs**.  

---

## 🎯 Features  
- 🔍 **Emergency Vehicle Detection** (YOLOv8 trained on custom dataset)  
- 🚗 **Congestion Analysis** (lane-wise vehicle counting)  
- 🚦 **Adaptive Traffic Signal Control**  
  - Lane with more vehicles gets priority  
  - Empty lanes stay red  
  - Emergency vehicles instantly get green  
- 💡 **Raspberry Pi GPIO Integration** to control physical LEDs (simulating traffic lights)   

---

## 🛠️ Tech Stack  
- **AI/ML:** Python, YOLOv8, OpenCV, NumPy  
- **IoT:** Raspberry Pi 4, GPIO LEDs. 

---

## 📊 Results  

| Metric                                    | Result   |
|-------------------------------------------|----------|
| Emergency Vehicle Detection Accuracy (mAP@50) | **92%** |
| Average Inference Latency (on Raspberry Pi 4) | **<200 ms** |
| Reduction in Average Traffic Wait Time        | **25%** |
| Emergency Vehicle Priority Response Time      | **<2 seconds** |


---

## 🔮 Future Improvements  
- 🌙 Improve detection in low-light and night conditions using **IR cameras** and **image enhancement techniques**  
- ☁️ Deploy a **cloud-based dashboard** for monitoring multiple intersections in real-time  
- 🚦 Integration with **real-world traffic light controllers** for large-scale implementation  
- 📊 Add **multi-lane congestion optimization** with database logging & predictive analytics  
- 🤖 Explore **reinforcement learning** for adaptive signal timing optimization  

---

## 👨‍💻 Author  
**Akshit**  
- ✉️ Email: kharbakshit010@gmail.com  

