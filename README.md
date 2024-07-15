Please run the code with the provided yolov8 model or any other yolov8 model. the program amy not be compatible with early yolo models

The cut in detection condition works on the principles of apparent width, time to collision, the amount by which the detection object has moved and proximity. 

If object has not moved a lot and is maintaining close distance, it is not considered cut in

If obect has moved quite a bit into the region ahead of the car, cut in is detected. 

Below are some explainatory images: 

![WhatsApp Image 2024-07-15 at 00 49 59_57e05c7d](https://github.com/user-attachments/assets/a72bd68f-c29d-430d-8d81-8ece83f62b4b)

The Blue line is the apparent width of the car if it was at that distance, while the Red line is the actual width of the car. the yellow lines connect these two and act as a boundary. In the above image, the car is only skimming the boundary.

![w2](https://github.com/user-attachments/assets/47ded64c-7ac0-4d5e-bd7b-de6ee8faac63)

In the above image, a car has breached into the region and it has done so by moving a lot, indicating that its not a car that is gently shifting lanes or maintaining a close distance, but has just cut in front of our car.

NOTE:
Iou has been used to compare if an object has newly entered the frame and by how much it has moved.

Actual width and apparent width are both caculated in meters but then converted into pixels.

Please feel free to alter the parameters like speed, focal length etc at the beginning of the program. These are intended to be dynamic values.

Thank You -
Praneeth K
Sampreeth 
Harshak 
Shrish
