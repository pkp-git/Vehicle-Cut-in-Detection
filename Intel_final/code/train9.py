import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define vehicle classes
vehicle_classes = ["car", "truck", "bus", "motorbike", "bicycle"]

# Define focal length (in pixels) and real-world width of a car (in meters)
focal_length = 700  # Example value, replace with actual focal length
real_car_width = 1.8  # Average width of a car in meters

# Function to detect vehicles in an image
def detect_vehicles(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Prepare the image for YOLO model
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected bounding boxes, confidences, class IDs, and distances
    boxes = []
    confidences = []
    class_ids = []
    distances = []

    # Extract information from the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in vehicle_classes:
                # Get the bounding box coordinates and dimensions
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                # Calculate distance
                distance = (real_car_width * focal_length) / w
                distances.append(distance)

    # Apply non-max suppression to eliminate redundant overlapping boxes with lower confidences
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_vehicles = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            detected_vehicles.append((x, y, w, h, distances[i]))

    return detected_vehicles, img, width, height

# Intersection over Union (IoU) function
def iou(box1, box2):
    x1, y1, w1, h1, _ = box1
    x2, y2, w2, h2, _ = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

# Detect potential cut-ins and calculate speed and TTC
def detect_potential_cut_ins_and_ttc(frame1_vehicles, frame2_vehicles, width, height, time_interval, distance_threshold=3.5, iou_threshold=0.3):
    potential_cut_ins = []
    ttc_vehicles = []

    for box2 in frame2_vehicles:
        best_match = None
        best_iou = 0
        for box1 in frame1_vehicles:
            current_iou = iou(box1, box2)
            if current_iou > best_iou:
                best_iou = current_iou
                best_match = box1

        if best_iou < iou_threshold:
            # This is a new vehicle or a vehicle that has moved significantly
            x2, y2, w2, h2, dist2 = box2
            if dist2 < distance_threshold:  # Check if the vehicle is within the distance threshold
                distance_from_center = np.sqrt((x2 + w2/2 - width/2)**2 + (y2 + h2/2 - height)**2)
                if x2 < width / 2 and distance_from_center < width / 2:  # Assuming cut-in from left
                    speed = w2 / time_interval
                    if speed > 0:
                        ttc = distance_from_center / speed  # time to collision in seconds
                        if ttc < 2:  # Dangerous if TTC is less than 2 seconds
                            potential_cut_ins.append(box2)
                            ttc_vehicles.append((box2, ttc))

    return potential_cut_ins, ttc_vehicles

# Paths to images
frame_paths =  [
    "images2/0001799.jpeg", "images2/0001800.jpeg", "images2/0001801.jpeg",
    "images2/0001802.jpeg", "images2/0001824.jpeg", "images2/0001825.jpeg",
    "images2/0001826.jpeg", "images2/0001827.jpeg", "images2/0001828.jpeg",
    "images2/0001829.jpeg"
]

# Time interval between consecutive frames (in seconds)
time_interval = 0.1  # Example value, adjust as needed

# Process all frames
for i in range(len(frame_paths) - 1):
    frame1_path = frame_paths[i]
    frame2_path = frame_paths[i + 1]

    # Detect vehicles in both frames
    frame1_vehicles, frame1_img, width, height = detect_vehicles(frame1_path)
    frame2_vehicles, frame2_img, _, _ = detect_vehicles(frame2_path)

    # Detect potential cut-ins and calculate TTC between the two frames
    potential_cut_ins, ttc_vehicles = detect_potential_cut_ins_and_ttc(frame1_vehicles, frame2_vehicles, width, height, time_interval)

    # Draw bounding boxes and calculate distances
    for (x, y, w, h, distance) in frame2_vehicles:
        color = (0, 255, 0)  # Green color for bounding box
        if distance < 3.5:
            color = (0, 0, 255)  # Red color if distance is less than 3.5 meters

        # Draw bounding box
        cv2.rectangle(frame2_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame2_img, f'Dist: {distance:.2f}m', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw TTC for cut-in vehicles
    for ((x, y, w, h, distance), ttc) in ttc_vehicles:
        color = (0, 0, 255)  # Red color for TTC
        cv2.putText(frame2_img, f'TTC: {ttc:.2f}s', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame2_img, f'Dist: {distance:.2f}m', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Add the time to the image
    current_time = time_interval * (i + 1)
    cv2.putText(frame2_img, f'Time: {current_time:.1f}s', (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the result using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(frame2_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Frame {i+2}')
    plt.show()
