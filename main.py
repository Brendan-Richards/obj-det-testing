import cv2
import time
import torch
from PIL import Image
import torchvision
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor


FPS = 10
NUM_FRAMES = 100
HF_MODEL_NAME = "PekingU/rtdetr_v2_r18vd"
CONFIDENCE_THRESHOLD = 0.5
DEVICE = "cuda"

image_processor = RTDetrImageProcessor.from_pretrained(HF_MODEL_NAME, device=DEVICE)
model = RTDetrV2ForObjectDetection.from_pretrained(HF_MODEL_NAME).to(DEVICE)
id_to_label = model.config.id2label

# 0 is usually the default USB camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"frame height: {h}, frame width: {w}")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, FPS, (w,  h))

i = 0
start = time.time()
while i < NUM_FRAMES:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # run detection inference
    display_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(display_img)
    inputs = image_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(img.height, img.width)]), threshold=CONFIDENCE_THRESHOLD)
    # (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    label_strings = [id_to_label[int(x)] for x in results[0]["labels"]]
    boxes = results[0]["boxes"]

    display_img = torchvision.utils.draw_bounding_boxes(
        image=torch.from_numpy(display_img).permute(2, 0, 1),
        boxes=boxes,
        labels=label_strings,
    )

    display_img = display_img.permute(1, 2, 0).numpy()
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

    # cv2.imshow("USB Camera", frame)q
    out.write(display_img)

    i += 1

    # Press 'q' to quit
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

elapsed = time.time() - start
print(f"elapsed time: {round(elapsed, 2)} seconds")
print(f"fps: {round(NUM_FRAMES / elapsed, 2)}")

cap.release()
out.release()
cv2.destroyAllWindows()