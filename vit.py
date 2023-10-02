import requests
from PIL import Image
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# img_result = requests.get(url, stream=True).raw

image_path = "./robot_scene.png"
# image_path = "./cluttered_table.jpg"
image = Image.open(image_path).convert("RGB")

texts = [["a microwave", "a sink", "a cupboard", "a kettle"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.01)
i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Display image with bounding boxes
fig, ax = plt.subplots(1)
ax.imshow(image)
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
    
    # Plot rectangle for bounding box
    x, y, xmax, ymax = box
    rect = Rectangle((x, y), xmax - x, ymax - y, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()


# for box, score, label in zip(boxes, scores, labels):
#     box = [round(i, 2) for i in box.tolist()]
#     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")