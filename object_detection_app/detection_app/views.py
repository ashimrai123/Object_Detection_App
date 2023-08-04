
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image, ImageDraw
from yolov5 import YOLOv5
import os 
from django.conf import settings

def get_class_labels():
    return {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        29: 'frisbee',
        30: 'skis',
        31: 'snowboard',
        32: 'sports ball',
        33: 'kite',
        34: 'baseball bat',
        35: 'baseball glove',
        36: 'skateboard',
        37: 'surfboard',
        38: 'tennis racket',
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        61: 'toilet',
        62: 'tv',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',
        66: 'keyboard',
        67: 'cell phone',
        68: 'microwave',
        69: 'oven',
        70: 'toaster',
        71: 'sink',
        72: 'refrigerator',
        73: 'book',
        74: 'clock',
        75: 'vase',
        76: 'scissors',
        77: 'teddy bear',
        78: 'hair drier',
        79: 'toothbrush'
    }

def detect_objects(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        image = Image.open(uploaded_image)

        # Perform object detection
        model = YOLOv5('yolov5n.pt')
        results = model.predict(image)

        # Get class labels
        class_labels = get_class_labels()

        # Define colors for bounding boxes
        colors = ['red', 'green', 'blue', 'yellow', 'orange']

        # Prepare data for rendering
        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)
        detections = []
        for i, result in enumerate(results.pred[0]):
            class_index = result[0]
            confidence_score = result[1]
            bbox = result[2:]
            xmin, ymin, xmax, ymax = bbox.tolist()

            class_name = class_labels.get(int(class_index), 'Unknown')
            color = colors[i % len(colors)]
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)
            draw.text((xmin, ymin - 10), f'{class_name}: {confidence_score:.2f}', fill=color)

            # Prepare data for rendering checkboxes
            tick_icon = '✔️'
            cross_icon = '❌'
            checkbox_key = f'{class_name}_{i}_{xmin}_{ymin}_{xmax}_{ymax}'
            checkbox_label = f'{tick_icon} Select {class_name}'
            checkbox_remove_label = f'{cross_icon} Remove {class_name}'
            detections.append({
                'class_name': class_name,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'checkbox_key': checkbox_key,
                'checkbox_label': checkbox_label,
                'checkbox_remove_label': checkbox_remove_label,
            })

        # Save the output image
        output_path = 'output.jpg'
        output_image.save(os.path.join(settings.MEDIA_ROOT, output_path))

        return render(request, 'result.html', {
            'uploaded_image': os.path.join(settings.MEDIA_URL, str(uploaded_image)),
            'output_image_path': os.path.join(settings.MEDIA_URL, output_path),
            'detections': detections,
        })

    return render(request, 'index.html')
