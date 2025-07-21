
import cv2
from scanner import *
from detector import *  
from tracker import * 
from viewer import *
import datetime

model = r'C:\Users\stefa\Documents\Projects\CardScanner\mmdetection\work_dirs\card_detector_rtm3\card_detector_rtm2.py'
weights = r'C:\Users\stefa\Documents\Projects\CardScanner\mmdetection\work_dirs\card_detector_rtm3\best_coco_bbox_mAP_epoch_3.pth'
imagePath = 'esempio2.jpg'
size = 1080
scoreThreshold = .5
save_image = True
mirror = False
include_flipped = True

def main():
    print("Starting process - inference with image")

    image_original = read_image(imagePath, size)
    image_copy = image_original.copy()

    # Initialize the DetInferencer
    detector =  Detector(model, weights)

    detections = detector.detect_objects(image_original, scoreThreshold)

    process_masks_to_cards(image_original, detections, mirror)
    hash_cards(detections, include_flipped)
    match_hashes(detections, include_flipped)

    draw_boxes(image_copy, detections)
    draw_masks(image_copy, detections)
    write_card_labels(image_copy, detections)

    # for detection in detections:
    #     if 'card_image' in detection:
    #         scanner.showImageWait(detection['card_image'])

    if save_image is True:
        # Save the image
        output_path = imagePath.split('.')[0]+'_output_image.jpg'
        cv2.imwrite(f'output/{output_path}', image_copy)
        print('Image saved successfully.')

    show_image_wait(image_copy)


if __name__ == '__main__':
    main()