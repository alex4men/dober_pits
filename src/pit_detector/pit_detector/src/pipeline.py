import os
from PIL import Image
import json
import argparse

import cv2

from tracker import Tracker
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T

from torchvision.ops import box_convert, nms
import torch
import numpy as np
import supervision as sv

import warnings
warnings.filterwarnings("ignore")


def img_to_tensor(image: np.array) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(Image.fromarray(image), None)
    return image_transformed


class Pipeline:


    def __init__(self, 
                 config_path: str = '../model_data/GroundingDINO_SwinB_cfg.py', 
                 weights_path: str = '../model_data/groundingdino_swinb_cogcoor.pth'):
        self.model = load_model(config_path, weights_path)


    def infer_video(self, prompt: str, vin_path: str, vout_path: str, 
                    json_path: str = None, box_thresh: float = 0.2, 
                    text_thresh: float = 0.2, nms_thresh: float = 0.5) -> dict:

        if not os.path.isfile(vin_path):
            return
        
        box_annotator = sv.BoxAnnotator()
        tracker = Tracker()

        cap = cv2.VideoCapture(vin_path)
        ret, frame = cap.read()

        cap_out = cv2.VideoWriter(vout_path, 
                                  cv2.VideoWriter_fourcc(*'MP4V'), 
                                  cap.get(cv2.CAP_PROP_FPS),
                                  (frame.shape[1], frame.shape[0]))

        i = 0
        frames = []
        while ret:
            print('Processing frame %d\r'%i, end="")
            detections = []
            # Process only 1 FPS
            if i % cap.get(cv2.CAP_PROP_FPS) != 0:
                i += 1
                ret, frame = cap.read()
                continue
            boxes, logits, phrases = predict(
                model=self.model, 
                image=img_to_tensor(frame), 
                caption=prompt, 
                box_threshold=box_thresh, 
                text_threshold=text_thresh
            )

            h, w, _ = frame.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxys = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
            idxs = nms(xyxys, logits, nms_thresh)

            for xyxy, score, phrase in zip(xyxys.index_select(0, idxs).numpy(), logits.index_select(0, idxs), [phrases[idx] for idx in idxs]):
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                
                detections.append([x1, y1, x2, y2, score, phrase])

            tracker.update(frame, detections)

            try:
                detections = sv.Detections(xyxy=np.array([track.bbox for track in tracker.tracks]), tracker_id=np.array([track.track_id for track in tracker.tracks]))


                # annotated_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=[track.phrase for track in tracker.tracks])

                cap_out.write(annotated_frame)
            
                frame_objects = [{"track_id": track.track_id, 
                                "name": track.phrase, 
                                "bbox": [int(x) for x in track.bbox]} 
                                for track in tracker.tracks]
            
                frames.append({"timestamp": i/cap.get(cv2.CAP_PROP_FPS), 
                            "objects": frame_objects})

            except:
                cap_out.write(frame)
            finally:
                i += 1
                ret, frame = cap.read()

        res_dict = {'active_frames': frames}
        if json_path is not None:
            json.dump(res_dict, open(json_path, 'w'), indent=2, ensure_ascii=False)
        return res_dict


if __name__ == '__main__':

    HOME = '..'
    CONFIG_PATH = os.path.join(HOME, "model_data/GroundingDINO_SwinB_cfg.py")
    print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))
    WEIGHTS_NAME = "groundingdino_swinb_cogcoor.pth"
    WEIGHTS_PATH = os.path.join(HOME, "model_data", WEIGHTS_NAME)
    print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

    TEXT_PROMPT = "pothole" #"crane, excavator"
    BOX_TRESHOLD = 0.20
    TEXT_TRESHOLD = 0.20
    
    VIN_PATH = os.path.join('test_data', 'pits_demo.m4v')
    VOUT_PATH = os.path.join('test_data', 'pits_demo_out.mp4')
    JSON_PATH = os.path.join('test_data', 'pits_demo_out.json')

    pipe = Pipeline(CONFIG_PATH, WEIGHTS_PATH)

    pipe.infer_video(TEXT_PROMPT, VIN_PATH, VOUT_PATH, JSON_PATH, BOX_TRESHOLD, TEXT_TRESHOLD)
