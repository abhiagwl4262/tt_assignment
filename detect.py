import argparse
import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch


from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    check_requirements,
    non_max_suppression,
    print_args,
    scale_boxes,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    output,
    weights,  
    source,  
    imgsz=(640, 640),  
    conf_thres=0.4,  
    iou_thres=0.4,
    max_det=1000,
    device="",
    save_conf=False,
    visualize=False
):
    source = str(source)

    # # Directories
    if visualize:
        path = os.path.join(output, "plots")
        Path(path).mkdir(parents=True, exist_ok=True)
    else:
        os.makedirs(output, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    seen, _, dt = 0, [], (Profile(), Profile(), Profile())
    labels_dict = {
        "filename": [],
        "xmin": [],
        "ymin": [],
        "xmax": [],
        "ymax": [],
        "class": [],
    }

    for path, im, im0s, _, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, None, False, max_det=max_det
            )

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0 = path, im0s.copy()
            h, w = im0s.shape[:2]
            p = Path(p)  # to Path
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                image_name = os.path.basename(path)
                for *xyxy, conf, cls in reversed(det):
                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )  # normalized xywh
                    line = (
                        (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    )  # label format
                    label = int(line[0].detach().cpu().numpy().tolist())

                    xc, yc, w_box, h_box = line[1:]
                    xc = float(xc) * w
                    yc = float(yc) * h
                    w_box = float(w_box) * w
                    h_box = float(h_box) * h

                    x1 = int(xc - w_box / 2.0)
                    y1 = int(yc - h_box / 2.0)
                    x2 = int(xc + w_box / 2.0)
                    y2 = int(yc + h_box / 2.0)

                    labels_dict["filename"].append(image_name)
                    labels_dict["xmin"].append(x1)
                    labels_dict["ymin"].append(y1)
                    labels_dict["xmax"].append(x2)
                    labels_dict["ymax"].append(y2)
                    labels_dict["class"].append(label)

                    if visualize:
                        cv2.rectangle(im0s, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                labels_dict["filename"].append(image_name)
                labels_dict["xmin"].append(0)
                labels_dict["ymin"].append(0)
                labels_dict["xmax"].append(1)
                labels_dict["ymax"].append(1)
                labels_dict["class"].append(0)

        if visualize:
            out_path = os.path.join(output, "plots", image_name)
            cv2.imwrite(out_path, im0s)

        # Print time (inference-only)
        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms"
        )

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}"
        % t
    )

    df = pd.DataFrame(labels_dict, index=None)
    np.savetxt(
        f"{output}/results.csv", df, delimiter=";", fmt=["%s", "%d", "%d", "%d", "%d", "%d"]
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str, default="output", help="path to save model output"
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="best.pt",
        help="model path",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/images",
        help="Path to image directory",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument('--visualize', action='store_true',
                        help='visualize results')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements("requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
