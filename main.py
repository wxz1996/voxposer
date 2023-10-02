import argparse
from typing import Any
from matplotlib import pyplot as plt

from pathlib import Path

import cv2
import numpy as np
import torch
import open3d as o3d
from typing import List, Tuple, Any, Dict
from environment import FrankaKitchenVisionEnv
from transformers import OwlViTForObjectDetection, OwlViTProcessor
from huggingface_hub import hf_hub_download
from segment_anything import build_sam_vit_b, SamPredictor


def visualize_point_cloud(pointcloud):
    pcd = o3d.geometry.PointCloud()
    points, colors = pointcloud[:, 0:3], pointcloud[:, 3:6]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    o3d.visualization.draw_geometries([pcd])


def parse_args():
    parser = argparse.ArgumentParser(description='Run VoxPoser')
    parser.add_argument('--vis_rgb',
                        default=False,
                        action='store_true',
                        help='Visualize the point cloud')
    parser.add_argument('--vis_pointcloud',
                        default=False,
                        action='store_true',
                        help='Visualize the point cloud')
    parser.add_argument('--save_path',
                        type=Path,
                        default=Path('./scene_frames/'),
                        help='Save path for scene frames')
    args = parser.parse_args()
    return args


def fetch_segment_anything(device='cuda'):
    chkpt_path = hf_hub_download("ybelkada/segment-anything",
                                 "checkpoints/sam_vit_b_01ec64.pth")
    predictor = SamPredictor(build_sam_vit_b(checkpoint=chkpt_path))
    predictor.model.to(device)
    return predictor


class OpenVocabDetector():

    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32")

    def __call__(self, rgb_image: np.ndarray,
                 object_targets: List[str]) -> Dict[str, Any]:
        # save the rgb image as a file
        # plt.imshow(rgb_image)
        # plt.show()

        h, w, _ = rgb_image.shape

        assert len(
            object_targets) > 0, "Must provide at least one object target"
        for object_target in object_targets:
            assert isinstance(object_target,
                              str), "Object targets must be strings"
        detector_inputs = self.processor(text=[object_targets],
                                         images=[rgb_image],
                                         return_tensors="pt")
        detector_outputs = self.model(**detector_inputs)
        target_sizes = torch.Tensor([[h, w]])
        results_dict_list = self.processor.post_process_object_detection(
            outputs=detector_outputs,
            target_sizes=target_sizes,
            threshold=0.05)
        results_dict = results_dict_list[0]

        # convert boxes to tuples from tensor
        results_dict["boxes"] = [
            tuple([int(e) for e in box.tolist()])
            for box in results_dict["boxes"]
        ]

        def box_to_box_centroid(box) -> Tuple[int, int]:
            return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

        results_dict["box_centroids"] = [
            box_to_box_centroid(box) for box in results_dict["boxes"]
        ]
        results_dict["labels"] = [
            object_targets[label] for label in results_dict["labels"]
        ]
        return results_dict

    def draw_boxes(self, rgb_image: np.ndarray,
                   results: Dict[str, Any]) -> np.ndarray:
        bgr_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        print(f"Detected {len(boxes)} boxes")
        print(f"Detected {len(scores)} scores")
        print(f"Detected {len(labels)} labels")

        for box, score, label in zip(boxes, scores, labels):
            print(
                f"Detected {label} with confidence {round(score.item(), 3)} at location {box}"
            )
            # visualize the box in red
            cv2.rectangle(bgr_image, (box[0], box[1]), (box[2], box[3]),
                          (0, 0, 255), 2)
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


class ImageSegmenter():

    def __init__(self, device='cuda'):
        chkpt_path = hf_hub_download("ybelkada/segment-anything",
                                     "checkpoints/sam_vit_b_01ec64.pth")
        self.segment_anything = SamPredictor(
            build_sam_vit_b(checkpoint=chkpt_path))
        self.segment_anything.model.to(device)

    def _centroid_to_mask(self, centroids: Tuple[int, int],
                          bounding_box: Tuple[int, int, int, int]):
        assert isinstance(
            centroids,
            tuple), f"Centroids must be a tuple, got {type(centroids)}"
        assert len(centroids) == 2, "Centroids must be a tuple of length 2"

        assert isinstance(
            bounding_box,
            tuple), f"Bounding box must be a tuple, got {type(bounding_box)}"
        assert len(
            bounding_box) == 4, "Bounding box must be a tuple of length 4"
        x_start, y_start, x_end, y_end = bounding_box

        input_points = np.array(centroids).reshape(1, 2)
        input_labels = np.ones(input_points.shape[0], dtype=np.int32)
        masks, scores, logits = self.segment_anything.predict(
            input_points, input_labels, multimask_output=True)

        # We need to select which mask to use. We will use the mask with the highest IoU with the bounding box.

        bounding_box_mask = np.zeros_like(masks[0])
        bounding_box_mask[y_start:y_end, x_start:x_end] = 1

        # Visualize all masks

        # num_masks = len(masks)

        # plt.subplot(1, num_masks + 1, 1)
        # plt.imshow(bounding_box_mask)
        # plt.title("Bounding box mask")

        # for i in range(num_masks):
        #     plt.subplot(1, num_masks + 1, i + 2)
        #     plt.imshow(masks[i])
        #     plt.title(f"Mask {i}")

        # plt.show()

        def _compute_iou(mask1, mask2):
            assert mask1.shape == mask2.shape, "Masks must have the same shape"
            intersection = np.logical_and(mask1, mask2)
            union = np.logical_or(mask1, mask2)
            iou = np.sum(intersection) / np.sum(union)
            return iou

        ious = [_compute_iou(bounding_box_mask, mask) for mask in masks]

        max_idx = np.argmax(ious)
        mask = masks[max_idx]
        # breakpoint()
        return mask

    def __call__(self, rgb_image: np.ndarray,
                 centroids: List[List[Tuple[int, int]]],
                 bounding_boxes: List[Tuple[int, int, int, int]]):
        self.segment_anything.set_image(rgb_image)
        masks = [
            self._centroid_to_mask(centroid, bounding_box)
            for centroid, bounding_box in zip(centroids, bounding_boxes)
        ]
        return masks

    def visualize(self, rgb_image: np.ndarray, masks: List[np.ndarray]):
        h, w, _ = rgb_image.shape
        overlay = rgb_image.copy()
        for mask in masks:

            # breakpoint()
            # Convert boolean mask to a [h, w, 3] size mask with green color
            mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            mask_rgb[mask] = [255, 0, 0]
            # Create an alpha (transparency) channel based on the boolean mask
            alpha = np.zeros((h, w), dtype=np.uint8)
            alpha[mask] = 255  # Fully opaque where mask is True

            # Overlay the mask on the image
            overlay = cv2.addWeighted(
                overlay, 1, mask_rgb, 0.5,
                0)  # The 0.5 here is the transparency level

        return overlay


def main(args):

    env = FrankaKitchenVisionEnv(tasks_to_complete=['microwave', 'kettle'])
    env.reset()

    object_targets = ["a handle"]
    detector = OpenVocabDetector()
    segmenter = ImageSegmenter(device='cuda')

    # Clean out existing frames
    args.save_path.mkdir(exist_ok=True)
    for f in args.save_path.glob("*.png"):
        f.unlink()

    for env_idx in range(1000):
        print("Step", env_idx)
        rgb_image = env.render(render_mode='rgb_array')
        depth_image = env.render(render_mode='depth_array')

        # Detect objects
        print(f"Detecting {len(object_targets)} objects")
        detector_results = detector(rgb_image, object_targets)
        rgb_image_with_boxes = detector.draw_boxes(rgb_image, detector_results)

        segmentation_masks = segmenter(rgb_image_with_boxes,
                                       detector_results["box_centroids"],
                                       detector_results["boxes"])
        rgb_image_with_masks = segmenter.visualize(rgb_image_with_boxes,
                                                   segmentation_masks)

        if args.vis_rgb:
            # Display the result
            cv2.imshow('Overlay Image',
                       cv2.cvtColor(rgb_image_with_masks, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.vis_pointcloud:
            pointcloud = env.project_rgbd_to_pointcloud(
                depth_image, rgb_image, np.linalg.inv(env.K))
            visualize_point_cloud(pointcloud)

        # Save the images
        cv2.imwrite(str(args.save_path / f"rgb_image_{env_idx:06d}.png"),
                    cv2.cvtColor(rgb_image_with_masks, cv2.COLOR_RGB2BGR))

        _ = env.step(env.action_space.sample())

    env.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)