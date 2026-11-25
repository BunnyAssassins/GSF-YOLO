# Derived from Ultralytics; AGPL-3.0. See the repository LICENSE.
"""
Ultralytics Results, Boxes, Masks, Keypoints, Probs, and OBB classes for handling inference results.

Usage: See https://docs.ultralytics.com/modes/predict/
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, DataExportMixin, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box

class BaseTensor(SimpleClass):
    """Base tensor class with additional methods for easy manipulation and device handling.

    This class provides a foundation for tensor-like objects with device management capabilities, supporting both
    PyTorch tensors and NumPy arrays. It includes methods for moving data between devices and converting between tensor
    types.

    Attributes:
        data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
        orig_shape (tuple[int, int]): Original shape of the image, typically in the format (height, width).

    Methods:
        cpu: Return a copy of the tensor stored in CPU memory.
        numpy: Return a copy of the tensor as a numpy array.
        cuda: Move the tensor to GPU memory, returning a new instance if necessary.
        to: Return a copy of the tensor with the specified device and dtype.

    Examples:
        >>> import torch
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> cpu_tensor = base_tensor.cpu()
        >>> numpy_array = base_tensor.numpy()
        >>> gpu_tensor = base_tensor.cuda()
    """

    def __init__(self, data: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """Initialize BaseTensor with prediction data and the original shape of the image.

        Args:
            data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
            orig_shape (tuple[int, int]): Original shape of the image in (height, width) format.
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), 'data must be torch.Tensor or np.ndarray'
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying data tensor.

        Returns:
            (tuple[int, ...]): The shape of the data tensor.

        Examples:
            >>> data = torch.rand(100, 4)
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            (100, 4)
        """
        return self.data.shape

    def cpu(self):
        """Return a copy of the tensor stored in CPU memory.

        Returns:
            (BaseTensor): A new BaseTensor object with the data tensor moved to CPU memory.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> cpu_tensor = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseTensor)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """Return a copy of this object with its data converted to a NumPy array.

        Returns:
            (BaseTensor): A new instance with `data` as a NumPy array.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> numpy_tensor = base_tensor.numpy()
            >>> print(type(numpy_tensor.data))
            <class 'numpy.ndarray'>
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """Move the tensor to GPU memory.

        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to GPU memory.

        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> gpu_tensor = base_tensor.cuda()
            >>> print(gpu_tensor.data.device)
            cuda:0
        """
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """Return a copy of the tensor with the specified device and dtype.

        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().
            **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().

        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to the specified device and/or dtype.

        Examples:
            >>> base_tensor = BaseTensor(torch.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor = base_tensor.to("cuda")
            >>> float16_tensor = base_tensor.to(dtype=torch.float16)
        """
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self) -> int:
        """Return the length of the underlying data tensor.

        Returns:
            (int): The number of elements in the first dimension of the data tensor.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Return a new BaseTensor instance containing the specified indexed elements of the data tensor.

        Args:
            idx (int | list[int] | torch.Tensor): Index or indices to select from the data tensor.

        Returns:
            (BaseTensor): A new BaseTensor instance containing the indexed data.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> result = base_tensor[0]  # Select the first row
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        return self.__class__(self.data[idx], self.orig_shape)

class Results(SimpleClass, DataExportMixin):
    """A class for storing and manipulating inference results.

    This class provides comprehensive functionality for handling inference results from various Ultralytics models,
    including detection, segmentation, classification, and pose estimation. It supports visualization, data export, and
    various coordinate transformations.

    Attributes:
        orig_img (np.ndarray): The original image as a numpy array.
        orig_shape (tuple[int, int]): Original image shape in (height, width) format.
        boxes (Boxes | None): Detected bounding boxes.
        masks (Masks | None): Segmentation masks.
        probs (Probs | None): Classification probabilities.
        keypoints (Keypoints | None): Detected keypoints.
        obb (OBB | None): Oriented bounding boxes.
        speed (dict): Dictionary containing inference speed information.
        names (dict): Dictionary mapping class indices to class names.
        path (str): Path to the input image file.
        save_dir (str | None): Directory to save results.

    Methods:
        update: Update the Results object with new detection data.
        cpu: Return a copy of the Results object with all tensors moved to CPU memory.
        numpy: Convert all tensors in the Results object to numpy arrays.
        cuda: Move all tensors in the Results object to GPU memory.
        to: Move all tensors to the specified device and dtype.
        new: Create a new Results object with the same image, path, names, and speed attributes.
        plot: Plot detection results on an input BGR image.
        show: Display the image with annotated inference results.
        save: Save annotated inference results image to file.
        verbose: Return a log string for each task in the results.
        save_txt: Save detection results to a text file.
        save_crop: Save cropped detection images to specified directory.
        summary: Convert inference results to a summarized dictionary.
        to_df: Convert detection results to a Polars DataFrame.
        to_json: Convert detection results to JSON format.
        to_csv: Convert detection results to a CSV format.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> result = results[0]  # Get the first result
        >>> boxes = result.boxes  # Get the boxes for the first result
        >>> masks = result.masks  # Get the masks for the first result
        >>> for result in results:
        ...     result.plot()  # Plot detection results
    """

    def __init__(self, orig_img: np.ndarray, path: str, names: dict[int, str], boxes: torch.Tensor | None=None, masks: torch.Tensor | None=None, probs: torch.Tensor | None=None, keypoints: torch.Tensor | None=None, obb: torch.Tensor | None=None, speed: dict[str, float] | None=None) -> None:
        """Initialize the Results class for storing and manipulating inference results.

        Args:
            orig_img (np.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (dict): A dictionary of class names.
            boxes (torch.Tensor | None): A 2D tensor of bounding box coordinates for each detection.
            masks (torch.Tensor | None): A 3D tensor of detection masks, where each mask is a binary image.
            probs (torch.Tensor | None): A 1D tensor of probabilities of each class for classification task.
            keypoints (torch.Tensor | None): A 2D tensor of keypoint coordinates for each detection.
            obb (torch.Tensor | None): A 2D tensor of oriented bounding box coordinates for each detection.
            speed (dict | None): A dictionary containing preprocess, inference, and postprocess speeds (ms/image).

        Notes:
            For the default pose model, keypoint indices for human body pose estimation are:
            0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
            5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
            9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
            13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = speed if speed is not None else {'preprocess': None, 'inference': None, 'postprocess': None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = ('boxes', 'masks', 'probs', 'keypoints', 'obb')

    def __getitem__(self, idx):
        """Return a Results object for a specific index of inference results.

        Args:
            idx (int | slice): Index or slice to retrieve from the Results object.

        Returns:
            (Results): A new Results object containing the specified subset of inference results.

        Examples:
            >>> results = model("path/to/image.jpg")  # Perform inference
            >>> single_result = results[0]  # Get the first result
            >>> subset_results = results[1:4]  # Get a slice of results
        """
        return self._apply('__getitem__', idx)

    def __len__(self) -> int:
        """Return the number of detections in the Results object.

        Returns:
            (int): The number of detections, determined by the length of the first non-empty attribute in (boxes, masks,
                probs, keypoints, or obb).

        Examples:
            >>> results = Results(orig_img, path, names, boxes=torch.rand(5, 6))
            >>> len(results)
            5
        """
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes: torch.Tensor | None=None, masks: torch.Tensor | None=None, probs: torch.Tensor | None=None, obb: torch.Tensor | None=None, keypoints: torch.Tensor | None=None):
        """Update the Results object with new detection data.

        This method allows updating the boxes, masks, keypoints, probabilities, and oriented bounding boxes (OBB) of
        the Results object. It ensures that boxes are clipped to the original image shape.

        Args:
            boxes (torch.Tensor | None): A tensor of shape (N, 6) containing bounding box coordinates and confidence
                scores. The format is (x1, y1, x2, y2, conf, class).
            masks (torch.Tensor | None): A tensor of shape (N, H, W) containing segmentation masks.
            probs (torch.Tensor | None): A tensor of shape (num_classes,) containing class probabilities.
            obb (torch.Tensor | None): A tensor of shape (N, 7) or (N, 8) containing oriented bounding box coordinates.
            keypoints (torch.Tensor | None): A tensor of shape (N, K, 3) containing keypoints, were K=17 for persons.

        Examples:
            >>> results = model("image.jpg")
            >>> new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])
            >>> results[0].update(boxes=new_boxes)
        """
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)
        if keypoints is not None:
            self.keypoints = Keypoints(keypoints, self.orig_shape)

    def _apply(self, fn: str, *args, **kwargs):
        """Apply a function to all non-empty attributes and return a new Results object with modified attributes.

        This method is internally called by methods like .to(), .cuda(), .cpu(), etc.

        Args:
            fn (str): The name of the function to apply.
            *args (Any): Variable length argument list to pass to the function.
            **kwargs (Any): Arbitrary keyword arguments to pass to the function.

        Returns:
            (Results): A new Results object with attributes modified by the applied function.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result_cuda = result.cuda()
            ...     result_cpu = result.cpu()
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """Return a copy of the Results object with all its tensors moved to CPU memory.

        This method creates a new Results object with all tensor attributes (boxes, masks, probs, keypoints, obb)
        transferred to CPU memory. It's useful for moving data from GPU to CPU for further processing or saving.

        Returns:
            (Results): A new Results object with all tensor attributes on CPU memory.

        Examples:
            >>> results = model("path/to/image.jpg")  # Perform inference
            >>> cpu_result = results[0].cpu()  # Move the first result to CPU
            >>> print(cpu_result.boxes.device)  # Output: cpu
        """
        return self._apply('cpu')

    def numpy(self):
        """Convert all tensors in the Results object to numpy arrays.

        Returns:
            (Results): A new Results object with all tensors converted to numpy arrays.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> numpy_result = results[0].numpy()
            >>> type(numpy_result.boxes.data)
            <class 'numpy.ndarray'>

        Notes:
            This method creates a new Results object, leaving the original unchanged. It's useful for
            interoperability with numpy-based libraries or when CPU-based operations are required.
        """
        return self._apply('numpy')

    def cuda(self):
        """Move all tensors in the Results object to GPU memory.

        Returns:
            (Results): A new Results object with all tensors moved to CUDA device.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> cuda_results = results[0].cuda()  # Move first result to GPU
            >>> for result in results:
            ...     result_cuda = result.cuda()  # Move each result to GPU
        """
        return self._apply('cuda')

    def to(self, *args, **kwargs):
        """Move all tensors in the Results object to the specified device and dtype.

        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().
            **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().

        Returns:
            (Results): A new Results object with all tensors moved to the specified device and dtype.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> result_cuda = results[0].to("cuda")  # Move first result to GPU
            >>> result_cpu = results[0].to("cpu")  # Move first result to CPU
            >>> result_half = results[0].to(dtype=torch.float16)  # Convert first result to half precision
        """
        return self._apply('to', *args, **kwargs)

    def new(self):
        """Create a new Results object with the same image, path, names, and speed attributes.

        Returns:
            (Results): A new Results object with copied attributes from the original instance.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(self, conf: bool=True, line_width: float | None=None, font_size: float | None=None, font: str='Arial.ttf', pil: bool=False, img: np.ndarray | None=None, im_gpu: torch.Tensor | None=None, kpt_radius: int=5, kpt_line: bool=True, labels: bool=True, boxes: bool=True, masks: bool=True, probs: bool=True, show: bool=False, save: bool=False, filename: str | None=None, color_mode: str='class', txt_color: tuple[int, int, int]=(255, 255, 255)) -> np.ndarray:
        """Plot detection results on an input BGR image.

        Args:
            conf (bool): Whether to plot detection confidence scores.
            line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
            font_size (float | None): Font size for text. If None, scaled to image size.
            font (str): Font to use for text.
            pil (bool): Whether to return the image as a PIL Image.
            img (np.ndarray | None): Image to plot on. If None, uses original image.
            im_gpu (torch.Tensor | None): Normalized image on GPU for faster mask plotting.
            kpt_radius (int): Radius of drawn keypoints.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot labels of bounding boxes.
            boxes (bool): Whether to plot bounding boxes.
            masks (bool): Whether to plot masks.
            probs (bool): Whether to plot classification probabilities.
            show (bool): Whether to display the annotated image.
            save (bool): Whether to save the annotated image.
            filename (str | None): Filename to save image if save is True.
            color_mode (str): Specify the color mode, e.g., 'instance' or 'class'.
            txt_color (tuple[int, int, int]): Text color in BGR format for classification output.

        Returns:
            (np.ndarray | PIL.Image.Image): Annotated image as a NumPy array (BGR) or PIL image (RGB) if `pil=True`.

        Examples:
            >>> results = model("image.jpg")
            >>> for result in results:
            ...     im = result.plot()
            ...     im.show()
        """
        assert color_mode in {'instance', 'class'}, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).byte().cpu().numpy()
        names = self.names
        is_obb = self.obb is not None
        (pred_boxes, show_boxes) = (self.obb if is_obb else self.boxes, boxes)
        (pred_masks, show_masks) = (self.masks, masks)
        (pred_probs, show_probs) = (self.probs, probs)
        annotator = Annotator(deepcopy(self.orig_img if img is None else img), line_width, font_size, font, pil or (pred_probs is not None and show_probs), example=names)
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device).permute(2, 0, 1).flip(0).contiguous() / 255
            idx = pred_boxes.id if pred_boxes.is_track and color_mode == 'instance' else pred_boxes.cls if pred_boxes and color_mode == 'class' else reversed(range(len(pred_masks)))
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)
        if pred_boxes is not None and show_boxes:
            for (i, d) in enumerate(reversed(pred_boxes)):
                (c, d_conf, id) = (int(d.cls), float(d.conf) if conf else None, int(d.id.item()) if d.is_track else None)
                name = ('' if id is None else f'id:{id} ') + names[c]
                label = (f'{name} {d_conf:.2f}' if conf else name) if labels else None
                box = d.xyxyxyxy.squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(box, label, color=colors(c if color_mode == 'class' else id if id is not None else i if color_mode == 'instance' else None, True))
        if pred_probs is not None and show_probs:
            text = '\n'.join((f'{(names[j] if names else j)} {pred_probs.data[j]:.2f}' for j in pred_probs.top5))
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=txt_color, box_color=(64, 64, 64, 128))
        if self.keypoints is not None:
            for (i, k) in enumerate(reversed(self.keypoints.data)):
                annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line, kpt_color=colors(i, True) if color_mode == 'instance' else None)
        if show:
            annotator.show(self.path)
        if save:
            annotator.save(filename or f'results_{Path(self.path).name}')
        return annotator.result(pil)

    def show(self, *args, **kwargs):
        """Display the image with annotated inference results.

        This method plots the detection results on the original image and displays it. It's a convenient way to
        visualize the model's predictions directly.

        Args:
            *args (Any): Variable length argument list to be passed to the `plot()` method.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot()` method.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> results[0].show()  # Display the first result
            >>> for result in results:
            ...     result.show()  # Display all results
        """
        self.plot(*args, show=True, **kwargs)

    def save(self, filename: str | None=None, *args, **kwargs) -> str:
        """Save annotated inference results image to file.

        This method plots the detection results on the original image and saves the annotated image to a file. It
        utilizes the `plot` method to generate the annotated image and then saves it to the specified filename.

        Args:
            filename (str | None): The filename to save the annotated image. If None, a default filename is generated
                based on the original image path.
            *args (Any): Variable length argument list to be passed to the `plot` method.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot` method.

        Returns:
            (str): The filename where the image was saved.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save("annotated_image.jpg")
            >>> # Or with custom plot arguments
            >>> for result in results:
            ...     result.save("annotated_image.jpg", conf=False, line_width=2)
            >>> # Directory will be created automatically if it does not exist
            >>> result.save("path/to/annotated_image.jpg")
        """
        if not filename:
            filename = f'results_{Path(self.path).name}'
        Path(filename).absolute().parent.mkdir(parents=True, exist_ok=True)
        self.plot(*args, save=True, filename=filename, **kwargs)
        return filename

    def verbose(self) -> str:
        """Return a log string for each task in the results, detailing detection and classification outcomes.

        This method generates a human-readable string summarizing the detection and classification results. It includes
        the number of detections for each class and the top probabilities for classification tasks.

        Returns:
            (str): A formatted string containing a summary of the results. For detection tasks, it includes the number
                of detections per class. For classification tasks, it includes the top 5 class probabilities.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     print(result.verbose())
            2 persons, 1 car, 3 traffic lights,
            dog 0.92, cat 0.78, horse 0.64,

        Notes:
            - If there are no detections, the method returns "(no detections), " for detection tasks.
            - For classification tasks, it returns the top 5 class probabilities and their corresponding class names.
            - The returned string is comma-separated and ends with a comma and a space.
        """
        boxes = self.obb if self.obb is not None else self.boxes
        if len(self) == 0:
            return '' if self.probs is not None else '(no detections), '
        if self.probs is not None:
            return f"{', '.join((f'{self.names[j]} {self.probs.data[j]:.2f}' for j in self.probs.top5))}, "
        if boxes:
            counts = boxes.cls.int().bincount()
            return ''.join((f"{n} {self.names[i]}{'s' * (n > 1)}, " for (i, n) in enumerate(counts) if n > 0))

    def save_txt(self, txt_file: str | Path, save_conf: bool=False) -> str:
        """Save detection results to a text file.

        Args:
            txt_file (str | Path): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        Returns:
            (str): Path to the saved text file.

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo26n.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_txt("output.txt")

        Notes:
            - The file will contain one line per detection or classification with the following structure:
              - For detections: `class x_center y_center width height [confidence] [track_id]`
              - For classifications: `confidence class_name`
              - For masks and keypoints, the specific formats will vary accordingly.
            - The function will create the output directory if it does not exist.
            - If save_conf is False, the confidence scores will be excluded from the output.
            - Existing contents of the file will not be overwritten; new results will be appended.
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            [texts.append(f'{probs.data[j]:.2f} {self.names[j]}') for j in probs.top5]
        elif boxes:
            for (j, d) in enumerate(boxes):
                (c, conf, id) = (int(d.cls), float(d.conf), int(d.id.item()) if d.is_track else None)
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(('%g ' * len(line)).rstrip() % line)
        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
            with open(txt_file, 'a', encoding='utf-8') as f:
                f.writelines((text + '\n' for text in texts))
        return str(txt_file)

    def save_crop(self, save_dir: str | Path, file_name: str | Path=Path('im.jpg')):
        """Save cropped detection images to specified directory.

        This method saves cropped images of detected objects to a specified directory. Each crop is saved in a
        subdirectory named after the object's class, with the filename based on the input file_name.

        Args:
            save_dir (str | Path): Directory path where cropped images will be saved.
            file_name (str | Path): Base filename for the saved cropped images.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_crop(save_dir="path/to/crops", file_name="detection")

        Notes:
            - This method does not support Classify or Oriented Bounding Box (OBB) tasks.
            - Crops are saved as 'save_dir/class_name/file_name.jpg'.
            - The method will create necessary subdirectories if they don't exist.
            - Original image is copied before cropping to avoid modifying the original.
        """
        if self.probs is not None:
            LOGGER.warning('Classify task does not support `save_crop`.')
            return
        if self.obb is not None:
            LOGGER.warning('OBB task does not support `save_crop`.')
            return
        for d in self.boxes:
            save_one_box(d.xyxy, self.orig_img.copy(), file=Path(save_dir) / self.names[int(d.cls)] / Path(file_name).with_suffix('.jpg'), BGR=True)

    def summary(self, normalize: bool=False, decimals: int=5) -> list[dict[str, Any]]:
        """Convert inference results to a summarized dictionary with optional normalization for box coordinates.

        This method creates a list of detection dictionaries, each containing information about a single detection or
        classification result. For classification tasks, it returns the top 5 classes and their
        confidences. For detection tasks, it includes class information, bounding box coordinates, and
        optionally mask segments and keypoints.

        Args:
            normalize (bool): Whether to normalize bounding box coordinates by image dimensions.
            decimals (int): Number of decimal places to round the output values to.

        Returns:
            (list[dict[str, Any]]): A list of dictionaries, each containing summarized information for a single
                detection or classification result. The structure of each dictionary varies based on the task type
                (classification or detection) and available information (boxes, masks, keypoints).

        Examples:
            >>> results = model("image.jpg")
            >>> for result in results:
            ...     summary = result.summary()
            ...     print(summary)
        """
        results = []
        if self.probs is not None:
            for (class_id, conf) in zip(self.probs.top5, self.probs.top5conf.tolist()):
                class_id = int(class_id)
                results.append({'name': self.names[class_id], 'class': class_id, 'confidence': round(conf, decimals)})
            return results
        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        (h, w) = self.orig_shape if normalize else (1, 1)
        for (i, row) in enumerate(data):
            (class_id, conf) = (int(row.cls), round(row.conf.item(), decimals))
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            xy = {}
            for (j, b) in enumerate(box):
                xy[f'x{j + 1}'] = round(b[0] / w, decimals)
                xy[f'y{j + 1}'] = round(b[1] / h, decimals)
            result = {'name': self.names[class_id], 'class': class_id, 'confidence': conf, 'box': xy}
            if data.is_track:
                result['track_id'] = int(row.id.item())
            if self.masks:
                result['segments'] = {'x': (self.masks.xy[i][:, 0] / w).round(decimals).tolist(), 'y': (self.masks.xy[i][:, 1] / h).round(decimals).tolist()}
            if self.keypoints is not None:
                kpt = self.keypoints[i]
                if kpt.has_visible:
                    (x, y, visible) = kpt.data[0].cpu().unbind(dim=1)
                else:
                    (x, y) = kpt.data[0].cpu().unbind(dim=1)
                result['keypoints'] = {'x': (x / w).numpy().round(decimals).tolist(), 'y': (y / h).numpy().round(decimals).tolist()}
                if kpt.has_visible:
                    result['keypoints']['visible'] = visible.numpy().round(decimals).tolist()
            results.append(result)
        return results

class Boxes(BaseTensor):
    """A class for managing and manipulating detection boxes.

    This class provides comprehensive functionality for handling detection boxes, including their coordinates,
    confidence scores, class labels, and optional tracking IDs. It supports various box formats and offers methods for
    easy manipulation and conversion between different coordinate systems.

    Attributes:
        data (torch.Tensor | np.ndarray): The raw tensor containing detection boxes and associated data.
        orig_shape (tuple[int, int]): The original image dimensions (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xyxy (torch.Tensor | np.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | np.ndarray): Confidence scores for each box.
        cls (torch.Tensor | np.ndarray): Class labels for each box.
        id (torch.Tensor | None): Tracking IDs for each box (if available).
        xywh (torch.Tensor | np.ndarray): Boxes in [x, y, width, height] format.
        xyxyn (torch.Tensor | np.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
        xywhn (torch.Tensor | np.ndarray): Normalized [x, y, width, height] boxes relative to orig_shape.

    Methods:
        cpu: Return a copy of the object with all tensors on CPU memory.
        numpy: Return a copy of the object with all tensors as numpy arrays.
        cuda: Return a copy of the object with all tensors on GPU memory.
        to: Return a copy of the object with tensors on specified device and dtype.

    Examples:
        >>> import torch
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # height, width
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
    """

    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """Initialize the Boxes class with detection box data and the original image shape.

        This class manages detection boxes, providing easy access and manipulation of box coordinates, confidence
        scores, class identifiers, and optional tracking IDs. It supports multiple formats for box coordinates,
        including both absolute and normalized forms.

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape (num_boxes, 6) or
                (num_boxes, 7). Columns should contain [x1, y1, x2, y2, (optional) track_id, confidence, class].
            orig_shape (tuple[int, int]): The original image shape as (height, width). Used for normalization.
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f'expected 6 or 7 values but got {n}'
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self) -> torch.Tensor | np.ndarray:
        """Return bounding boxes in [x1, y1, x2, y2] format.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or numpy array of shape (n, 4) containing bounding box coordinates in
                [x1, y1, x2, y2] format, where n is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> xyxy = boxes.xyxy
            >>> print(xyxy)
        """
        return self.data[:, :4]

    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        """Return the confidence scores for each detection box.

        Returns:
            (torch.Tensor | np.ndarray): A 1D tensor or array containing confidence scores for each detection, with
                shape (N,) where N is the number of detections.

        Examples:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
            >>> conf_scores = boxes.conf
            >>> print(conf_scores)
            tensor([0.9000])
        """
        return self.data[:, -2]

    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        """Return the class ID tensor representing category predictions for each bounding box.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or numpy array containing the class IDs for each detection box. The
                shape is (N,), where N is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> class_ids = boxes.cls
            >>> print(class_ids)  # tensor([0., 2., 1.])
        """
        return self.data[:, -1]

    @property
    def id(self) -> torch.Tensor | np.ndarray | None:
        """Return the tracking IDs for each detection box if available.

        Returns:
            (torch.Tensor | np.ndarray | None): A tensor or array containing tracking IDs for each box if tracking is
                enabled, otherwise None. Shape is (N,) where N is the number of boxes.

        Examples:
            >>> results = model.track("path/to/video.mp4")
            >>> for result in results:
            ...     boxes = result.boxes
            ...     if boxes.is_track:
            ...         track_ids = boxes.id
            ...         print(f"Tracking IDs: {track_ids}")
            ...     else:
            ...         print("Tracking is not enabled for these boxes.")

        Notes:
            - This property is only available when tracking is enabled (i.e., when `is_track` is True).
            - The tracking IDs are typically used to associate detections across multiple frames in video analysis.
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xywh(self) -> torch.Tensor | np.ndarray:
        """Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.

        Returns:
            (torch.Tensor | np.ndarray): Boxes in [x_center, y_center, width, height] format, where x_center, y_center
                are the coordinates of the center point of the bounding box, width, height are the dimensions of the
                bounding box and the shape of the returned tensor is (N, 4), where N is the number of boxes.

        Examples:
            >>> boxes = Boxes(
            ...     torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]]), orig_shape=(480, 640)
            ... )
            >>> xywh = boxes.xywh
            >>> print(xywh)
            tensor([[125.0000,  75.0000,  50.0000,  50.0000],
                    [250.0000, 200.0000, 100.0000, 100.0000]])
        """
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self) -> torch.Tensor | np.ndarray:
        """Return normalized bounding box coordinates relative to the original image size.

        This property calculates and returns the bounding box coordinates in [x1, y1, x2, y2] format, normalized to the
        range [0, 1] based on the original image dimensions.

        Returns:
            (torch.Tensor | np.ndarray): Normalized bounding box coordinates with shape (N, 4), where N is the number of
                boxes. Each row contains [x1, y1, x2, y2] values normalized to [0, 1].

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xyxyn
            >>> print(normalized)
            tensor([[0.1562, 0.1042, 0.4688, 0.8333]])
        """
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self) -> torch.Tensor | np.ndarray:
        """Return normalized bounding boxes in [x, y, width, height] format.

        This property calculates and returns the normalized bounding box coordinates in the format [x_center, y_center,
        width, height], where all values are relative to the original image dimensions.

        Returns:
            (torch.Tensor | np.ndarray): Normalized bounding boxes with shape (N, 4), where N is the number of boxes.
                Each row contains [x_center, y_center, width, height] values normalized to [0, 1] based on the original
                image dimensions.

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xywhn
            >>> print(normalized)
            tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
        """
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh
