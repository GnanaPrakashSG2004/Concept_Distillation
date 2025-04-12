"""
Preparing the imagenet dataset - Reference: https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh

- Downloading the dataset
    - If you are on ada, first run `mkdir -p /scratch/$USER/ && cd /scratch/$USER/`
    - Then run, `scp $USER@ada:/share1/dataset/Imagenet2012/Imagenet-orig.tar .`
    - If not, just replacing $USER with ada username should work just fine, provided you are on college wifi.


- Downloading metadata needed by torchvision datasets class
    `wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz` (2.45 MB)


- Dataset preparation
    - Then run `tar -xvf Imagenet-orig.tar` to extract the dataset.
    - This will generate the following files:
        - ILSVRC2012_img_train.tar (about 138 GB)
        - ILSVRC2012_img_val.tar (about 6 GB)
        - README.md (276 bytes)


- Directory structure setting
    - Then, `cd Imagenet-orig/ && mkdir -p val`


- Val set extraction
    `mv ILSVRC2012_img_val.tar val/ && cd val/ && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar`


- Preparing a list of 50k images into imagenet structure
    `wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash`


- Sanity check
    `find -name *.JPEG | wc -l` should return 50000


----------------------------------------------------------------------------------------------------------------------------------------


- Train set extraction - Not done yet
    - Go back a directory level and prepare the directory structure
    `cd .. && mkdir -p train && cd train/`


- Train set extraction
    `mv ILSVRC2012_img_train.tar train/ && cd train/ && tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar`


- We will now have 1000 tar files, one for each category
- For each .tar file: 
    - create directory with same name as .tar file
    - extract and copy contents of .tar file into directory
    - remove .tar file
    `find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done`


- Sanity check
    `find -name *.JPEG | wc -l` should return 1281167

"""


import os
from pprint import pprint
from collections import defaultdict

import argparse
import numpy as np
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as torch_datasets

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

username = os.getenv("USER")
username = "gp" if username == "punnavajhala.prakash" else username


def parse_args():
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(description="Restructure ImageNet Dataset")

    arg_parser.add_argument(
        "--model_name",
        type=str,
        default="nf_resnet50.ra2_in1k",
        help="Name of the model to use for restructuring"
    )
    arg_parser.add_argument(
        "--imagenet_path",
        type=str,
        default=f"/scratch/{username}/Imagenet-orig/",
        help="Path to the ImageNet dataset"
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for DataLoader"
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/",
        help="Directory to save the restructured dataset mapping"
    )
    arg_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    return arg_parser.parse_args()


def restructure_imagenet_split(
        args: argparse.Namespace,
        split: str,
        model: torch.nn.Module,
        transform: callable,
        accelerator: Accelerator
    ) -> None:
    """
    Restructure the ImageNet dataset for a given split (train/val).

    Args:
        args: Command line arguments
        split: Split of the dataset to restructure (train/val)
        model: Pre-trained model to use for restructuring
        transform: Transformations to apply to the images
        accelerator: Accelerator object for distributed inference
    """
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, f"restructured_map_{split}.npy")

    if not os.path.exists(output_file_path):
        # to keep track of actual image indices when using accelerator
        # cuz batch index will be local to each individual gpu and not global
        # which means that we can't rely on the batch index to get the actual image index
        # which we need for accurate mapping to the predicted classes for later on
        class IndexedImageNet(torch_datasets.ImageNet):
            def __getitem__(self, index):
                img, label = super().__getitem__(index)
                return img, label, index

        dataset = IndexedImageNet(
            root=args.imagenet_path,
            split=split,
            transform=transform,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        data_loader, model = accelerator.prepare(data_loader, model)

        class_to_indices = defaultdict(list)
        with torch.inference_mode():
            pbar = tqdm(
                data_loader,
                desc=f"Restructuring ImageNet {split.capitalize()} Set",
                disable=not accelerator.is_local_main_process,
            )

            for images, _, indices in pbar:
                logits = model(images)
                probs = torch.softmax(logits, dim=-1)
                predicted_classes = probs.argmax(dim=-1).cpu().numpy()

                for i, pred_class in enumerate(predicted_classes):
                    class_to_indices[pred_class].append(indices[i].item())

        gathered_class_to_indices = defaultdict(list)
        for class_idx in sorted(class_to_indices.keys()):
            indices_tensor = torch.tensor(class_to_indices[class_idx], device=accelerator.device)

            gathered_indices = accelerator.gather(indices_tensor)
            gathered_class_to_indices[class_idx] = gathered_indices.cpu().numpy()

        class_to_indices = dict(sorted(gathered_class_to_indices.items()))
        for index, indices in class_to_indices.items():
            accelerator.print(f"Class {index} -> {len(indices)} images")

        if accelerator.is_main_process:
            np.save(output_file_path, class_to_indices)

        accelerator.wait_for_everyone()

    else:
        accelerator.print(f"Restructured map for {split} set already exists")

    accelerator.print()
    accelerator.print("=" * 50)
    accelerator.print()


if __name__ == "__main__":
    restructuring_args = parse_args()

    accelerator = Accelerator(split_batches=False)
    set_seed(restructuring_args.seed, deterministic=True)

    if accelerator.is_main_process:
        accelerator.print()
        accelerator.print("Run configuration:")
        pprint(vars(restructuring_args))
        accelerator.print()

        accelerator.print("Accelerator configuration:")
        accelerator.print(accelerator.state)

    model = timm.create_model(
        restructuring_args.model_name,
        pretrained=True,
        cache_dir=f"/scratch/{username}/timm_cache/"
    ).eval().requires_grad_(False).to(accelerator.device)

    config = resolve_data_config({}, model=model)
    imagenet_transform = create_transform(**config)

    restructure_imagenet_split(restructuring_args, "val",   model, imagenet_transform, accelerator)
    restructure_imagenet_split(restructuring_args, "train", model, imagenet_transform, accelerator)


"""
- Command to run the file with accelerate:
    `accelerate launch restructure_config.yml restructure_imagenet.py`
"""


"""
- Compute requirements
    - Batch size of 64 needs roughly 2GB VRAM
    - Val restructuring got done in ~3 minutes
    - Train will take roughly 75-80 minutes

    - Batch size of 128 needs roughly 3.3GB VRAM
    - Took roughly same time
    - Would've been faster if I had more CPU cores (would've set `num_workers` to a higher value)
    - Currently operating with 12 cores

    - Successfully set up accelerate, but will have to test it to confirm it works correctly
    - We'll get insane speedups if this actually works
"""


"""
- Output for val set

Class 0 -> 48 images
Class 1 -> 46 images
Class 2 -> 53 images
Class 3 -> 50 images
Class 4 -> 52 images
Class 5 -> 44 images
Class 6 -> 57 images
Class 7 -> 43 images
Class 8 -> 59 images
Class 9 -> 51 images
Class 10 -> 49 images
Class 11 -> 55 images
Class 12 -> 52 images
Class 13 -> 51 images
Class 14 -> 48 images
Class 15 -> 52 images
Class 16 -> 50 images
Class 17 -> 48 images
Class 18 -> 51 images
Class 19 -> 50 images
Class 20 -> 44 images
Class 21 -> 44 images
Class 22 -> 53 images
Class 23 -> 58 images
Class 24 -> 48 images
Class 25 -> 54 images
Class 26 -> 49 images
Class 27 -> 43 images
Class 28 -> 55 images
Class 29 -> 53 images
Class 30 -> 50 images
Class 31 -> 55 images
Class 32 -> 30 images
Class 33 -> 55 images
Class 34 -> 41 images
Class 35 -> 56 images
Class 36 -> 41 images
Class 37 -> 57 images
Class 38 -> 48 images
Class 39 -> 60 images
Class 40 -> 56 images
Class 41 -> 54 images
Class 42 -> 48 images
Class 43 -> 46 images
Class 44 -> 47 images
Class 45 -> 46 images
Class 46 -> 48 images
Class 47 -> 47 images
Class 48 -> 53 images
Class 49 -> 48 images
Class 50 -> 54 images
Class 51 -> 50 images
Class 52 -> 52 images
Class 53 -> 57 images
Class 54 -> 48 images
Class 55 -> 50 images
Class 56 -> 51 images
Class 57 -> 49 images
Class 58 -> 62 images
Class 59 -> 42 images
Class 60 -> 42 images
Class 61 -> 54 images
Class 62 -> 41 images
Class 63 -> 53 images
Class 64 -> 52 images
Class 65 -> 44 images
Class 66 -> 61 images
Class 67 -> 54 images
Class 68 -> 25 images
Class 69 -> 50 images
Class 70 -> 55 images
Class 71 -> 59 images
Class 72 -> 67 images
Class 73 -> 31 images
Class 74 -> 56 images
Class 75 -> 49 images
Class 76 -> 47 images
Class 77 -> 56 images
Class 78 -> 44 images
Class 79 -> 46 images
Class 80 -> 47 images
Class 81 -> 47 images
Class 82 -> 59 images
Class 83 -> 47 images
Class 84 -> 49 images
Class 85 -> 51 images
Class 86 -> 47 images
Class 87 -> 49 images
Class 88 -> 52 images
Class 89 -> 51 images
Class 90 -> 50 images
Class 91 -> 47 images
Class 92 -> 49 images
Class 93 -> 53 images
Class 94 -> 53 images
Class 95 -> 50 images
Class 96 -> 48 images
Class 97 -> 51 images
Class 98 -> 46 images
Class 99 -> 56 images
Class 100 -> 50 images
Class 101 -> 56 images
Class 102 -> 53 images
Class 103 -> 48 images
Class 104 -> 51 images
Class 105 -> 48 images
Class 106 -> 42 images
Class 107 -> 49 images
Class 108 -> 38 images
Class 109 -> 56 images
Class 110 -> 46 images
Class 111 -> 48 images
Class 112 -> 47 images
Class 113 -> 49 images
Class 114 -> 52 images
Class 115 -> 57 images
Class 116 -> 48 images
Class 117 -> 46 images
Class 118 -> 52 images
Class 119 -> 57 images
Class 120 -> 40 images
Class 121 -> 45 images
Class 122 -> 47 images
Class 123 -> 51 images
Class 124 -> 50 images
Class 125 -> 56 images
Class 126 -> 40 images
Class 127 -> 46 images
Class 128 -> 47 images
Class 129 -> 50 images
Class 130 -> 50 images
Class 131 -> 55 images
Class 132 -> 54 images
Class 133 -> 49 images
Class 134 -> 45 images
Class 135 -> 52 images
Class 136 -> 51 images
Class 137 -> 51 images
Class 138 -> 52 images
Class 139 -> 56 images
Class 140 -> 48 images
Class 141 -> 46 images
Class 142 -> 50 images
Class 143 -> 52 images
Class 144 -> 52 images
Class 145 -> 49 images
Class 146 -> 53 images
Class 147 -> 49 images
Class 148 -> 50 images
Class 149 -> 47 images
Class 150 -> 52 images
Class 151 -> 61 images
Class 152 -> 43 images
Class 153 -> 56 images
Class 154 -> 54 images
Class 155 -> 55 images
Class 156 -> 53 images
Class 157 -> 53 images
Class 158 -> 33 images
Class 159 -> 54 images
Class 160 -> 48 images
Class 161 -> 54 images
Class 162 -> 70 images
Class 163 -> 30 images
Class 164 -> 51 images
Class 165 -> 37 images
Class 166 -> 62 images
Class 167 -> 19 images
Class 168 -> 53 images
Class 169 -> 55 images
Class 170 -> 48 images
Class 171 -> 53 images
Class 172 -> 57 images
Class 173 -> 43 images
Class 174 -> 50 images
Class 175 -> 40 images
Class 176 -> 52 images
Class 177 -> 49 images
Class 178 -> 54 images
Class 179 -> 47 images
Class 180 -> 59 images
Class 181 -> 48 images
Class 182 -> 58 images
Class 183 -> 51 images
Class 184 -> 51 images
Class 185 -> 49 images
Class 186 -> 52 images
Class 187 -> 51 images
Class 188 -> 43 images
Class 189 -> 52 images
Class 190 -> 45 images
Class 191 -> 47 images
Class 192 -> 55 images
Class 193 -> 43 images
Class 194 -> 51 images
Class 195 -> 57 images
Class 196 -> 45 images
Class 197 -> 58 images
Class 198 -> 51 images
Class 199 -> 47 images
Class 200 -> 55 images
Class 201 -> 46 images
Class 202 -> 54 images
Class 203 -> 58 images
Class 204 -> 50 images
Class 205 -> 57 images
Class 206 -> 49 images
Class 207 -> 61 images
Class 208 -> 60 images
Class 209 -> 49 images
Class 210 -> 46 images
Class 211 -> 46 images
Class 212 -> 47 images
Class 213 -> 44 images
Class 214 -> 49 images
Class 215 -> 50 images
Class 216 -> 50 images
Class 217 -> 63 images
Class 218 -> 57 images
Class 219 -> 58 images
Class 220 -> 38 images
Class 221 -> 46 images
Class 222 -> 40 images
Class 223 -> 61 images
Class 224 -> 47 images
Class 225 -> 53 images
Class 226 -> 34 images
Class 227 -> 48 images
Class 228 -> 48 images
Class 229 -> 51 images
Class 230 -> 51 images
Class 231 -> 42 images
Class 232 -> 62 images
Class 233 -> 43 images
Class 234 -> 61 images
Class 235 -> 50 images
Class 236 -> 54 images
Class 237 -> 52 images
Class 238 -> 68 images
Class 239 -> 57 images
Class 240 -> 30 images
Class 241 -> 57 images
Class 242 -> 48 images
Class 243 -> 55 images
Class 244 -> 47 images
Class 245 -> 51 images
Class 246 -> 44 images
Class 247 -> 50 images
Class 248 -> 62 images
Class 249 -> 58 images
Class 250 -> 39 images
Class 251 -> 53 images
Class 252 -> 46 images
Class 253 -> 52 images
Class 254 -> 54 images
Class 255 -> 51 images
Class 256 -> 54 images
Class 257 -> 56 images
Class 258 -> 52 images
Class 259 -> 54 images
Class 260 -> 48 images
Class 261 -> 51 images
Class 262 -> 50 images
Class 263 -> 61 images
Class 264 -> 47 images
Class 265 -> 50 images
Class 266 -> 49 images
Class 267 -> 59 images
Class 268 -> 49 images
Class 269 -> 58 images
Class 270 -> 51 images
Class 271 -> 37 images
Class 272 -> 46 images
Class 273 -> 48 images
Class 274 -> 49 images
Class 275 -> 54 images
Class 276 -> 48 images
Class 277 -> 46 images
Class 278 -> 53 images
Class 279 -> 53 images
Class 280 -> 52 images
Class 281 -> 65 images
Class 282 -> 27 images
Class 283 -> 53 images
Class 284 -> 58 images
Class 285 -> 45 images
Class 286 -> 48 images
Class 287 -> 43 images
Class 288 -> 51 images
Class 289 -> 48 images
Class 290 -> 50 images
Class 291 -> 56 images
Class 292 -> 63 images
Class 293 -> 52 images
Class 294 -> 56 images
Class 295 -> 61 images
Class 296 -> 51 images
Class 297 -> 39 images
Class 298 -> 49 images
Class 299 -> 52 images
Class 300 -> 57 images
Class 301 -> 61 images
Class 302 -> 38 images
Class 303 -> 43 images
Class 304 -> 42 images
Class 305 -> 54 images
Class 306 -> 48 images
Class 307 -> 50 images
Class 308 -> 55 images
Class 309 -> 49 images
Class 310 -> 57 images
Class 311 -> 55 images
Class 312 -> 48 images
Class 313 -> 52 images
Class 314 -> 43 images
Class 315 -> 50 images
Class 316 -> 53 images
Class 317 -> 50 images
Class 318 -> 44 images
Class 319 -> 53 images
Class 320 -> 51 images
Class 321 -> 52 images
Class 322 -> 51 images
Class 323 -> 48 images
Class 324 -> 52 images
Class 325 -> 49 images
Class 326 -> 48 images
Class 327 -> 52 images
Class 328 -> 50 images
Class 329 -> 44 images
Class 330 -> 46 images
Class 331 -> 51 images
Class 332 -> 51 images
Class 333 -> 54 images
Class 334 -> 50 images
Class 335 -> 51 images
Class 336 -> 58 images
Class 337 -> 46 images
Class 338 -> 47 images
Class 339 -> 50 images
Class 340 -> 48 images
Class 341 -> 42 images
Class 342 -> 56 images
Class 343 -> 53 images
Class 344 -> 50 images
Class 345 -> 36 images
Class 346 -> 52 images
Class 347 -> 51 images
Class 348 -> 49 images
Class 349 -> 55 images
Class 350 -> 51 images
Class 351 -> 49 images
Class 352 -> 58 images
Class 353 -> 42 images
Class 354 -> 52 images
Class 355 -> 49 images
Class 356 -> 33 images
Class 357 -> 55 images
Class 358 -> 44 images
Class 359 -> 57 images
Class 360 -> 52 images
Class 361 -> 51 images
Class 362 -> 51 images
Class 363 -> 49 images
Class 364 -> 54 images
Class 365 -> 47 images
Class 366 -> 45 images
Class 367 -> 57 images
Class 368 -> 53 images
Class 369 -> 43 images
Class 370 -> 49 images
Class 371 -> 44 images
Class 372 -> 56 images
Class 373 -> 53 images
Class 374 -> 41 images
Class 375 -> 48 images
Class 376 -> 54 images
Class 377 -> 42 images
Class 378 -> 55 images
Class 379 -> 55 images
Class 380 -> 52 images
Class 381 -> 31 images
Class 382 -> 73 images
Class 383 -> 41 images
Class 384 -> 58 images
Class 385 -> 48 images
Class 386 -> 44 images
Class 387 -> 51 images
Class 388 -> 49 images
Class 389 -> 48 images
Class 390 -> 39 images
Class 391 -> 54 images
Class 392 -> 51 images
Class 393 -> 52 images
Class 394 -> 49 images
Class 395 -> 49 images
Class 396 -> 49 images
Class 397 -> 48 images
Class 398 -> 46 images
Class 399 -> 55 images
Class 400 -> 37 images
Class 401 -> 54 images
Class 402 -> 45 images
Class 403 -> 51 images
Class 404 -> 59 images
Class 405 -> 46 images
Class 406 -> 58 images
Class 407 -> 56 images
Class 408 -> 42 images
Class 409 -> 60 images
Class 410 -> 52 images
Class 411 -> 58 images
Class 412 -> 54 images
Class 413 -> 35 images
Class 414 -> 38 images
Class 415 -> 53 images
Class 416 -> 56 images
Class 417 -> 52 images
Class 418 -> 44 images
Class 419 -> 50 images
Class 420 -> 50 images
Class 421 -> 59 images
Class 422 -> 38 images
Class 423 -> 44 images
Class 424 -> 67 images
Class 425 -> 52 images
Class 426 -> 49 images
Class 427 -> 46 images
Class 428 -> 51 images
Class 429 -> 48 images
Class 430 -> 50 images
Class 431 -> 54 images
Class 432 -> 55 images
Class 433 -> 58 images
Class 434 -> 42 images
Class 435 -> 53 images
Class 436 -> 65 images
Class 437 -> 45 images
Class 438 -> 35 images
Class 439 -> 52 images
Class 440 -> 54 images
Class 441 -> 58 images
Class 442 -> 48 images
Class 443 -> 43 images
Class 444 -> 55 images
Class 445 -> 57 images
Class 446 -> 47 images
Class 447 -> 38 images
Class 448 -> 54 images
Class 449 -> 51 images
Class 450 -> 52 images
Class 451 -> 47 images
Class 452 -> 52 images
Class 453 -> 61 images
Class 454 -> 58 images
Class 455 -> 54 images
Class 456 -> 49 images
Class 457 -> 56 images
Class 458 -> 55 images
Class 459 -> 42 images
Class 460 -> 61 images
Class 461 -> 39 images
Class 462 -> 48 images
Class 463 -> 49 images
Class 464 -> 41 images
Class 465 -> 38 images
Class 466 -> 52 images
Class 467 -> 56 images
Class 468 -> 47 images
Class 469 -> 42 images
Class 470 -> 51 images
Class 471 -> 52 images
Class 472 -> 64 images
Class 473 -> 48 images
Class 474 -> 56 images
Class 475 -> 50 images
Class 476 -> 52 images
Class 477 -> 65 images
Class 478 -> 47 images
Class 479 -> 33 images
Class 480 -> 46 images
Class 481 -> 43 images
Class 482 -> 31 images
Class 483 -> 56 images
Class 484 -> 48 images
Class 485 -> 46 images
Class 486 -> 50 images
Class 487 -> 64 images
Class 488 -> 35 images
Class 489 -> 56 images
Class 490 -> 44 images
Class 491 -> 56 images
Class 492 -> 51 images
Class 493 -> 21 images
Class 494 -> 53 images
Class 495 -> 55 images
Class 496 -> 55 images
Class 497 -> 58 images
Class 498 -> 54 images
Class 499 -> 31 images
Class 500 -> 52 images
Class 501 -> 31 images
Class 502 -> 50 images
Class 503 -> 41 images
Class 504 -> 56 images
Class 505 -> 62 images
Class 506 -> 52 images
Class 507 -> 42 images
Class 508 -> 71 images
Class 509 -> 43 images
Class 510 -> 55 images
Class 511 -> 55 images
Class 512 -> 53 images
Class 513 -> 60 images
Class 514 -> 48 images
Class 515 -> 56 images
Class 516 -> 35 images
Class 517 -> 50 images
Class 518 -> 62 images
Class 519 -> 43 images
Class 520 -> 53 images
Class 521 -> 46 images
Class 522 -> 53 images
Class 523 -> 46 images
Class 524 -> 47 images
Class 525 -> 48 images
Class 526 -> 59 images
Class 527 -> 74 images
Class 528 -> 52 images
Class 529 -> 54 images
Class 530 -> 49 images
Class 531 -> 43 images
Class 532 -> 56 images
Class 533 -> 51 images
Class 534 -> 40 images
Class 535 -> 52 images
Class 536 -> 48 images
Class 537 -> 47 images
Class 538 -> 52 images
Class 539 -> 52 images
Class 540 -> 47 images
Class 541 -> 42 images
Class 542 -> 52 images
Class 543 -> 62 images
Class 544 -> 48 images
Class 545 -> 53 images
Class 546 -> 48 images
Class 547 -> 51 images
Class 548 -> 60 images
Class 549 -> 42 images
Class 550 -> 44 images
Class 551 -> 52 images
Class 552 -> 49 images
Class 553 -> 59 images
Class 554 -> 52 images
Class 555 -> 49 images
Class 556 -> 31 images
Class 557 -> 47 images
Class 558 -> 53 images
Class 559 -> 52 images
Class 560 -> 53 images
Class 561 -> 51 images
Class 562 -> 62 images
Class 563 -> 56 images
Class 564 -> 47 images
Class 565 -> 50 images
Class 566 -> 43 images
Class 567 -> 41 images
Class 568 -> 52 images
Class 569 -> 57 images
Class 570 -> 67 images
Class 571 -> 50 images
Class 572 -> 47 images
Class 573 -> 51 images
Class 574 -> 47 images
Class 575 -> 47 images
Class 576 -> 52 images
Class 577 -> 43 images
Class 578 -> 54 images
Class 579 -> 49 images
Class 580 -> 55 images
Class 581 -> 53 images
Class 582 -> 55 images
Class 583 -> 58 images
Class 584 -> 38 images
Class 585 -> 45 images
Class 586 -> 51 images
Class 587 -> 47 images
Class 588 -> 45 images
Class 589 -> 47 images
Class 590 -> 31 images
Class 591 -> 53 images
Class 592 -> 56 images
Class 593 -> 55 images
Class 594 -> 50 images
Class 595 -> 58 images
Class 596 -> 46 images
Class 597 -> 51 images
Class 598 -> 51 images
Class 599 -> 54 images
Class 600 -> 31 images
Class 601 -> 48 images
Class 602 -> 55 images
Class 603 -> 58 images
Class 604 -> 49 images
Class 605 -> 53 images
Class 606 -> 51 images
Class 607 -> 49 images
Class 608 -> 61 images
Class 609 -> 51 images
Class 610 -> 54 images
Class 611 -> 53 images
Class 612 -> 54 images
Class 613 -> 55 images
Class 614 -> 56 images
Class 615 -> 49 images
Class 616 -> 59 images
Class 617 -> 63 images
Class 618 -> 47 images
Class 619 -> 44 images
Class 620 -> 43 images
Class 621 -> 54 images
Class 622 -> 38 images
Class 623 -> 22 images
Class 624 -> 45 images
Class 625 -> 51 images
Class 626 -> 51 images
Class 627 -> 43 images
Class 628 -> 52 images
Class 629 -> 44 images
Class 630 -> 50 images
Class 631 -> 55 images
Class 632 -> 40 images
Class 633 -> 29 images
Class 634 -> 50 images
Class 635 -> 39 images
Class 636 -> 57 images
Class 637 -> 53 images
Class 638 -> 50 images
Class 639 -> 47 images
Class 640 -> 55 images
Class 641 -> 46 images
Class 642 -> 48 images
Class 643 -> 45 images
Class 644 -> 51 images
Class 645 -> 53 images
Class 646 -> 49 images
Class 647 -> 47 images
Class 648 -> 56 images
Class 649 -> 48 images
Class 650 -> 45 images
Class 651 -> 50 images
Class 652 -> 62 images
Class 653 -> 46 images
Class 654 -> 65 images
Class 655 -> 62 images
Class 656 -> 35 images
Class 657 -> 41 images
Class 658 -> 51 images
Class 659 -> 62 images
Class 660 -> 38 images
Class 661 -> 58 images
Class 662 -> 45 images
Class 663 -> 46 images
Class 664 -> 49 images
Class 665 -> 45 images
Class 666 -> 50 images
Class 667 -> 58 images
Class 668 -> 51 images
Class 669 -> 56 images
Class 670 -> 52 images
Class 671 -> 55 images
Class 672 -> 52 images
Class 673 -> 32 images
Class 674 -> 48 images
Class 675 -> 30 images
Class 676 -> 36 images
Class 677 -> 41 images
Class 678 -> 39 images
Class 679 -> 88 images
Class 680 -> 45 images
Class 681 -> 73 images
Class 682 -> 52 images
Class 683 -> 45 images
Class 684 -> 49 images
Class 685 -> 51 images
Class 686 -> 42 images
Class 687 -> 48 images
Class 688 -> 53 images
Class 689 -> 34 images
Class 690 -> 60 images
Class 691 -> 38 images
Class 692 -> 54 images
Class 693 -> 44 images
Class 694 -> 49 images
Class 695 -> 50 images
Class 696 -> 49 images
Class 697 -> 52 images
Class 698 -> 63 images
Class 699 -> 49 images
Class 700 -> 44 images
Class 701 -> 50 images
Class 702 -> 44 images
Class 703 -> 55 images
Class 704 -> 54 images
Class 705 -> 39 images
Class 706 -> 47 images
Class 707 -> 49 images
Class 708 -> 48 images
Class 709 -> 61 images
Class 710 -> 47 images
Class 711 -> 53 images
Class 712 -> 40 images
Class 713 -> 59 images
Class 714 -> 44 images
Class 715 -> 48 images
Class 716 -> 54 images
Class 717 -> 44 images
Class 718 -> 59 images
Class 719 -> 58 images
Class 720 -> 46 images
Class 721 -> 67 images
Class 722 -> 56 images
Class 723 -> 51 images
Class 724 -> 49 images
Class 725 -> 52 images
Class 726 -> 48 images
Class 727 -> 46 images
Class 728 -> 39 images
Class 729 -> 29 images
Class 730 -> 46 images
Class 731 -> 52 images
Class 732 -> 52 images
Class 733 -> 45 images
Class 734 -> 56 images
Class 735 -> 51 images
Class 736 -> 48 images
Class 737 -> 55 images
Class 738 -> 61 images
Class 739 -> 52 images
Class 740 -> 36 images
Class 741 -> 44 images
Class 742 -> 43 images
Class 743 -> 52 images
Class 744 -> 49 images
Class 745 -> 51 images
Class 746 -> 57 images
Class 747 -> 38 images
Class 748 -> 57 images
Class 749 -> 40 images
Class 750 -> 44 images
Class 751 -> 58 images
Class 752 -> 48 images
Class 753 -> 50 images
Class 754 -> 60 images
Class 755 -> 51 images
Class 756 -> 58 images
Class 757 -> 52 images
Class 758 -> 44 images
Class 759 -> 61 images
Class 760 -> 60 images
Class 761 -> 55 images
Class 762 -> 57 images
Class 763 -> 54 images
Class 764 -> 56 images
Class 765 -> 46 images
Class 766 -> 52 images
Class 767 -> 57 images
Class 768 -> 49 images
Class 769 -> 55 images
Class 770 -> 49 images
Class 771 -> 53 images
Class 772 -> 40 images
Class 773 -> 47 images
Class 774 -> 58 images
Class 775 -> 50 images
Class 776 -> 43 images
Class 777 -> 59 images
Class 778 -> 50 images
Class 779 -> 51 images
Class 780 -> 53 images
Class 781 -> 57 images
Class 782 -> 21 images
Class 783 -> 59 images
Class 784 -> 42 images
Class 785 -> 56 images
Class 786 -> 48 images
Class 787 -> 53 images
Class 788 -> 57 images
Class 789 -> 47 images
Class 790 -> 45 images
Class 791 -> 54 images
Class 792 -> 54 images
Class 793 -> 48 images
Class 794 -> 50 images
Class 795 -> 45 images
Class 796 -> 46 images
Class 797 -> 47 images
Class 798 -> 39 images
Class 799 -> 58 images
Class 800 -> 50 images
Class 801 -> 59 images
Class 802 -> 52 images
Class 803 -> 54 images
Class 804 -> 49 images
Class 805 -> 56 images
Class 806 -> 49 images
Class 807 -> 45 images
Class 808 -> 44 images
Class 809 -> 56 images
Class 810 -> 29 images
Class 811 -> 34 images
Class 812 -> 49 images
Class 813 -> 36 images
Class 814 -> 55 images
Class 815 -> 45 images
Class 816 -> 55 images
Class 817 -> 53 images
Class 818 -> 30 images
Class 819 -> 80 images
Class 820 -> 51 images
Class 821 -> 52 images
Class 822 -> 51 images
Class 823 -> 50 images
Class 824 -> 43 images
Class 825 -> 48 images
Class 826 -> 43 images
Class 827 -> 40 images
Class 828 -> 48 images
Class 829 -> 50 images
Class 830 -> 39 images
Class 831 -> 57 images
Class 832 -> 53 images
Class 833 -> 48 images
Class 834 -> 55 images
Class 835 -> 47 images
Class 836 -> 24 images
Class 837 -> 47 images
Class 838 -> 40 images
Class 839 -> 42 images
Class 840 -> 55 images
Class 841 -> 28 images
Class 842 -> 42 images
Class 843 -> 55 images
Class 844 -> 52 images
Class 845 -> 45 images
Class 846 -> 54 images
Class 847 -> 54 images
Class 848 -> 72 images
Class 849 -> 65 images
Class 850 -> 59 images
Class 851 -> 55 images
Class 852 -> 41 images
Class 853 -> 47 images
Class 854 -> 53 images
Class 855 -> 48 images
Class 856 -> 36 images
Class 857 -> 52 images
Class 858 -> 49 images
Class 859 -> 47 images
Class 860 -> 38 images
Class 861 -> 43 images
Class 862 -> 43 images
Class 863 -> 51 images
Class 864 -> 48 images
Class 865 -> 49 images
Class 866 -> 57 images
Class 867 -> 56 images
Class 868 -> 51 images
Class 869 -> 51 images
Class 870 -> 56 images
Class 871 -> 51 images
Class 872 -> 51 images
Class 873 -> 53 images
Class 874 -> 55 images
Class 875 -> 54 images
Class 876 -> 46 images
Class 877 -> 51 images
Class 878 -> 49 images
Class 879 -> 52 images
Class 880 -> 62 images
Class 881 -> 57 images
Class 882 -> 57 images
Class 883 -> 43 images
Class 884 -> 60 images
Class 885 -> 20 images
Class 886 -> 51 images
Class 887 -> 46 images
Class 888 -> 46 images
Class 889 -> 46 images
Class 890 -> 52 images
Class 891 -> 50 images
Class 892 -> 51 images
Class 893 -> 60 images
Class 894 -> 66 images
Class 895 -> 62 images
Class 896 -> 61 images
Class 897 -> 47 images
Class 898 -> 51 images
Class 899 -> 38 images
Class 900 -> 50 images
Class 901 -> 36 images
Class 902 -> 50 images
Class 903 -> 54 images
Class 904 -> 48 images
Class 905 -> 50 images
Class 906 -> 22 images
Class 907 -> 47 images
Class 908 -> 40 images
Class 909 -> 60 images
Class 910 -> 62 images
Class 911 -> 32 images
Class 912 -> 46 images
Class 913 -> 48 images
Class 914 -> 46 images
Class 915 -> 53 images
Class 916 -> 64 images
Class 917 -> 59 images
Class 918 -> 51 images
Class 919 -> 53 images
Class 920 -> 61 images
Class 921 -> 56 images
Class 922 -> 52 images
Class 923 -> 51 images
Class 924 -> 45 images
Class 925 -> 47 images
Class 926 -> 63 images
Class 927 -> 57 images
Class 928 -> 59 images
Class 929 -> 55 images
Class 930 -> 57 images
Class 931 -> 51 images
Class 932 -> 47 images
Class 933 -> 58 images
Class 934 -> 54 images
Class 935 -> 46 images
Class 936 -> 47 images
Class 937 -> 53 images
Class 938 -> 54 images
Class 939 -> 44 images
Class 940 -> 53 images
Class 941 -> 49 images
Class 942 -> 45 images
Class 943 -> 63 images
Class 944 -> 49 images
Class 945 -> 51 images
Class 946 -> 52 images
Class 947 -> 31 images
Class 948 -> 53 images
Class 949 -> 41 images
Class 950 -> 57 images
Class 951 -> 52 images
Class 952 -> 47 images
Class 953 -> 53 images
Class 954 -> 48 images
Class 955 -> 49 images
Class 956 -> 48 images
Class 957 -> 53 images
Class 958 -> 47 images
Class 959 -> 50 images
Class 960 -> 49 images
Class 961 -> 31 images
Class 962 -> 50 images
Class 963 -> 51 images
Class 964 -> 47 images
Class 965 -> 55 images
Class 966 -> 50 images
Class 967 -> 44 images
Class 968 -> 47 images
Class 969 -> 36 images
Class 970 -> 51 images
Class 971 -> 54 images
Class 972 -> 52 images
Class 973 -> 56 images
Class 974 -> 50 images
Class 975 -> 46 images
Class 976 -> 32 images
Class 977 -> 53 images
Class 978 -> 46 images
Class 979 -> 56 images
Class 980 -> 48 images
Class 981 -> 57 images
Class 982 -> 64 images
Class 983 -> 54 images
Class 984 -> 50 images
Class 985 -> 54 images
Class 986 -> 50 images
Class 987 -> 60 images
Class 988 -> 51 images
Class 989 -> 52 images
Class 990 -> 52 images
Class 991 -> 54 images
Class 992 -> 64 images
Class 993 -> 51 images
Class 994 -> 50 images
Class 995 -> 51 images
Class 996 -> 49 images
Class 997 -> 56 images
Class 998 -> 44 images
Class 999 -> 39 images
"""
