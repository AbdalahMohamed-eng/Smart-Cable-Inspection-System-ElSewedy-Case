{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "A46PCBJpAdPA"
      },
      "source": [
        "# Cable Defect Detection — YOLO26 "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PkNIbA93AdPE"
      },
      "source": [
        "## Step 1 — Install"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qNxKfawCAdPG",
        "outputId": "ea7e771b-4908-416f-da3e-b72c7069ecde"
      },
      "outputs": [],
      "source": [
        "!pip install ultralytics albumentations split-folders -q\n",
        "\n",
        "import os, shutil, yaml, random, glob, zipfile\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.patches as patches\n",
        "from PIL import Image\n",
        "import albumentations as A\n",
        "import cv2\n",
        "import splitfolders\n",
        "from ultralytics import YOLO\n",
        "from google.colab import files\n",
        "import pandas as pd\n",
        "\n",
        "print('Done!')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kLL0m7mnAdPI"
      },
      "source": [
        "## Step 2 — Upload ZIP"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 107
        },
        "id": "vdEWSMa5AdPJ",
        "outputId": "93d61cc4-2023-4502-f6e1-fb4b393037de"
      },
      "outputs": [],
      "source": [
        "print('Upload your ZIP file...')\n",
        "uploaded = files.upload()\n",
        "zip_name = list(uploaded.keys())[0]\n",
        "\n",
        "EXTRACT_PATH = '/content/dataset'\n",
        "os.makedirs(EXTRACT_PATH, exist_ok=True)\n",
        "with zipfile.ZipFile(zip_name, 'r') as z:\n",
        "    z.extractall(EXTRACT_PATH)\n",
        "\n",
        "RAW_IMAGES = '/content/dataset/Project/images'\n",
        "RAW_LABELS = '/content/dataset/Project/labels'\n",
        "images = sorted(glob.glob(f'{RAW_IMAGES}/*.*'))\n",
        "labels = sorted(glob.glob(f'{RAW_LABELS}/*.txt'))\n",
        "print(f'Images: {len(images)}  |  Labels: {len(labels)}')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "udMKSU6HAdPJ"
      },
      "source": [
        "## Step 3 — Labels (Map to 3 classes: defect | label | water)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0ngeLx95AdPK",
        "outputId": "f79cba22-fbe1-4bc0-9d76-ea65854cb648"
      },
      "outputs": [],
      "source": [
        "FIXED_IMAGES = '/content/fixed/images'\n",
        "FIXED_LABELS = '/content/fixed/labels'\n",
        "os.makedirs(FIXED_IMAGES, exist_ok=True)\n",
        "os.makedirs(FIXED_LABELS, exist_ok=True)\n",
        "\n",
        "def polygon_to_bbox(coords):\n",
        "    xs = coords[0::2]\n",
        "    ys = coords[1::2]\n",
        "    x_min, x_max = min(xs), max(xs)\n",
        "    y_min, y_max = min(ys), max(ys)\n",
        "    cx = (x_min + x_max) / 2\n",
        "    cy = (y_min + y_max) / 2\n",
        "    w  = x_max - x_min\n",
        "    h  = y_max - y_min\n",
        "    return cx, cy, w, h\n",
        "\n",
        "\n",
        "# map original class ids -> new class ids\n",
        "CLASS_MAP = {\n",
        "    0:0,  # defect\n",
        "    1:1,  # label\n",
        "    2:2   # water\n",
        "}\n",
        "\n",
        "converted = 0\n",
        "\n",
        "for lbl_path in labels:\n",
        "    stem = os.path.splitext(os.path.basename(lbl_path))[0]\n",
        "\n",
        "    img_path = None\n",
        "    for ext in ['.jpg','.jpeg','.png','.bmp']:\n",
        "        c = f'{RAW_IMAGES}/{stem}{ext}'\n",
        "        if os.path.exists(c):\n",
        "            img_path = c\n",
        "            break\n",
        "    if img_path is None:\n",
        "        continue\n",
        "\n",
        "    new_lines = []\n",
        "\n",
        "    with open(lbl_path) as f:\n",
        "        for line in f:\n",
        "            parts = line.strip().split()\n",
        "            if len(parts) < 5:\n",
        "                continue\n",
        "\n",
        "            cls = int(parts[0])\n",
        "            cls = CLASS_MAP.get(cls, cls)   # map class\n",
        "\n",
        "            coords = list(map(float, parts[1:]))\n",
        "\n",
        "            if len(coords) == 4:\n",
        "                cx, cy, w, h = coords\n",
        "            elif len(coords) >= 6 and len(coords) % 2 == 0:\n",
        "                cx, cy, w, h = polygon_to_bbox(coords)\n",
        "            else:\n",
        "                cx, cy, w, h = coords[:4]\n",
        "\n",
        "            cx = max(0.0, min(1.0, cx))\n",
        "            cy = max(0.0, min(1.0, cy))\n",
        "            w  = max(0.001, min(1.0, w))\n",
        "            h  = max(0.001, min(1.0, h))\n",
        "\n",
        "            new_lines.append(f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')\n",
        "\n",
        "    if new_lines:\n",
        "        with open(f'{FIXED_LABELS}/{stem}.txt','w') as f:\n",
        "            f.write('\\n'.join(new_lines)+'\\n')\n",
        "\n",
        "        shutil.copy(img_path, f'{FIXED_IMAGES}/{os.path.basename(img_path)}')\n",
        "        converted += 1\n",
        "\n",
        "print(f'Converted {converted} label files -> 3 classes (defect, label, water)')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_JjWrPleAdPM"
      },
      "source": [
        "## Step 4 — Visual Check of Labels"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 592
        },
        "id": "ahtFQcdRAdPN",
        "outputId": "e727d30d-8b62-45f8-8a7a-1115bbd3014e"
      },
      "outputs": [],
      "source": [
        "CLASS_NAMES = ['defect','label','water']\n",
        "CLASS_COLORS = ['red','blue','green']\n",
        "\n",
        "fixed_imgs = sorted(glob.glob(f'{FIXED_IMAGES}/*.*'))\n",
        "samples = random.sample(fixed_imgs, min(6, len(fixed_imgs)))\n",
        "\n",
        "fig, axes = plt.subplots(2, 3, figsize=(16, 10))\n",
        "axes = axes.flatten()\n",
        "\n",
        "for i, img_path in enumerate(samples):\n",
        "    img = np.array(Image.open(img_path).convert('RGB'))\n",
        "    h, w = img.shape[:2]\n",
        "\n",
        "    ax = axes[i]\n",
        "    ax.imshow(img)\n",
        "\n",
        "    stem = os.path.splitext(os.path.basename(img_path))[0]\n",
        "    lbl  = f'{FIXED_LABELS}/{stem}.txt'\n",
        "\n",
        "    if os.path.exists(lbl):\n",
        "        with open(lbl) as f:\n",
        "            for line in f:\n",
        "                p = line.strip().split()\n",
        "                if len(p) == 5:\n",
        "\n",
        "                    cls, cx, cy, bw, bh = map(float, p)\n",
        "                    cls = int(cls)\n",
        "\n",
        "                    x1 = int((cx - bw/2) * w)\n",
        "                    y1 = int((cy - bh/2) * h)\n",
        "                    x2 = int((cx + bw/2) * w)\n",
        "                    y2 = int((cy + bh/2) * h)\n",
        "\n",
        "                    color = CLASS_COLORS[cls]\n",
        "                    name  = CLASS_NAMES[cls]\n",
        "\n",
        "                    ax.add_patch(\n",
        "                        patches.Rectangle(\n",
        "                            (x1,y1),\n",
        "                            x2-x1,\n",
        "                            y2-y1,\n",
        "                            linewidth=2,\n",
        "                            edgecolor=color,\n",
        "                            facecolor='none'\n",
        "                        )\n",
        "                    )\n",
        "\n",
        "                    ax.text(\n",
        "                        x1,\n",
        "                        max(0,y1-5),\n",
        "                        name,\n",
        "                        color='white',\n",
        "                        fontsize=8,\n",
        "                        bbox=dict(facecolor=color,alpha=0.7,pad=1)\n",
        "                    )\n",
        "\n",
        "    ax.set_title(os.path.basename(img_path), fontsize=7)\n",
        "    ax.axis('off')\n",
        "\n",
        "plt.suptitle('Fixed Labels - Bounding Boxes (3 classes)', fontsize=13, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('Check: are the boxes correctly around defect / label / water?')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3ZsRmu57AdPP"
      },
      "source": [
        "## Step 5 — Augmentation (70 -> 700+ images)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DYhpAXmyAdPP",
        "outputId": "31bb9343-f58f-494a-bdcc-5bc601faf155"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import cv2\n",
        "import glob\n",
        "import shutil\n",
        "import random\n",
        "import albumentations as A\n",
        "import numpy as np\n",
        "from collections import Counter\n",
        "from PIL import Image\n",
        "\n",
        "FIXED_IMAGES = '/content/fixed/images'\n",
        "FIXED_LABELS = '/content/fixed/labels'\n",
        "\n",
        "AUG_IMAGES = '/content/augmented/images'\n",
        "AUG_LABELS = '/content/augmented/labels'\n",
        "\n",
        "os.makedirs(AUG_IMAGES, exist_ok=True)\n",
        "os.makedirs(AUG_LABELS, exist_ok=True)\n",
        "\n",
        "CLASS_NAMES = {0:'defect',1:'label',2:'water'}\n",
        "\n",
        "# ─────────────────────────────────────────────\n",
        "# Base augmentation\n",
        "# ─────────────────────────────────────────────\n",
        "\n",
        "transform_base = A.Compose(\n",
        "[\n",
        "    A.HorizontalFlip(p=0.5),\n",
        "    A.VerticalFlip(p=0.3),\n",
        "    A.Rotate(limit=20, p=0.5),\n",
        "\n",
        "    A.RandomBrightnessContrast(\n",
        "        brightness_limit=0.3,\n",
        "        contrast_limit=0.3,\n",
        "        p=0.6\n",
        "    ),\n",
        "\n",
        "    A.HueSaturationValue(\n",
        "        hue_shift_limit=10,\n",
        "        sat_shift_limit=30,\n",
        "        val_shift_limit=20,\n",
        "        p=0.4\n",
        "    ),\n",
        "\n",
        "    A.GaussNoise(std_range=(0.04,0.1), p=0.4),\n",
        "\n",
        "    A.GaussianBlur(blur_limit=3, p=0.3),\n",
        "\n",
        "    A.CLAHE(clip_limit=2.0, p=0.3),\n",
        "\n",
        "    A.RandomShadow(p=0.2),\n",
        "\n",
        "    A.Perspective(scale=(0.02,0.08), p=0.3),\n",
        "\n",
        "    A.RandomScale(scale_limit=0.2, p=0.3),\n",
        "\n",
        "],\n",
        "bbox_params=A.BboxParams(\n",
        "    format='yolo',\n",
        "    label_fields=['class_labels'],\n",
        "    min_visibility=0.4\n",
        ")\n",
        ")\n",
        "\n",
        "# ─────────────────────────────────────────────\n",
        "# Extra water augmentation\n",
        "# ─────────────────────────────────────────────\n",
        "\n",
        "transform_water = A.Compose(\n",
        "[\n",
        "    A.HorizontalFlip(p=0.5),\n",
        "    A.VerticalFlip(p=0.5),\n",
        "\n",
        "    A.Rotate(limit=30, p=0.7),\n",
        "\n",
        "    A.RandomBrightnessContrast(\n",
        "        brightness_limit=0.4,\n",
        "        contrast_limit=0.4,\n",
        "        p=0.8\n",
        "    ),\n",
        "\n",
        "    A.HueSaturationValue(\n",
        "        hue_shift_limit=20,\n",
        "        sat_shift_limit=50,\n",
        "        val_shift_limit=30,\n",
        "        p=0.7\n",
        "    ),\n",
        "\n",
        "    A.GaussNoise(std_range=(0.06,0.15), p=0.6),\n",
        "\n",
        "    A.GaussianBlur(blur_limit=5, p=0.5),\n",
        "\n",
        "    A.CLAHE(clip_limit=3.0, p=0.5),\n",
        "\n",
        "    A.RandomShadow(p=0.4),\n",
        "\n",
        "    A.Perspective(scale=(0.04,0.12), p=0.5),\n",
        "\n",
        "    A.RandomScale(scale_limit=0.3, p=0.5),\n",
        "\n",
        "    A.ElasticTransform(alpha=80, sigma=10, p=0.3),\n",
        "\n",
        "    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),\n",
        "\n",
        "    A.CoarseDropout(\n",
        "        num_holes_range=(1,4),\n",
        "        hole_height_range=(10,20),\n",
        "        hole_width_range=(10,20),\n",
        "        p=0.2\n",
        "    )\n",
        "\n",
        "],\n",
        "bbox_params=A.BboxParams(\n",
        "    format='yolo',\n",
        "    label_fields=['class_labels'],\n",
        "    min_visibility=0.35\n",
        ")\n",
        ")\n",
        "\n",
        "# ─────────────────────────────────────────────\n",
        "# Augmentation counts per class\n",
        "# ─────────────────────────────────────────────\n",
        "\n",
        "AUGMENT_COUNTS = {\n",
        "0:10,   # defect\n",
        "1:10,   # label\n",
        "2:30    # water (extra)\n",
        "}\n",
        "\n",
        "DEFAULT_AUG = 10\n",
        "\n",
        "# ─────────────────────────────────────────────\n",
        "# Start augmentation\n",
        "# ─────────────────────────────────────────────\n",
        "\n",
        "total = 0\n",
        "src_imgs = sorted(glob.glob(f'{FIXED_IMAGES}/*.*'))\n",
        "\n",
        "for img_path in src_imgs:\n",
        "\n",
        "    stem = os.path.splitext(os.path.basename(img_path))[0]\n",
        "    lbl_path = f'{FIXED_LABELS}/{stem}.txt'\n",
        "\n",
        "    img = cv2.imread(img_path)\n",
        "    if img is None:\n",
        "        continue\n",
        "\n",
        "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
        "\n",
        "    bboxes = []\n",
        "    class_labels = []\n",
        "\n",
        "    if os.path.exists(lbl_path):\n",
        "\n",
        "        with open(lbl_path) as f:\n",
        "            for line in f:\n",
        "                p = line.strip().split()\n",
        "\n",
        "                if len(p)==5:\n",
        "                    class_labels.append(int(float(p[0])))\n",
        "                    bboxes.append([float(x) for x in p[1:5]])\n",
        "\n",
        "    if not bboxes:\n",
        "        continue\n",
        "\n",
        "    dominant_cls = Counter(class_labels).most_common(1)[0][0]\n",
        "\n",
        "    n_aug = AUGMENT_COUNTS.get(dominant_cls, DEFAULT_AUG)\n",
        "\n",
        "    transform = transform_water if dominant_cls == 2 else transform_base\n",
        "\n",
        "    # copy original\n",
        "    shutil.copy(img_path, f'{AUG_IMAGES}/{stem}_orig.jpg')\n",
        "    shutil.copy(lbl_path, f'{AUG_LABELS}/{stem}_orig.txt')\n",
        "\n",
        "    # generate augmentations\n",
        "    for k in range(n_aug):\n",
        "\n",
        "        try:\n",
        "\n",
        "            aug = transform(\n",
        "                image=img,\n",
        "                bboxes=bboxes,\n",
        "                class_labels=class_labels\n",
        "            )\n",
        "\n",
        "            if not aug['bboxes']:\n",
        "                continue\n",
        "\n",
        "            name = f'{stem}_aug{k:03d}'\n",
        "\n",
        "            cv2.imwrite(\n",
        "                f'{AUG_IMAGES}/{name}.jpg',\n",
        "                cv2.cvtColor(aug['image'], cv2.COLOR_RGB2BGR)\n",
        "            )\n",
        "\n",
        "            with open(f'{AUG_LABELS}/{name}.txt','w') as f:\n",
        "\n",
        "                for c, box in zip(aug['class_labels'], aug['bboxes']):\n",
        "\n",
        "                    f.write(\n",
        "                        f'{int(c)} '\n",
        "                        + \" \".join([f\"{v:.6f}\" for v in box])\n",
        "                        + '\\n'\n",
        "                    )\n",
        "\n",
        "            total += 1\n",
        "\n",
        "        except:\n",
        "            pass\n",
        "\n",
        "# ─────────────────────────────────────────────\n",
        "# Dataset summary\n",
        "# ─────────────────────────────────────────────\n",
        "\n",
        "aug_imgs = sorted(glob.glob(f'{AUG_IMAGES}/*.jpg'))\n",
        "\n",
        "print(f'Original : {len(src_imgs)}')\n",
        "print(f'Generated: {total}')\n",
        "print(f'Total    : {len(aug_imgs)}')\n",
        "\n",
        "# class distribution\n",
        "dist2 = Counter()\n",
        "\n",
        "for t in glob.glob(f'{AUG_LABELS}/*.txt'):\n",
        "\n",
        "    for line in open(t):\n",
        "\n",
        "        p = line.strip().split()\n",
        "\n",
        "        if p:\n",
        "            dist2[int(float(p[0]))] += 1\n",
        "\n",
        "print('\\nAugmented class distribution:')\n",
        "\n",
        "for cid, cnt in sorted(dist2.items()):\n",
        "\n",
        "    print(f'class {cid} ({CLASS_NAMES[cid]}): {cnt} boxes')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wjTVK6pyAdPQ"
      },
      "source": [
        "## Step 6 — Split (Train 70% / Val 20% / Test 10%)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IjpbWUhUAdPR",
        "outputId": "9045cf8e-8a85-4a0e-e804-62b81c15a572"
      },
      "outputs": [],
      "source": [
        "SPLIT_DIR  = '/content/cable_dataset'\n",
        "PAIRED_DIR = '/content/paired'\n",
        "\n",
        "for img_path in glob.glob(f'{AUG_IMAGES}/*.jpg'):\n",
        "    stem = os.path.splitext(os.path.basename(img_path))[0]\n",
        "    lbl  = f'{AUG_LABELS}/{stem}.txt'\n",
        "    if not os.path.exists(lbl): continue\n",
        "    di = f'{PAIRED_DIR}/images/{stem}.jpg'\n",
        "    dl = f'{PAIRED_DIR}/labels/{stem}.txt'\n",
        "    os.makedirs(os.path.dirname(di), exist_ok=True)\n",
        "    os.makedirs(os.path.dirname(dl), exist_ok=True)\n",
        "    shutil.copy(img_path, di)\n",
        "    shutil.copy(lbl, dl)\n",
        "\n",
        "splitfolders.ratio(f'{PAIRED_DIR}/images', output=f'{SPLIT_DIR}/images',\n",
        "    seed=42, ratio=(0.7, 0.2, 0.1))\n",
        "\n",
        "for split in ['train','val','test']:\n",
        "    os.makedirs(f'{SPLIT_DIR}/labels/{split}', exist_ok=True)\n",
        "    for img_path in glob.glob(f'{SPLIT_DIR}/images/{split}/*.jpg'):\n",
        "        stem = os.path.splitext(os.path.basename(img_path))[0]\n",
        "        src  = f'{PAIRED_DIR}/labels/{stem}.txt'\n",
        "        if os.path.exists(src):\n",
        "            shutil.copy(src, f'{SPLIT_DIR}/labels/{split}/{stem}.txt')\n",
        "\n",
        "print(f'Train: {len(glob.glob(f\"{SPLIT_DIR}/images/train/*.jpg\"))}')\n",
        "print(f'Val  : {len(glob.glob(f\"{SPLIT_DIR}/images/val/*.jpg\"))}')\n",
        "print(f'Test : {len(glob.glob(f\"{SPLIT_DIR}/images/test/*.jpg\"))}')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jAjfs1KxAdPR"
      },
      "source": [
        "## Step 7 — Create YAML Config"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cpHb5R-_AdPS",
        "outputId": "ac9b946b-4eb8-465f-fba7-1b8974231408"
      },
      "outputs": [],
      "source": [
        "YAML_PATH = '/content/cable_dataset/dataset.yaml'\n",
        "with open(YAML_PATH, 'w') as f:\n",
        "    yaml.dump({\n",
        "        'path' : SPLIT_DIR,\n",
        "        'train': 'images/train',\n",
        "        'val'  : 'images/val',\n",
        "        'test' : 'images/test',\n",
        "        'nc'   : 3,\n",
        "        'names': ['defect', 'label', 'water']\n",
        "    }, f)\n",
        "print(open(YAML_PATH).read())\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "keakP78LAdPT"
      },
      "source": [
        "## Step 8 — Train YOLO26"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aP4KTph_AdPU",
        "outputId": "7553b941-3ddb-460f-ad55-db89ce4fe0f8"
      },
      "outputs": [],
      "source": [
        "model = YOLO('yolo26s.pt')\n",
        "results = model.train(\n",
        "    data=YAML_PATH,\n",
        "    epochs=160,           # more epochs — was 150\n",
        "    imgsz=640,\n",
        "    batch=16,\n",
        "    patience=40,          # wait longer before early stopping\n",
        "    lr0=0.01,\n",
        "    lrf=0.0005,           # lower final LR for finer convergence\n",
        "    momentum=0.937,\n",
        "    weight_decay=0.0005,\n",
        "    warmup_epochs=5,\n",
        "    # ── Augmentation ────────────────────────────────────────────────────────\n",
        "    hsv_h=0.02,           # more hue shift — helps water color variation\n",
        "    hsv_s=0.8,            # more saturation — water vs defect separation\n",
        "    hsv_v=0.5,\n",
        "    degrees=20.0,\n",
        "    translate=0.15,\n",
        "    scale=0.6,\n",
        "    flipud=0.4,\n",
        "    fliplr=0.5,\n",
        "    mosaic=1.0,\n",
        "    mixup=0.15,           # slight increase\n",
        "    copy_paste=0.15,      # helps minority class (water)\n",
        "    # ── Loss weights: boost water recall ────────────────────────────────────\n",
        "    cls=1.5,              # raise classification loss weight (default 0.5)\n",
        "                          # forces model to distinguish water vs defect\n",
        "    # ── Output ──────────────────────────────────────────────────────────────\n",
        "    project='/content/runs',\n",
        "    name='cable_defect_v3',\n",
        "    save=True,\n",
        "    plots=True,\n",
        "    verbose=True\n",
        ")\n",
        "print('Training complete!')\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gkR-NhG5AdPV"
      },
      "source": [
        "## Step 9 — Evaluate"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jIJ35-TgAdPV",
        "outputId": "519253f2-6999-4248-eeb2-0a4e5b7fd21e"
      },
      "outputs": [],
      "source": [
        "BEST_WEIGHTS = '/content/runs/cable_defect_v32/weights/best.pt'\n",
        "best_model   = YOLO(BEST_WEIGHTS)\n",
        "metrics = best_model.val(data=YAML_PATH, split='test',\n",
        "                          imgsz=640, conf=0.35, iou=0.5, plots=True)\n",
        "print('\\n===== MODEL PERFORMANCE (3-class) =====')\n",
        "print(f'mAP@0.5      : {metrics.box.map50:.4f}')\n",
        "print(f'mAP@0.5:0.95 : {metrics.box.map:.4f}')\n",
        "print(f'Precision    : {metrics.box.mp:.4f}')\n",
        "print(f'Recall       : {metrics.box.mr:.4f}')\n",
        "CLASS_NAMES_LIST = ['defect', 'label', 'water']\n",
        "TARGET_RECALL    = {'defect': 0.87, 'label': 0.86, 'water': 0.80}  # targets\n",
        "if hasattr(metrics.box, 'ap_class_index'):\n",
        "    print('\\n--- Per-class AP@0.5 (target: all >= 0.88) ---')\n",
        "    for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):\n",
        "        name = CLASS_NAMES_LIST[idx]\n",
        "        flag = 'OK' if ap >= 0.88 else 'needs improvement'\n",
        "        print(f'  {name:8s}: {ap:.4f}  {flag}')\n",
        "print('=======================================')\n",
        "if metrics.box.map50 >= 0.90:\n",
        "    print('Excellent — production ready!')\n",
        "elif metrics.box.map50 >= 0.85:\n",
        "    print('Model is production ready!')\n",
        "elif metrics.box.map50 >= 0.70:\n",
        "    print('Good — consider more epochs or data.')\n",
        "else:\n",
        "    print('Needs improvement — check label quality in Step 4.')\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "69nvkXgXAdPW"
      },
      "source": [
        "## Step 10 — Training Curves"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 659
        },
        "id": "sKvl96LsAdPX",
        "outputId": "c1daf0ab-6393-4c55-8f98-30d5cf6adc1a"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "csv_path = '/content/runs/cable_defect_v32/results.csv'\n",
        "df = pd.read_csv(csv_path)\n",
        "df.columns = df.columns.str.strip()\n",
        "\n",
        "# ── Debug: show available columns ────────────────────────────────────────────\n",
        "print('Available columns:', df.columns.tolist())\n",
        "\n",
        "# ── Auto-detect correct column names ─────────────────────────────────────────\n",
        "def find_col(df, candidates):\n",
        "    for c in candidates:\n",
        "        if c in df.columns: return c\n",
        "    return None\n",
        "\n",
        "epoch_col = find_col(df, ['epoch', 'Epoch'])\n",
        "\n",
        "plots = [\n",
        "    (find_col(df, ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']), 'Train Box Loss', 'blue'),\n",
        "    (find_col(df, ['val/box_loss',   'val/cls_loss',   'val/dfl_loss'  ]), 'Val Box Loss',   'orange'),\n",
        "    (find_col(df, ['metrics/mAP50(B)',  'metrics/mAP50', 'metrics/mAP_0.5'                 ]), 'mAP@0.5',        'green'),\n",
        "    (find_col(df, ['metrics/mAP50-95(B)','metrics/mAP50-95','metrics/mAP_0.5:0.95'         ]), 'mAP@0.5:0.95',  'purple'),\n",
        "    (find_col(df, ['metrics/precision(B)', 'metrics/precision'                              ]), 'Precision',      'red'),\n",
        "    (find_col(df, ['metrics/recall(B)',    'metrics/recall'                                 ]), 'Recall',         'teal'),\n",
        "]\n",
        "\n",
        "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
        "for ax, (col, title, color) in zip(axes.flatten(), plots):\n",
        "    if col and col in df.columns:\n",
        "        ax.plot(df[epoch_col], df[col], color=color, linewidth=2)\n",
        "        ax.set_title(title, fontweight='bold')\n",
        "        ax.set_xlabel('Epoch')\n",
        "        ax.grid(alpha=0.3)\n",
        "        # Annotate final value\n",
        "        final = df[col].iloc[-1]\n",
        "        ax.annotate(f'{final:.3f}', xy=(df[epoch_col].iloc[-1], final),\n",
        "                    fontsize=9, color=color, fontweight='bold',\n",
        "                    xytext=(-30, 8), textcoords='offset points')\n",
        "    else:\n",
        "        ax.set_title(f'{title} (not found)', fontweight='bold', color='gray')\n",
        "        ax.text(0.5, 0.5, 'Column not available', ha='center', va='center',\n",
        "                transform=ax.transAxes, color='gray')\n",
        "        ax.grid(alpha=0.3)\n",
        "\n",
        "plt.suptitle('Training Curves — Cable Defect v3', fontsize=14, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.savefig('/content/curves.png', dpi=150)\n",
        "plt.show()\n",
        "print('\\nFinal metrics at last epoch:')\n",
        "for col, title, _ in plots:\n",
        "    if col and col in df.columns:\n",
        "        print(f'  {title:20s}: {df[col].iloc[-1]:.4f}')\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QZsv9XM0AdPX"
      },
      "source": [
        "## Step 11 — Predict on Test Images"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 548
        },
        "id": "htTcFCPAAdPY",
        "outputId": "13c54d78-2381-4404-880b-f0900b914c8a"
      },
      "outputs": [],
      "source": [
        "test_images = glob.glob(f'{SPLIT_DIR}/images/test/*.jpg')[:8]\n",
        "preds = best_model.predict(source=test_images, conf=0.25, iou=0.45, imgsz=640)\n",
        "fig, axes = plt.subplots(2, 4, figsize=(22,10))\n",
        "axes = axes.flatten()\n",
        "for i, result in enumerate(preds[:8]):\n",
        "    axes[i].imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))\n",
        "    if result.boxes and len(result.boxes.conf) > 0:\n",
        "        c = result.boxes.conf.cpu().numpy()\n",
        "        axes[i].set_title(f'{len(c)} defect(s) | max: {c.max():.0%}',\n",
        "            fontsize=9, color='red', fontweight='bold')\n",
        "    else:\n",
        "        axes[i].set_title('No defect', fontsize=9, color='green')\n",
        "    axes[i].axis('off')\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WK9f4y1SAdPY"
      },
      "source": [
        "## Step 12 — Test Your Own Image"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 553
        },
        "id": "dyU8AqX2AdPY",
        "outputId": "c6f836f9-5d2f-4e3e-ad95-a284231d5b38"
      },
      "outputs": [],
      "source": [
        "CLASS_NAMES = ['defect', 'label', 'water']\n",
        "\n",
        "# ── Confidence Settings ───────────────────────────────────────────────────────\n",
        "CONF_THRESHOLD = 0.45   # slightly lower to improve water recall\n",
        "IOU_THRESHOLD  = 0.40\n",
        "# Per-class confidence override (water uses lower threshold)\n",
        "PER_CLASS_CONF = {0: 0.50, 1: 0.50, 2: 0.25}  # defect | label | water\n",
        "# ─────────────────────────────────────────────────────────────────────────────\n",
        "\n",
        "def predict_single(image_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,\n",
        "                   per_class_filter=True):\n",
        "    result = best_model.predict(\n",
        "        source=image_path,\n",
        "        conf=min(PER_CLASS_CONF.values()) if per_class_filter else conf,\n",
        "        iou=iou, imgsz=640, augment=True\n",
        "    )[0]\n",
        "\n",
        "    # Apply per-class confidence filter\n",
        "    if per_class_filter and result.boxes and len(result.boxes) > 0:\n",
        "        confs_all = result.boxes.conf.cpu().numpy()\n",
        "        clses_all = result.boxes.cls.cpu().numpy().astype(int)\n",
        "        keep = [i for i, (c, k) in enumerate(zip(confs_all, clses_all))\n",
        "                if c >= PER_CLASS_CONF.get(k, conf)]\n",
        "        if len(keep) < len(confs_all):\n",
        "            result.boxes = result.boxes[keep]\n",
        "\n",
        "    plt.figure(figsize=(11, 8))\n",
        "    plt.imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))\n",
        "\n",
        "    if result.boxes and len(result.boxes) > 0:\n",
        "        confs = result.boxes.conf.cpu().numpy()\n",
        "        clses = result.boxes.cls.cpu().numpy().astype(int)\n",
        "        avg_conf = confs.mean()\n",
        "        title = (f'{len(confs)} detection(s)  |  '\n",
        "                 f'avg conf: {avg_conf:.1%}  |  max: {confs.max():.1%}')\n",
        "        plt.title(title, color='red', fontsize=13, fontweight='bold')\n",
        "        from collections import Counter\n",
        "        print('\\n── Detection Report ──────────────────────')\n",
        "        class_counts = Counter(clses)\n",
        "        for cid, cnt in sorted(class_counts.items()):\n",
        "            cls_confs = confs[clses == cid]\n",
        "            print(f'  {CLASS_NAMES[cid]:8s} x{cnt}  |  '\n",
        "                  f'avg: {cls_confs.mean():.1%}  max: {cls_confs.max():.1%}  '\n",
        "                  f'(threshold: {PER_CLASS_CONF.get(cid, conf):.0%})')\n",
        "        print('─────────────────────────────────────────')\n",
        "        for j, (c, cid) in enumerate(zip(confs, clses)):\n",
        "            print(f'  [{j+1}] {CLASS_NAMES[cid]:8s}  {c:.1%}')\n",
        "    else:\n",
        "        plt.title('No Detection (try lowering conf threshold)',\n",
        "                  color='green', fontsize=13, fontweight='bold')\n",
        "        print(f'No detections. Try: predict_single(path, conf=0.25)')\n",
        "    plt.axis('off'); plt.tight_layout(); plt.show()\n",
        "\n",
        "# Test on a sample from test set\n",
        "predict_single(glob.glob(f'{SPLIT_DIR}/images/test/*.jpg')[0])\n",
        "\n",
        "# Upload your own image:\n",
        "# u = files.upload(); predict_single(list(u.keys())[0])\n",
        "\n",
        "# Fine-tune per-class thresholds:\n",
        "# PER_CLASS_CONF = {0: 0.55, 1: 0.55, 2: 0.30}  # aggressive water detection\n",
        "# PER_CLASS_CONF = {0: 0.60, 1: 0.60, 2: 0.45}  # conservative\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lJWUZ1hEAdPZ"
      },
      "source": [
        "## Step 13 — Export & Download"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 280
        },
        "id": "rf9BIhmOAdPZ",
        "outputId": "1bac2c77-68bd-4ff7-9d95-ea1f6cdf1397"
      },
      "outputs": [],
      "source": [
        "best_model.export(format='onnx', imgsz=640, simplify=True)\n",
        "EXPORT = '/content/export'\n",
        "os.makedirs(EXPORT, exist_ok=True)\n",
        "BEST_WEIGHTS = '/content/runs/cable_defect_v32/weights/best.pt'\n",
        "shutil.copy(BEST_WEIGHTS, f'{EXPORT}/best.pt')\n",
        "shutil.copy(YAML_PATH,    f'{EXPORT}/dataset.yaml')\n",
        "onnx = BEST_WEIGHTS.replace('best.pt','best.onnx')\n",
        "if os.path.exists(onnx): shutil.copy(onnx, f'{EXPORT}/best.onnx')\n",
        "shutil.make_archive('/content/cable_model_v32', 'zip', EXPORT)\n",
        "files.download('/content/cable_model_v32.zip')\n",
        "print('Done!')\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vVKYxeTSAdPa"
      },
      "source": [
        "## Step 14 — Batch Live Demo (Upload Multiple Images)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "jkgSgptUAdPa",
        "outputId": "5c4e9700-937c-4c7d-c25f-c455b353e4fa"
      },
      "outputs": [],
      "source": [
        "CLASS_NAMES    = ['defect', 'label', 'water']\n",
        "CONF_THRESHOLD = 0.45\n",
        "IOU_THRESHOLD  = 0.40\n",
        "PER_CLASS_CONF = {0: 0.50, 1: 0.50, 2: 0.35}\n",
        "\n",
        "print('Upload one or more images for batch prediction...')\n",
        "uploaded_batch = files.upload()\n",
        "\n",
        "all_results = []\n",
        "for fname in uploaded_batch:\n",
        "    r = best_model.predict(\n",
        "        source=fname,\n",
        "        conf=min(PER_CLASS_CONF.values()),\n",
        "        iou=IOU_THRESHOLD, imgsz=640, augment=True\n",
        "    )[0]\n",
        "    # Apply per-class filter\n",
        "    if r.boxes and len(r.boxes) > 0:\n",
        "        ca = r.boxes.conf.cpu().numpy()\n",
        "        ka = r.boxes.cls.cpu().numpy().astype(int)\n",
        "        keep = [i for i,(c,k) in enumerate(zip(ca,ka)) if c >= PER_CLASS_CONF.get(k, CONF_THRESHOLD)]\n",
        "        if len(keep) < len(ca): r.boxes = r.boxes[keep]\n",
        "    all_results.append((fname, r))\n",
        "\n",
        "n = len(all_results)\n",
        "cols = min(3, n); rows = (n + cols - 1) // cols\n",
        "fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 6*rows))\n",
        "axes_flat = axes.flatten() if n > 1 else [axes]\n",
        "\n",
        "print('\\n== Batch Confidence Report ===================')\n",
        "for ax, (fname, result) in zip(axes_flat, all_results):\n",
        "    ax.imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))\n",
        "    if result.boxes and len(result.boxes) > 0:\n",
        "        confs = result.boxes.conf.cpu().numpy()\n",
        "        clses = result.boxes.cls.cpu().numpy().astype(int)\n",
        "        summary = '  '.join([f'{CLASS_NAMES[k]}:{v:.0%}' for k,v in zip(clses,confs)])\n",
        "        ax.set_title(f'{os.path.basename(fname)}\\n{summary}',\n",
        "                     fontsize=8, color='red', fontweight='bold')\n",
        "        print(f'  {fname}: {summary}')\n",
        "    else:\n",
        "        ax.set_title(f'{os.path.basename(fname)}\\nNo Detection', fontsize=8, color='green')\n",
        "        print(f'  {fname}: No detection')\n",
        "    ax.axis('off')\n",
        "for ax in axes_flat[n:]: ax.axis('off')\n",
        "print('==============================================')\n",
        "plt.suptitle(f'Batch Results — defect|label|water', fontsize=13, fontweight='bold')\n",
        "plt.tight_layout(); plt.show()\n"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
