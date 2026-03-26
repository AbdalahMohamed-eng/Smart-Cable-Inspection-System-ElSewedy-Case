{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cable Defect Detection — YOLO26 Final "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1 — Install"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
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
   "metadata": {},
   "source": [
    "## Step 2 — Upload ZIP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
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
   "metadata": {},
   "source": [
    "## Step 3 — Labels "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "FIXED_IMAGES = '/content/fixed/images'\n",
    "FIXED_LABELS = '/content/fixed/labels'\n",
    "os.makedirs(FIXED_IMAGES, exist_ok=True)\n",
    "os.makedirs(FIXED_LABELS, exist_ok=True)\n",
    "\n",
    "# Class mapping: original_id -> new_id\n",
    "# 0=defect | 1=label | 2=water\n",
    "CLASS_MAP   = {0: 0, 1: 1, 2: 2}\n",
    "CLASS_NAMES = {0: 'defect', 1: 'label', 2: 'water'}\n",
    "DEFAULT_CLASS = 0\n",
    "\n",
    "def polygon_to_bbox(coords):\n",
    "    xs = coords[0::2]; ys = coords[1::2]\n",
    "    x_min, x_max = min(xs), max(xs)\n",
    "    y_min, y_max = min(ys), max(ys)\n",
    "    cx = (x_min + x_max) / 2; cy = (y_min + y_max) / 2\n",
    "    w  = x_max - x_min;       h  = y_max - y_min\n",
    "    return cx, cy, w, h\n",
    "\n",
    "converted = 0\n",
    "for lbl_path in labels:\n",
    "    stem = os.path.splitext(os.path.basename(lbl_path))[0]\n",
    "    img_path = None\n",
    "    for ext in ['.jpg','.jpeg','.png','.bmp']:\n",
    "        c = f'{RAW_IMAGES}/{stem}{ext}'\n",
    "        if os.path.exists(c): img_path = c; break\n",
    "    if img_path is None: continue\n",
    "    new_lines = []\n",
    "    with open(lbl_path) as f:\n",
    "        for line in f:\n",
    "            parts = line.strip().split()\n",
    "            if len(parts) < 5: continue\n",
    "            orig_cls = int(parts[0])\n",
    "            new_cls  = CLASS_MAP.get(orig_cls, DEFAULT_CLASS)\n",
    "            coords   = list(map(float, parts[1:]))\n",
    "            if len(coords) == 4:\n",
    "                cx, cy, w, h = coords\n",
    "            elif len(coords) >= 6 and len(coords) % 2 == 0:\n",
    "                cx, cy, w, h = polygon_to_bbox(coords)\n",
    "            else:\n",
    "                cx, cy, w, h = coords[:4]\n",
    "            cx = max(0.0, min(1.0, cx)); cy = max(0.0, min(1.0, cy))\n",
    "            w  = max(0.001, min(1.0, w)); h  = max(0.001, min(1.0, h))\n",
    "            new_lines.append(f'{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')\n",
    "    if new_lines:\n",
    "        with open(f'{FIXED_LABELS}/{stem}.txt','w') as f:\n",
    "            f.write('\\n'.join(new_lines)+'\\n')\n",
    "        shutil.copy(img_path, f'{FIXED_IMAGES}/{os.path.basename(img_path)}')\n",
    "        converted += 1\n",
    "\n",
    "print(f'Converted {converted} label files -> 3 classes: defect | label | water')\n",
    "from collections import Counter\n",
    "dist = Counter()\n",
    "for t in glob.glob(f'{FIXED_LABELS}/*.txt'):\n",
    "    for line in open(t):\n",
    "        p = line.strip().split()\n",
    "        if p: dist[int(p[0])] += 1\n",
    "for cid, cnt in sorted(dist.items()):\n",
    "    print(f'  class {cid} ({CLASS_NAMES.get(cid,\"?\")}): {cnt} boxes')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4 — Visual Check of Labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "CLASS_COLORS = {0: 'red', 1: 'dodgerblue', 2: 'lime'}\n",
    "CLASS_NAMES  = {0: 'defect', 1: 'label', 2: 'water'}\n",
    "\n",
    "fixed_imgs = sorted(glob.glob(f'{FIXED_IMAGES}/*.*'))\n",
    "samples = random.sample(fixed_imgs, min(6, len(fixed_imgs)))\n",
    "fig, axes = plt.subplots(2, 3, figsize=(16, 10))\n",
    "axes = axes.flatten()\n",
    "for i, img_path in enumerate(samples):\n",
    "    img = np.array(Image.open(img_path).convert('RGB'))\n",
    "    h, w = img.shape[:2]\n",
    "    ax = axes[i]\n",
    "    ax.imshow(img)\n",
    "    stem = os.path.splitext(os.path.basename(img_path))[0]\n",
    "    lbl  = f'{FIXED_LABELS}/{stem}.txt'\n",
    "    if os.path.exists(lbl):\n",
    "        with open(lbl) as f:\n",
    "            for line in f:\n",
    "                p = line.strip().split()\n",
    "                if len(p)==5:\n",
    "                    cls_id = int(p[0])\n",
    "                    cx,cy,bw,bh = map(float, p[1:])\n",
    "                    x1=int((cx-bw/2)*w); y1=int((cy-bh/2)*h)\n",
    "                    x2=int((cx+bw/2)*w); y2=int((cy+bh/2)*h)\n",
    "                    color = CLASS_COLORS.get(cls_id, 'yellow')\n",
    "                    label = CLASS_NAMES.get(cls_id, str(cls_id))\n",
    "                    ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,\n",
    "                        linewidth=2,edgecolor=color,facecolor='none'))\n",
    "                    ax.text(x1,max(0,y1-5),label,color='white',fontsize=8,\n",
    "                        bbox=dict(facecolor=color,alpha=0.7,pad=1))\n",
    "    ax.set_title(os.path.basename(img_path), fontsize=7)\n",
    "    ax.axis('off')\n",
    "from matplotlib.patches import Patch\n",
    "legend_elements = [Patch(facecolor=c, label=CLASS_NAMES[i]) for i,c in CLASS_COLORS.items()]\n",
    "fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,\n",
    "           title='Classes', title_fontsize=12, framealpha=0.9)\n",
    "plt.suptitle('Fixed Labels — 3 Classes: defect | label | water', fontsize=13, fontweight='bold')\n",
    "plt.tight_layout(rect=[0, 0.06, 1, 1])\n",
    "plt.show()\n",
    "print('Red=defect | Blue=label | Green=water')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5 — Augmentation (70 -> 700+ images)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "AUG_IMAGES = '/content/augmented/images'\n",
    "AUG_LABELS = '/content/augmented/labels'\n",
    "os.makedirs(AUG_IMAGES, exist_ok=True)\n",
    "os.makedirs(AUG_LABELS, exist_ok=True)\n",
    "\n",
    "transform = A.Compose([\n",
    "    A.HorizontalFlip(p=0.5),\n",
    "    A.VerticalFlip(p=0.3),\n",
    "    A.Rotate(limit=20, p=0.5),\n",
    "    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),\n",
    "    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),\n",
    "    A.GaussNoise(var_limit=(10,50), p=0.4),\n",
    "    A.GaussianBlur(blur_limit=3, p=0.3),\n",
    "    A.CLAHE(clip_limit=2.0, p=0.3),\n",
    "    A.RandomShadow(p=0.2),\n",
    "    A.Perspective(scale=(0.02,0.08), p=0.3),\n",
    "    A.RandomScale(scale_limit=0.2, p=0.3),\n",
    "], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.4))\n",
    "\n",
    "AUGMENT_PER_IMAGE = 10\n",
    "total    = 0\n",
    "src_imgs = sorted(glob.glob(f'{FIXED_IMAGES}/*.*'))\n",
    "\n",
    "for img_path in src_imgs:\n",
    "    stem     = os.path.splitext(os.path.basename(img_path))[0]\n",
    "    lbl_path = f'{FIXED_LABELS}/{stem}.txt'\n",
    "    img = cv2.imread(img_path)\n",
    "    if img is None: continue\n",
    "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "    bboxes, class_labels = [], []\n",
    "    if os.path.exists(lbl_path):\n",
    "        with open(lbl_path) as f:\n",
    "            for line in f:\n",
    "                p = line.strip().split()\n",
    "                if len(p)==5:\n",
    "                    class_labels.append(int(p[0]))\n",
    "                    bboxes.append([float(x) for x in p[1:5]])\n",
    "    if not bboxes: continue\n",
    "    shutil.copy(img_path,  f'{AUG_IMAGES}/{stem}_orig.jpg')\n",
    "    shutil.copy(lbl_path,  f'{AUG_LABELS}/{stem}_orig.txt')\n",
    "    for k in range(AUGMENT_PER_IMAGE):\n",
    "        try:\n",
    "            aug = transform(image=img, bboxes=bboxes, class_labels=class_labels)\n",
    "            if not aug['bboxes']: continue\n",
    "            name = f'{stem}_aug{k:03d}'\n",
    "            cv2.imwrite(f'{AUG_IMAGES}/{name}.jpg',\n",
    "                cv2.cvtColor(aug['image'], cv2.COLOR_RGB2BGR))\n",
    "            with open(f'{AUG_LABELS}/{name}.txt','w') as f:\n",
    "                for c,box in zip(aug['class_labels'], aug['bboxes']):\n",
    "                    f.write(f'{c} {\" \".join([f\"{v:.6f}\" for v in box])}\\n')\n",
    "            total += 1\n",
    "        except: pass\n",
    "\n",
    "print(f'Original : {len(src_imgs)}')\n",
    "print(f'Generated: {total}')\n",
    "print(f'Total    : {len(src_imgs) + total}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 6 — Split (Train 70% / Val 20% / Test 10%)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
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
   "metadata": {},
   "source": [
    "## Step 7 — Create YAML Config"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
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
   "metadata": {},
   "source": [
    "## Step 8 — Train YOLO26"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = YOLO('yolo26s.pt')\n",
    "results = model.train(\n",
    "    data=YAML_PATH, epochs=150, imgsz=640, batch=16,\n",
    "    patience=30, lr0=0.01, lrf=0.001, momentum=0.937,\n",
    "    weight_decay=0.0005, warmup_epochs=5,\n",
    "    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,\n",
    "    degrees=15.0, translate=0.1, scale=0.5,\n",
    "    flipud=0.3, fliplr=0.5, mosaic=1.0,\n",
    "    mixup=0.1, copy_paste=0.1,\n",
    "    project='/content/runs', name='cable_defect_final',\n",
    "    save=True, plots=True, verbose=True\n",
    ")\n",
    "print('Training complete!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 9 — Evaluate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "BEST_WEIGHTS = '/content/runs/cable_defect_final/weights/best.pt'\n",
    "best_model   = YOLO(BEST_WEIGHTS)\n",
    "metrics = best_model.val(data=YAML_PATH, split='test',\n",
    "                          imgsz=640, conf=0.35, iou=0.5, plots=True)\n",
    "print('\\n===== MODEL PERFORMANCE (3-class) =====')\n",
    "print(f'mAP@0.5      : {metrics.box.map50:.4f}')\n",
    "print(f'mAP@0.5:0.95 : {metrics.box.map:.4f}')\n",
    "print(f'Precision    : {metrics.box.mp:.4f}')\n",
    "print(f'Recall       : {metrics.box.mr:.4f}')\n",
    "CLASS_NAMES_LIST = ['defect', 'label', 'water']\n",
    "if hasattr(metrics.box, 'ap_class_index'):\n",
    "    print('\\n--- Per-class AP@0.5 ---')\n",
    "    for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):\n",
    "        print(f'  {CLASS_NAMES_LIST[idx]:8s}: {ap:.4f}')\n",
    "print('=======================================')\n",
    "if metrics.box.map50 >= 0.85:\n",
    "    print('Model is production ready!')\n",
    "elif metrics.box.map50 >= 0.70:\n",
    "    print('Good — consider more epochs or data.')\n",
    "else:\n",
    "    print('Needs improvement — check label quality in Step 4.')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 10 — Training Curves"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('/content/runs/cable_defect_final/results.csv')\n",
    "df.columns = df.columns.str.strip()\n",
    "fig, axes = plt.subplots(2, 3, figsize=(18,10))\n",
    "plots = [\n",
    "    ('train/box_loss','Train Loss','blue'),\n",
    "    ('val/box_loss','Val Loss','orange'),\n",
    "    ('metrics/mAP50','mAP@0.5','green'),\n",
    "    ('metrics/mAP50-95','mAP@0.5:0.95','purple'),\n",
    "    ('metrics/precision','Precision','red'),\n",
    "    ('metrics/recall','Recall','teal')\n",
    "]\n",
    "for ax,(col,title,color) in zip(axes.flatten(), plots):\n",
    "    if col in df.columns:\n",
    "        ax.plot(df['epoch'], df[col], color=color, linewidth=2)\n",
    "        ax.set_title(title, fontweight='bold')\n",
    "        ax.set_xlabel('Epoch')\n",
    "        ax.grid(alpha=0.3)\n",
    "plt.suptitle('Training Curves', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('/content/curves.png', dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 11 — Predict on Test Images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "CLASS_NAMES = ['defect', 'label', 'water']\n",
    "test_images = glob.glob(f'{SPLIT_DIR}/images/test/*.jpg')[:8]\n",
    "preds = best_model.predict(source=test_images, conf=0.35, iou=0.45, imgsz=640)\n",
    "fig, axes = plt.subplots(2, 4, figsize=(22,10))\n",
    "axes = axes.flatten()\n",
    "for i, result in enumerate(preds[:8]):\n",
    "    axes[i].imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))\n",
    "    if result.boxes and len(result.boxes.conf) > 0:\n",
    "        c   = result.boxes.conf.cpu().numpy()\n",
    "        cls = result.boxes.cls.cpu().numpy().astype(int)\n",
    "        summary = '  '.join([f'{CLASS_NAMES[k]}:{v:.0%}' for k,v in zip(cls,c)])\n",
    "        axes[i].set_title(summary, fontsize=8, color='red', fontweight='bold')\n",
    "    else:\n",
    "        axes[i].set_title('No detection', fontsize=9, color='green')\n",
    "    axes[i].axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 12 — Test Your Own Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "CLASS_NAMES = ['defect', 'label', 'water']\n",
    "\n",
    "# ── Confidence Boost Settings ────────────────────────────────────────────────\n",
    "CONF_THRESHOLD = 0.50   # raise (0.35-0.70) to reduce false positives\n",
    "IOU_THRESHOLD  = 0.40   # lower = fewer overlapping boxes\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "\n",
    "def predict_single(image_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD):\n",
    "    result = best_model.predict(\n",
    "        source=image_path, conf=conf, iou=iou,\n",
    "        imgsz=640, augment=True  # TTA for higher confidence\n",
    "    )[0]\n",
    "    plt.figure(figsize=(11, 8))\n",
    "    plt.imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))\n",
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
    "                  f'avg: {cls_confs.mean():.1%}  max: {cls_confs.max():.1%}')\n",
    "        print('─────────────────────────────────────────')\n",
    "        for j, (c, cid) in enumerate(zip(confs, clses)):\n",
    "            print(f'  [{j+1}] {CLASS_NAMES[cid]:8s}  {c:.1%}')\n",
    "    else:\n",
    "        plt.title('No Detection (try lowering conf threshold)',\n",
    "                  color='green', fontsize=13, fontweight='bold')\n",
    "        print(f'No detections above conf={conf:.0%}.')\n",
    "        print('Try: predict_single(path, conf=0.30)')\n",
    "    plt.axis('off'); plt.tight_layout(); plt.show()\n",
    "\n",
    "# Test on a sample from test set\n",
    "predict_single(glob.glob(f'{SPLIT_DIR}/images/test/*.jpg')[0])\n",
    "\n",
    "# Upload your own image:\n",
    "# u = files.upload(); predict_single(list(u.keys())[0])\n",
    "\n",
    "# Tune confidence:\n",
    "# predict_single('img.jpg', conf=0.40)  # stricter\n",
    "# predict_single('img.jpg', conf=0.30)  # more sensitive\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 13 — Export & Download"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model.export(format='onnx', imgsz=640, simplify=True)\n",
    "EXPORT = '/content/export'\n",
    "os.makedirs(EXPORT, exist_ok=True)\n",
    "shutil.copy(BEST_WEIGHTS, f'{EXPORT}/best.pt')\n",
    "shutil.copy(YAML_PATH,    f'{EXPORT}/dataset.yaml')\n",
    "onnx = BEST_WEIGHTS.replace('best.pt','best.onnx')\n",
    "if os.path.exists(onnx): shutil.copy(onnx, f'{EXPORT}/best.onnx')\n",
    "shutil.make_archive('/content/cable_model_final', 'zip', EXPORT)\n",
    "files.download('/content/cable_model_final.zip')\n",
    "print('Done!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 14 — Batch Live Demo (Upload Multiple Images)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "CLASS_NAMES    = ['defect', 'label', 'water']\n",
    "CONF_THRESHOLD = 0.50\n",
    "IOU_THRESHOLD  = 0.40\n",
    "\n",
    "print('Upload one or more images for batch prediction...')\n",
    "uploaded_batch = files.upload()\n",
    "\n",
    "all_results = []\n",
    "for fname in uploaded_batch:\n",
    "    r = best_model.predict(\n",
    "        source=fname, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,\n",
    "        imgsz=640, augment=True\n",
    "    )[0]\n",
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
    "        ax.set_title(f'{os.path.basename(fname)}\\nNo Detection',\n",
    "                     fontsize=8, color='green')\n",
    "        print(f'  {fname}: No detection above {CONF_THRESHOLD:.0%}')\n",
    "    ax.axis('off')\n",
    "for ax in axes_flat[n:]: ax.axis('off')\n",
    "print('==============================================')\n",
    "plt.suptitle(f'Batch Results — conf>={CONF_THRESHOLD:.0%} (defect|label|water)',\n",
    "             fontsize=13, fontweight='bold')\n",
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
 "nbformat_minor": 4
}
