# Portfolio v2 — AI Vehicle Damage Detection

A cross-platform Flutter app (web, Android, iOS, macOS, Windows, Linux) that lets a user upload a photo of a vehicle and get back an AI-generated damage report: per-class segmentation masks, damage percentage, and severity, powered by a custom-trained semantic segmentation model.

This project doubles as a personal portfolio piece — it combines a trained computer vision model, a serverless inference API, and a full Flutter front end into one working product.

## How it works

```
Flutter app  →  Cloudflare Pages Function (/api/predict)  →  Modal webhook  →  AxisDeepLabV3+ model
   (UI)              (auth check + proxy)                      (GPU inference)      (segmentation)
```

1. The user uploads a vehicle photo in the app.
2. The image is POSTed to `/api/predict` with a Firebase auth token.
3. The Cloudflare Function (`functions/api/predict.js`) validates the request and forwards the image to a Modal-hosted inference endpoint.
4. The model (`AI_model/`) returns per-class damage percentages, a severity summary, and colour-coded mask overlays.
5. The app renders the overlay and a breakdown of detected damage (dents, scratches, corrosion, cracked/shattered glass, panel misalignment, etc.) on the Home page.

## Repository structure

| Path | Description |
|---|---|
| `lib/` | Flutter app source — navigation shell, sidebar, and pages (Home, About, Settings, Account, Vehicle Events) |
| `AI_model/` | Model training and inference code (`AxisDeepLabV3+`, a ConvNeXtV2-based DeepLabV3+ segmentation model) — config, dataset loading, loss functions, metrics, training loop, and prediction/CLI script |
| `functions/api/predict.js` | Cloudflare Pages Function that authenticates requests and proxies image uploads to the Modal inference endpoint |
| `android/`, `ios/`, `macos/`, `windows/`, `linux/`, `web/` | Platform-specific Flutter build targets |

## Features

- **AI damage detection** — 9-class segmentation (dent, scratch, corrosion, cracked glass, shattered glass, panel misalignment, fragmentation, panel crumpling, interior texture) with test-time augmentation and per-class confidence thresholds.
- **Cross-platform UI** — one Flutter codebase targeting web, mobile, and desktop, with a collapsible sidebar that adapts to screen width.
- **Firebase-backed** — anonymous auth, Firestore, and Cloud Messaging wired in for user/session tracking.
- **Serverless inference proxy** — the Flutter app never talks to the model server directly; a Cloudflare Function handles auth and forwarding, keeping the model endpoint private.
- **Vehicle Events** — a section for tracking vehicle checkups/history (in progress).

## Getting started

### Prerequisites

- [Flutter SDK](https://docs.flutter.dev/get-started/install) (Dart ^3.12.2)
- A Firebase project (Auth, Firestore, Messaging enabled)
- Node.js + [Wrangler](https://developers.cloudflare.com/workers/wrangler/) if you want to run/deploy the Cloudflare Function locally
- Python 3.10+, PyTorch, and Albumentations if you want to train or run the model yourself

### Run the app

```bash
flutter pub get
flutter run
```

### Configure Firebase

This project uses Firebase (Auth, Firestore, Messaging). Rather than hardcoding credentials, generate your own config with the FlutterFire CLI and point `main.dart` at it:

```bash
dart pub global activate flutterfire_cli
flutterfire configure
```

### Run the prediction API locally

```bash
cd functions
npm install
wrangler pages dev
```

Set `MODAL_WEBHOOK_URL` as an environment variable (Cloudflare dashboard, or a local `.dev.vars` file) pointing at your deployed Modal inference endpoint.

### Train / run the model

```bash
cd AI_model
pip install torch torchvision albumentations opencv-python numpy   # no requirements.txt yet — pin versions as needed
python train.py
python predict.py --input path/to/images --output results/ --checkpoint checkpoints/best.pt
```

See `AI_model/config.py` for all hyperparameters (encoder, loss weights, per-class thresholds, TTA settings).

## Tech stack

- **Frontend:** Flutter / Dart, Material 3
- **Backend:** Firebase (Auth, Firestore, Messaging), Cloudflare Pages Functions
- **ML:** PyTorch, ConvNeXtV2 encoder, DeepLabV3+ decoder, Modal for GPU-hosted inference
- **Data:** Roboflow-exported COCO-format annotations

## Status

Actively evolving — the Vehicle Events page and account features are still in progress. Contributions and issues are welcome.