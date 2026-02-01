# Freshness Classifier

A Fruit/vegetable Freshness Classifier to identify the fruit/vegetable from the mentioned list of items, predict the freshness, and identify the mold level by uploading the image.

Live Demo: [Link to Live Demo](http://46.225.56.25:30800/)

Live DashBoard: [Dashboard](http://46.225.56.25:30300/d/d04fd85e-9659-4eb8-b5fe-558c854ac2c5/freshness-classifier?orgId=1&from=now-2d&to=now)


### Tech Stack

Following Tech stack was Utilized:

#### Data Storage

- Azure Blob Storage

#### Model Architecture

- Python
- Pytorch
- ResNet50
- ONNX

#### Model Training

- Azure ML Studio

#### Hosting

- Hetzner Cloud
- Frontend (HTML/CSS)
- Backend & API (uvicorn & FastAPI)
- Docker
- Kubernetes(K3's)
- Python
- Github

#### Monitoring & Observability

- Prometheus
- Grafana
- Custom Drift Detection Python script


## Project Structure
```
solutions/
├── src/
│   ├── training/
│   │   ├── model.py                    # Model architecture
│   │   ├── train.py                    # Training script
│   │   ├── test.py                     # Test model inference locally
│   │   ├── dataset.py                  # Data preprocessing, train-test split
│   │   └── conda_dependencies.yml      # Additional dependencies for cloud training
│   └── inference/
│       ├── classifier.py               # ONNX inference + monitoring + drift detection
│       └── inference.py                # ONNX inference CLI script
│
├── frontend/
│   └── index.html                      # Web UI for image upload
│
├── k3s/                                # Kubernetes manifests
│   ├── namespace.yml
│   ├── Deployment.yml
│   ├── svc.yml
│   ├── grafana-deployment.yaml
│   ├── grafana-svc.yaml
│   ├── prometheus-config.yaml
│   ├── prometheus-deployment.yaml
│   └── prometheus-svc.yaml
│
├── app.py                              # FastAPI REST API server
├── train-job-pipeline.py               # Azure ML training job submission
├── Dockerfile                          # Container definition
├── .env                                # Environment variables (not in repo)
├── .dockerignore                       # Docker build exclusions
├── .gitignore                          # Git exclusions
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```
