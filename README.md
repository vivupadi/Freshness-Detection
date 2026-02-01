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


### Project Structure

solutions/
├── src/
│   └── training/
│       ├── model.py                  # Model architecture
│       ├── train.py                  # Training script
│       ├── test.py                   # Test model inference locally
│       ├── dataset.py                # Data preprocessing, train-test split
│       └── conda_dependencies.yml    # Additional dependencies for cloud training
│   └── inference/
│       ├── classifier.py             # ONNX inference methods + monitoring metrics + custom drift method
│       └── inference.py              # ONNX Inference script
├── frontend/
│       └── index.html                # Web UI
├── k3s/                                  # Lightweight Kubernetes yaml for container orchestration
│       ├── namespace.yml                 
│       ├── Deployment.yml                
│       ├── svc.yml                       
│       ├── grafana-deployment.yaml       
│       ├── grafana-svc.yaml              
│       ├── prometheus-config.yaml        
│       ├── prometheus-deployment.yaml    
│       └── prometheus-svc.yaml           
├── app.py                             # FastAPI server
├── train-job-pipeline.py              # Training pipeline to be run on Azure ML Studio
├── Dockerfile                         # Dockerfile to build image
├── .env                               # Environment variables
├── .dockerignore                      # files to be ignored during docker build
├── .gitignore                         # files to be ignored during git upload
├── requirements.txt                   # Dependencies
└── README.md
