# pipeline_resnet50.py

import kfp
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, component
from typing import Annotated
import mlflow
import mlflow.pytorch
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

@component(packages_to_install=["torch", "torchvision", "mlflow"])
def train_model(mlflow_tracking_uri: str, dataset_path: str, model_output: Output[Model]):
    import os
    import mlflow
    import mlflow.pytorch
    import torch
    import torchvision.models as models
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import ToTensor, Compose, Resize
    from torch.utils.data import DataLoader
    from torch import nn, optim

    # Set MLflow Tracking
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("defeito-visao-computacional")

    with mlflow.start_run():
        # Load dataset
        transform = Compose([Resize((224, 224)), ToTensor()])
        dataset = ImageFolder(root=dataset_path, transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Modelo ResNet pré-treinado
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

        # Treinamento simples
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = 1  # Para demo rápida

        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        
        # Log metrics e modelo
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("model", "resnet50")
        mlflow.log_metric("loss", running_loss)

        # Salva o modelo
        model_path = os.path.join(model_output.path, "model")
        mlflow.pytorch.save_model(model, model_path)
        print(f"Modelo salvo em: {model_path}")

@dsl.pipeline(name="pipeline-visao-computacional")
def defect_detection_pipeline(
    dataset_path: str = "/mnt/data/dataset-imagens/",
    mlflow_tracking_uri: str = "http://mlflow-server:5000"
):
    train_model_task = train_model(
        mlflow_tracking_uri=mlflow_tracking_uri,
        dataset_path=dataset_path
    )

