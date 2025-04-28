"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com

This script is used to train the UNet to predict the noise at different
timestamps of the diffusion process. The loss function is a simple mean
squared error between the actual noise and the predicted noise based on
the diffused image, according to the original paper: 
https://arxiv.org/pdf/2006.11239.pdf
"""

from tqdm import tqdm, trange

import torch
from torch import optim
from mnist_dataset import get_mnist_dataloader
from diffusion_model import DiffusionProcess

from models import ConditionalUNet

if __name__ == "__main__":
    # Prepare images 데이터 준비
    trainloader, testloader = get_mnist_dataloader()
    idx, (images, labels) = next(enumerate(testloader))

    # Prepare model and training
    device = "cpu" # 현재 CPU만 사용
    model = ConditionalUNet().to(device) # ConditionalUNet 모델 초기화
    process = DiffusionProcess() # Diffusion 과정 객체 초기화
    optimizer = optim.Adam(model.parameters(), lr=2e-4) # Adam 옵티마이저
    scheduler = optim.lr_scheduler.StepLR(optimizer, 80) # 80 에폭마다 러닝레이트 감소
    criterion = torch.nn.MSELoss()  # 손실 함수: 평균 제곱 오차 (MSE)

    # Training Loop
    epochs = 10 #너무 오래 돌아가서 epochs는 100에서 10으로 조정했습니다.
    for e in trange(epochs):
        running_loss = 0 # 에폭 동안 누적 손실
        progress_bar = tqdm(trainloader, leave=False) # 학습 진행 상황 bar로 표시
        for image, label in progress_bar:
            # Sampling t, epsilon, and diffused image
            # 노이즈 추가 및 Diffusion 적용
            t = torch.randint(0, 1000, (image.shape[0],)) # 각 샘플마다 랜덤 timestep t 선택 (0~999)
            epsilon = torch.randn(image.shape) # 노이즈 샘플링
            diffused_image = process.forward(image, t, epsilon) # 입력 이미지에 노이즈를 입힌 결과 생성

            # Backprop
            optimizer.zero_grad() # 옵티마이저 초기화
            output = model(diffused_image.to(device), t.to(device), label.to(device)) # 모델이 노이즈를 예측
            loss = criterion(epsilon.to(device), output) # 실제 노이즈(epsilon)와 예측 노이즈(output) 간 MSE 계산
            loss.backward() # 역전파
            optimizer.step() # 모델 파라미터 업데이트
            # 진행 표시
            loss_value = loss.cpu().item()
            running_loss += loss_value
            progress_bar.set_description(f"Loss: {loss_value:.4f}")
        # 학습률 조정
        scheduler.step()

        # Save model after every epoch 에폭마다 모델 저장
        torch.save(model.state_dict(), "unet_mnist.pth")

        # Logging results 에폭별 평균 손실 출력
        running_loss /= len(trainloader)
        tqdm.write(f"Mean loss for Epoch {e + 1}: {running_loss:.4f}")
