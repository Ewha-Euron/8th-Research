"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com

This script performs the sampling given the trained UNet model
"""
from tqdm import trange

import torch
from models import ConditionalUNet
from diffusion_model import DiffusionProcess

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# TkAgg 백엔드를 사용해 외부 그림 창을 띄움
# PyCharm에서 처음에 그림이 안띄워져서 추가로 작성한 코드

if __name__ == "__main__":
    # Prepare model # 모델 준비
    device = "cpu" # CPU 사용
    batch_size = 100 # 한 번에 100장의 샘플을 생성
    model = ConditionalUNet().to(device) # ConditionalUNet 모델 생성
    model.load_state_dict(torch.load("unet_mnist.pth")) # 저장된 학습된 모델 가중치 불러오기
    process = DiffusionProcess() # Diffusion Process 객체 초기화

    # Sampling 초기 상태로부터 샘플링 시작
    xt = torch.randn(batch_size, 1, 28, 28)  # 완전히 노이즈만 있는 이미지 생성 (시작점)
    digit_to_sample = torch.Tensor([9]).to(dtype=torch.long).to(device)  # 생성할 숫자를 9로 설정

    # 모델을 평가 모드로 설정
    model.eval()
    with torch.no_grad(): # 그래디언트 계산 비활성화
        for t in trange(999, -1, -1): # timestep 999 → 0까지 거꾸로 진행 (reverse diffusion)
            time = torch.ones(batch_size) * t # 현재 timestep 정보를 batch_size만큼 생성
            et = model(xt.to(device), time.to(device), digit_to_sample)  # predict noise 현재 이미지에 대한 노이즈 예측
            xt = process.inverse(xt, et.cpu(), t) # 예측된 노이즈를 제거하면서 이미지 복원

    # 생성된 이미지 시각화
    labels = ["Generated Images"] * 9 # 타이틀용 라벨 설정

    for i in range(9):
        plt.subplot(3, 3, i + 1) # 3x3 그리드에 이미지 배치
        plt.tight_layout() # 레이아웃 자동 정리
        plt.imshow(xt[i][0], cmap="gray", interpolation="none") # 생성된 이미지 출력
        plt.title(labels[i])
    plt.show(block=True) # 창이 안떠서 강제로 그림 띄우는 코드로 수정
