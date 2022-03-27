---
layout: post
title: TabNet, Attentive Interpretable Tabular Learning (1) [kor]
date: 2022-03-20
category: Paper-review
toc: true
toc_sticky: true
toc_label: "Tabnet review"
layout: single
author_profile: true
---

<br>
Google Cloud AI 가 2020 ICLR 에서 발표한 정형 데이터를 위한 딥러닝 네트워크에 관한 논문이다. 
논문 주소 : [https://arxiv.org/abs/1908.07442](https://arxiv.org/abs/1908.07442#)
{: .notice--primary}
<br> <br>


## Introduction
---


&nbsp;&nbsp;&nbsp; Tabular data 에서는 딥러닝 보다 tree 기반 앙상블 모델이 성능이 더 우수하다고 한다. 앙상블 모델은 1) tabular data 에 효율적임 2) Highly-interpretable 3) 학습이 빠른 장점이 있는 반면, 딥러닝은 over-parameterized 되어 tabular data 에 적합하지 않다. <br>

&nbsp;&nbsp;&nbsp;그럼에도 불구하고 저자는 딥러닝을 tabular data 에 적용할만한 가치가 있다고 주장하는데, 그 이유는 

1) 학습 데이터가 많으면 성능이 향상되고 multi-modal learning 이 가능

2) Feature engineering 과 같은 단계를 크게 요구하지 않음 

3) Domain adaptation, Generative modeling, Semi-supervised learning 과 같은 기법 적용 가능

하기 때문이다.

&nbsp;&nbsp;&nbsp;Tabnet 의 contribution 은 다음과 같이 크게 4가지가 있다.

1) 전처리 없이 raw tabular data 를 input 으로 사용 가능하며, gradient-descent 기반 end-to-end learning 이 가능

2) Sequential attention mechanism 을 사용해 instance-wise feature selection 으로 interpretability 향상

3) 성능이 향상되고 local & global interpretability 을 제공

4) Self-supervised learning 사용 <br> <br>

## Tabnet architecture
---

### Encoder

<img src="/assets/images/Tabnet1/Untitled.png" height="400"/> {: .align-center} <br>

&nbsp;&nbsp;&nbsp;전체적인 encoder 구조는 다음과 같다. 보라색 박스의 Step 을 N 번 반복하는 구조이다. 한 Step 에서는 크게 feature importance mask 를 생성하는 Attentive transformer 와 Feature importance 를 고려해 feature 를 encoding 하는 Feature transformer 의 모듈이 있다. 

&nbsp;&nbsp;&nbsp;각 Step 의 input 과 output, 그리고 flow 를 조금 더 자세히 살펴보면,

<img src="/assets/images/Tabnet1/Untitled1.png" height="400"/> {: .align-center} <br>

&nbsp;&nbsp;&nbsp; 우선 `step i-1 의 encoded feature` 가 attentive transformer 를 통과해 `step 0 의 feature` 와 함께 Mask 를 생성하는데 사용된다. 이렇게 생성된 Mask 는 Feature transformer 를 통과하여 1) `step i 의 encoded feature` 가 되어 다음 step 에 사용되고, 2) `step i-1 까지의 누적된 feature` 들과 더해져서 `최종 output 을 위한 feature` 를 만들어나간다. 그리고 `step i 까지의 누적된 feature` 는 Mask 와 곱한 후 `step i-1 까지의 누적된 feature importance` 과 더해져 최종 feature attribute 를 만들어나간다. 

&nbsp;&nbsp;&nbsp; 이제 각 Attentive transformer 와 Feature transformer, 그리고 Feature attribute 생성 과정 의 구조를 살펴보자. <br> <br>

**Attentive transformer** 

&nbsp;&nbsp;&nbsp; Attentive transformer 를 풀어서 그리게 되면 아래 화살표 오른쪽과 같이 표현할 수 있다.

<img src="/assets/images/Tabnet1/Untitled2.png" height="200"/> <br>

&nbsp;&nbsp;&nbsp; 이 모듈을 설명하는 수식은 ![equation](https://latex.codecogs.com/svg.image?M[i]=sparsemax(P[i-1]\cdot&space;h_{i}(a[i-1])) ) 이고, 이때  ![equation](https://latex.codecogs.com/svg.image?M[i] )&nbsp; 은 해당 i 번째 step 의 Mask, ![equation](https://latex.codecogs.com/svg.image?a[i-1] )&nbsp;  은 i-1 번째 Feature transformer 의 출력, ![equation](https://latex.codecogs.com/svg.image?P[i-1] )&nbsp; 은 prior scale term, 그리고 ![equation](https://latex.codecogs.com/svg.image?h_{i} )&nbsp; 는 FC layer + BN으로 이루어진 neural net layer 를 의미한다. 

&nbsp;&nbsp;&nbsp; 우선 이전 step 의 Encoded feature (![equation](https://latex.codecogs.com/svg.image?a[i-1] ) )&nbsp; 가 input 으로 들어오면 ![equation](https://latex.codecogs.com/svg.image?h_{i} )&nbsp; 를 통과한 후 prior scale term 이라 불리는 ![equation](https://latex.codecogs.com/svg.image?P[i-1] )&nbsp; 과 곱해지고 sparsemax 를 통과하면 i 번째 step 의 Mask 가 생성된다. 이때 output 인 Mask 는 어떤 feature 를 주로 사용할 것인지에 대한 Mask (feature importance mask)를 의미한다. prior scale term 은 어떤 feature 를 위주로 처리할 것 인지 ( relaxation factor인 γ로 조절 ) 에 대한 정보를 부여하는 것인데, i 번째 새로운 Mask 에 이전 Mask 의 정보를 부여해주는 과정이다. 마지막을 sparsemax 는 softmax 의 sparser version 이라고 생각할 수 있는데, sparsemax activation 을 사용하게 되면 많은 dimension 이 0 값을 가지기 때문에 instance-wise feature selection 이 가능해진다. <br> <br>

**Feature transformer**

<img src="/assets/images/Tabnet1/Untitled3.png" height="200"/><br>

&nbsp;&nbsp;&nbsp; Feature transformer 는 4 개의 GLU block (FC + BN + GLU) 으로 구성되어 있는데, 이 중 2 개는 shared, 2 개는 step dependent 로 weight 가 반영된다. input 은 feature importance ![equation](https://latex.codecogs.com/svg.image?M[i] )&nbsp; 가 반영된 Step 0 의 feature ![equation](https://latex.codecogs.com/svg.image?M[i]\cdot&space;f)&nbsp; 이고, output 은 feature importance 를 고려한 encoded feature ![equation](https://latex.codecogs.com/svg.image?f_{i}(M[i]\cdot&space;f))&nbsp; 가 된다. 그 후 ![equation](https://latex.codecogs.com/svg.image?d[i])&nbsp; , ![equation](https://latex.codecogs.com/svg.image?a[i])&nbsp; 로 나뉘어 하나는 최종 output 생성에, 하나는 다음 step 의 Mask 생성에 사용된다. (여기서 헷갈렸던 부분이 왜 갑자기, 그리고 어떤 방식으로 output 을 ![equation](https://latex.codecogs.com/svg.image?d[i])&nbsp; , ![equation](https://latex.codecogs.com/svg.image?a[i])&nbsp; 로 나누는지 였는데, 코드를 확인해보니 split 하는게 아니라 같은 결과를 path 2개로 넣어준다는 의미였다.) <br> <br>

**Feature attribute (Interpretability)**

<img src="/assets/images/Tabnet1/Untitled4.png" height="400"/><br>

&nbsp;&nbsp;&nbsp; 해당 부분은 feature attribute 를 구하는 과정이다. i 번째 step 의 Mask 에 Feature transformer 를 거친 encoded feature ![equation](https://latex.codecogs.com/svg.image?d[i])&nbsp; 를 이용한 aggregate decision contribution ![equation](https://latex.codecogs.com/svg.image?\eta_{b}[i])&nbsp; 를  scaling 으로 곱해주면 해당 step 의 feature importance 가 생성된다. 그리고 전체 step 의 feature attribute 는 각 step 의 feature importance 를 누적시켜 다음과 같은 수식을 통해 산출된다.

<img src="/assets/images/Tabnet1/Untitled5.png" height="30"/> <br> <br>

### Decoder

<img src="/assets/images/Tabnet1/Untitled6.png" height="300"/><br>

&nbsp;&nbsp;&nbsp; Decoder 는 encoder 파트에서 설명한 feature transformer 와 FC layer 로 구성되어있다. Decoder 는 self-supervised learning 을 통해 학습이 이루어지는데, self-supervised learning 의 과정은 다음과 같다.

1. Pretext task (연구자가 직접 만든 task) 를 정의 : 이 논문에서는 결측치 예측 task 가 된다.
2. unsupervised pre-training 을 진행한다.
3. 2.에서 학습시킨 모델을 downstream task 로 transfer learning (supervised fine-tuning) 을 진행한다. <br> <br>

## Conclusion
---


&nbsp;&nbsp;&nbsp; 정리하자면, Tabnet 은 tabular learning 을 위한 deep learning architecture 이다. Sequential attention mechanism 을 도입함으로써 decision 단계에서 feature 들만 사용하도록 선택해준다. 또한 Instance-wise feature selection 이 가능하기 때문에 model capacity 가 중요한 feature 에 대부분 집중될 수 있도록 해주고 feature importance 의 시각화로 interpretable decision making 이 가능하다. 마지막으로 self-supervised learning 을 도입해 pre-training 단계에서 un-supervised learning 으로 performance 를 향상시켜준다.
