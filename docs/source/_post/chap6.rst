.. _Chapter6:

***************************
Chapter 6. 통계 기반 머신러닝 1 - 확률분포와 모델링
***************************

이 문서는 한빛미디어에서 나온 ``처음 배우는 인공지능`` 을 공부하면서 정리한 것이다.

01 통계 모델과 확률분포
######################

확률기반
********

`확률분포 <https://ko.wikipedia.org/wiki/%ED%99%95%EB%A5%A0%EB%B6%84%ED%8F%AC>`_ 란 확률변수가 특정한 값을 가질 확률을 나타내는 함수를 의미합니다.

* 연속 확률분포 : 확률 밀도 함수를 이용해 표현할 수 있는 확률분포를 의미합니다. 정규분포, 연속균등분포, 카이제곱분포 등이 있습니다.

* 이산 확률분포 : 이산 확률변수가 가지는 확률분포를 의미합니다. 이산균등분포, 푸아송분포, 베르누이분포, 기하분포,초기하분포,이항분포 등이 있습니다.


머신러닝
********

머신러닝이란 "기계가 학습한다"는 개념을 의미하는 용어로, 입력 데이터의 특성과 분포, 경향 등에서 자동으로 데이터를 나누거나 재구성 하는 것을 의미합니다.

* 지도학습(Supervised Learning) : 데이터에 정답 정보가 결합된 학습 데이터(또는 훈련 데이터)로 데이터의 특징을 모델링하는 과정을 의미합니다. 주로 식별과 예측 등을 목적으로 둘 때가 많으므로 데이터를 선형 결합으로 나타내려는 특성을 이용합니다.

* 자율학습(Unsupervised Learning) : 입력 데이터의 정답을 모르는 상태에서 사용하는 것으로 클러스터 분석, 차원압축, 밀도추정 등이 해당합니다.

.. image:: imgs/머신러닝_types.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text


기저함수
********

`기저함수 <https://ko.wikipedia.org/wiki/%EA%B8%B0%EC%A0%80_%ED%95%A8%EC%88%98>`_ 란 서로 직교하면서 선형적으로 독립적인 함수의 집합을 의미합니다. 기저함수를 구성하는 설명 변수(기저벡터)들은 선형 독립이어야 합니다.

주요 기저함수
*************

기저함수는 확률분포 모델에 따라 연속 확률분포와 이산 확률분포로 분류됩니다.

1. 연속 확률분포

* 정규분포

`정규분포 <https://ko.wikipedia.org/wiki/%EC%A0%95%EA%B7%9C%EB%B6%84%ED%8F%AC>`_ 는 가장 많이 사용하는 분포 개념으로 가우스 분포(Gaussian distribution)라고도 합니다. 실험의 측정 오차나 사회 현상 등 자연계의 현상은 정규분포를 따르는 경향이 있습니다.

.. image:: imgs/정규분포_식.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        
.. image:: imgs/정규분포_그래프.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text


* 감마분포

`감마분포 <https://ko.wikipedia.org/wiki/%EA%B0%90%EB%A7%88_%EB%B6%84%ED%8F%AC>`_ 는 특정 수의 사건이 일어날 때까지 걸리는 시간에 관한 연속 확률분포 입니다. 모양 매개변수가 k이고 크기 매개변수가 θ일 때, 평균은 kθ, 분산은 kθ^2입니다. k=1 일 때 지수분포, k가 반정수((2n-1)/2)고 θ=2 일 때 카이제곱 분포라고 합니다.

.. image:: imgs/감마분포_식.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        
.. image:: imgs/감마분포_그래프.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text


* 지수분포

`지수분포 <https://ko.wikipedia.org/wiki/%EC%A7%80%EC%88%98%EB%B6%84%ED%8F%AC>`_ 는 감마분포의 모양 매개변수 k=1 일 때 사건이 일어나는 시간 간격의 확률분포를 의미합니다. 푸아송 분포와도 깊은 연관이 있습니다. 

.. image:: imgs/지수분포_식.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        
.. image:: imgs/지수분포_그래프.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text


* 베타분포

`베타분포 <https://ko.wikipedia.org/wiki/%EB%B2%A0%ED%83%80_%EB%B6%84%ED%8F%AC>`_ 는 2개의 변수를 갖는 특수 함수인 베타함수를 이용한 분포입니다. 매개변수 a, b를 바꾸면 다양한 분포를 나타낼 수 있으므로 베이즈 통계학에서는 사전분포 모델로 이용할 때가 많습니다.

.. image:: imgs/베타분포_식.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        
.. image:: imgs/베타분포_그래프.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        
        
* 디리클레 분포

`디리클레 분포(Dirichlet distribution) <https://ko.wikipedia.org/wiki/%EB%94%94%EB%A6%AC%ED%81%B4%EB%A0%88_%EB%B6%84%ED%8F%AC>`_ 는 베타분포를 다변량으로 확장한 것으로 다변량 베타분포라고도 합니다. 연속함수지만 2차원 평면에서는 연속함수로 나타낼 수 없습니다. 확률자체를 확률분포로 두는 분포로 자연어 처리 등에 많이 사용합니다.

.. image:: imgs/디리클레분포_식.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        
.. image:: imgs/디리클레분포_그래프.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text


2. 이산 확률분포

* 이항분포

`이항분포 <https://ko.wikipedia.org/wiki/%EC%9D%B4%ED%95%AD_%EB%B6%84%ED%8F%AC>`_ 란 베르누이 시행(두 가지 종류 중 어느 하나가 나오는가와 같은 실험)을 여러 번 시행했을 때 확률분포를 의미합니다. 정규분포나 포아송분포와 비슷합니다.

.. image:: imgs/이항분포_식.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        
.. image:: imgs/이항분포_그래프.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text


* 음이항분포

`음이항분포 <https://en.wikipedia.org/wiki/Negative_binomial_distribution>`_ 란 r번 성공하는데 필요한 시행횟수 k의 분포를 의미합니다. 생명과학 분야에서 많이 사용합니다.

.. image:: imgs/음이항분포_식.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        
.. image:: imgs/음이항분포_그래프.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text


* 푸아송분포

`푸아송분포 <https://ko.wikipedia.org/wiki/%ED%91%B8%EC%95%84%EC%86%A1_%EB%B6%84%ED%8F%AC>`_ 란 일정 시간 간격에 걸쳐 평균 λ번 일어나는 현상이 k번 발생할 확률분포를 의미합니다. 지수분포가 사건이 일어난 후 다시 발생할 때까지의 시간 간격에 대한 확률밀도를 나타내는 반면, 푸아송분포는 단위 시간에 사건이 일어날 확률을 나타내므로 이 둘은 같은 사건의 발생 확률을 다른 측면에서 본다고 이해할 수 있습니다.

.. image:: imgs/푸아송분포_식.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        
.. image:: imgs/푸아송분포_그래프.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text


* 카이제곱분포
`카이제곱분포 <https://ko.wikipedia.org/wiki/%EC%B9%B4%EC%9D%B4%EC%A0%9C%EA%B3%B1_%EB%B6%84%ED%8F%AC>`_ 란 집단을 몇 가지로 나눴을 때 크기가 작은 집단에 보편성이 있는지 확인할 수 있는 분포입니다. 통계적 추론에서는 카이제곱 검정(독립성 검정)으로 자주 이용하며, 임상시험이나 사회과학 설문조사 등에 자주 사용합니다.

.. image:: imgs/카이제곱분포_식.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        
.. image:: imgs/카이제곱분포_그래프.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text


* 초기하분포
`초기하분포 <https://en.wikipedia.org/wiki/Hypergeometric_distribution>`_ 란 반복하지 않는 시도에서 사건이 발생할 확률분포를 의미합니다.

.. image:: imgs/초기하분포_식.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        
.. image:: imgs/초기하분포_그래프.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
      
      
* 로지스틱분포
`로지스틱분포 <https://en.wikipedia.org/wiki/Logistic_distribution>`_ 란 확률분포의 확률변수가 특정 값보다 작거나 같은 확률을 나타내는 누적분포 함수가 로지스틱 함수인 분포를 의미합니다. 정규분포와 비슷하지만 그래프의 아래가 길어 평균에서 멀어지더라도 정규분포처럼 곡선이 내려가지 않습니다.

.. image:: imgs/로지스틱분포_식.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        
.. image:: imgs/로지스틱분포_그래프.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        

* 확률분포 정리

.. image:: imgs/분포정리_그림.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text
        

손실함수와 경사 하강법
*********************

* 손실함수
생성된 모델을 통해 도출한 결과 값과 기대하던 값 사이 오차를 계산하는 함수입니다. 이 오차를 제곱한 값의 합계가 최솟값이 되도록 구하는 과정은 경사 하강법을 사용합니다.

1. 경사 하강법(Gradient descent method)

`경사 하강법 <https://ko.wikipedia.org/wiki/%EA%B2%BD%EC%82%AC_%ED%95%98%EA%B0%95%EB%B2%95>`_ 은 손실함수 위 한 점 w1에서 편미분 한 결과(기울기)가 0이 될 때까지 w의 위치에 변화를 주면서 손실 함수의 최솟값을 찾아 나가는 방법입니다. 경사 혹은 기울기는 x-y 평면에서 두 변수의 관계를 설명해 줍니다. 우리는 신경망의 오차와 각 계수의 관계에 관심이 있습니다. 즉, 각 계수의 값을 증가 혹은 감소시켰을 때 신경망의 오차가 어떻게 변화하는지 그 관계에 주목합니다. 모델의 학습은 모델의 계수를 업데이트해서 오차를 줄여나가는 과정입니다.


.. image:: imgs/경사하강법_그래프.png
        :width: 500px
        :align: center
        :height: 500px
        :alt: alternate text





02 베이즈 통계학과 베이즈 추론
############################
베이즈 정리
**********

최대가능도 방법과 EM 알고리즘
***************************

베이즈 추론
**********

베이즈 판별분석
***************



03 마르코프 연쇄 몬테카를로 방법
##############################




04 은닉 마르코프 모델과 베이즈 네트워크
#####################################
