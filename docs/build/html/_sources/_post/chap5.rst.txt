.. _Chapter5:

************************************
Chapter 5. 가중치와 최적화 프로그램
************************************

이 문서는 한빛미디어에서 나온 ``처음 배우는 인공지능`` 을 공부하면서 정리한 것이다.

.. _graph01:

01 그래프 이론
##############

그래프
******
`그래프 <https://ko.wikipedia.org/wiki/%EA%B7%B8%EB%9E%98%ED%94%84>`_ 란 점과 점을 연결하는 선으로 구성된 형태를 말합니다. 점은 정점 (Vertex) 또는 노드 (node) 라고 부르고 선은 간선 (edge)라고 부릅니다.

.. image:: imgs/chap5_1.png
	:width: 400px
	:align: center
	:height: 100px
	:alt: alternate text

무향 그래프와 유향 그래프
*************************
* 무향 그래프 (undirected graph): 그래프의 간선에 방향이 없는 그래프
* 유향 그래프 (directed graph): 그래프의 간선에 방향이 있는 그래프
* 유향 순회 그래프 (directed cyclic graph): 어떤 노드에서 자신으로 돌아오는 path가 있는 그래프 
* `유향 비순회 그래프 (directed acyclic graph) <https://en.wikipedia.org/wiki/Directed_acyclic_graph#Definitions>`_: cycle 이 없는 유향 그래프 (책의 설명이 좀 이상합니다. - 어떤 정점에서 출발한 후 해당 정점에 돌아오는 경로가 하나인 그래프)
* 간선 가중 그래프: 간선에 가중치가 있는 그래프 (aka network)
* 정점 가중 그래프: 정점에 가중치가 있는 그래프 (aka network)

그래프의 행렬 표현
******************
그래프를 나타내는 방법 중에서 행렬로 나타내는 방법을 소개합니다. 이 책에서는 인접행렬(adjacency matrix) 와 근접행렬(incidence matrix) 를 설명합니다.

* `인접행렬 <https://ko.wikipedia.org/wiki/%EC%9D%B8%EC%A0%91%ED%96%89%EB%A0%AC>`_: 정점 사이의 관계를 타나내는 행렬
* `근접행렬 <https://en.wikipedia.org/wiki/incidence_matrix>`_: 정점과 변의 관계를 타나내는 행렬 

.. image:: imgs/chap5_2.png
	:width: 500px
	:align: center
	:height: 100px
	:alt: alternate text

`트리 구조 그래프 <https://en.wikipedia.org/wiki/Tree_structure>`_
******************************************************************
트리 구조는 그래프구조의 하나로 조상이 없는 노드를 root 노드로 두고 그 아래로 자식관계를 가지는 후손 노드들이 펼쳐지는 구조이다. 이를 통해 하나의 정점에서 다른 정점으로 가는 경로가 단 한개만 존재한다. 

.. _graph02:

02 그래프 탐색과 최적화
#######################

탐색 트리 구축
**************
출발점에서 목적지까지를 노드로 정의하고 각 노드와 에지에 이익, 비용과 같은 평가 값을 저장해두고 목적지에 도달하는 최적 경로를 찾을 수 있도록 트리를 구축하는 것을 말하고자 합니다.
가장 대표적인 것이 데이터베이스의 인덱스에 사용하는 이진 탐색 트리나 미로 찾기같은 경로 탐색을 위한 트리가 있습니다.

탐색 트리 추적 방법
*******************
* 깊이 우선 탐색:
  
.. image:: imgs/chap5.2.1.png
	:width: 500px
	:align: center
	:height: 100px
	:alt: alternate text
	      
* 너비 우선 탐색:
  
.. image:: imgs/chap5.2.2.png
	:width: 500px
	:align: center
	:height: 100px
	:alt: alternate text

효율 좋은 탐색 방법
*******************
비용이라는 개념없이 순서만 처리하는 깊이 우선 탐색이나 너비 우선 탐색에는 한계가 있기 때문에 비용이라는 개념을 바탕으로 효율을 높여야 합니다.

비용에 따른 탐색 방법
=====================
*A\* algorithm의* `wiki <https://ko.wikipedia.org/wiki/A*_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98>`_ *를 참조해서 일단 개념을 이해 하고 책 내용을 보는 것이 좋겠다.*

예를 들어 부산에서 서울로 출발할 때 동대구를 통과하느냐 경주를 통과하느냐에 따라 시간과 비용에 차이가 나는데 이런 경우 우선 비용을 정의해야 합니다. 이 책에서는 비용에 대해 다음의 세종류를 들고 있습니다.

* 초기 상태 -> 상태 :math:`s` 의 최적 경로 이동에 드는 비용의 총합 :math:`g(s)`
* 상태 :math:`s` -> 목표하는 최적 경로 이동에 드는 비용의 총합 :math:`h(s)`
* 상태 :math:`s` 를 거치는 초기 상태 -> 목표의 최적 경로 이동에 드는 비용의 총합 :math:`f(s) (= g(s)+h(s))`

:math:`\hat{g}(s)` 를 최소화 하도록 노드 선택: 최적 탐색이라고 함 (optimal search). 탐색량이 많은 단점
      
:math:`\hat{h}(s)` 를 최소화 하도록 노드 탐색: 최선 우선 탐색 (Best-first search). 잘못 된 결과가 나올 수 있는 단점
      
      
.. 문법참조: https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#restructured-text-rest-and-sphinx-cheatsheet
