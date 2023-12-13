# object_LVDM
final project for genAI course in SNU (2023 Fall)
![image](https://github.com/Sangyoon-Bae/object_LVDM/assets/90450600/91bc178c-230e-45af-b018-2d1fb14ab502)
## Slot loss
1. 가까운 frame끼리 similarity가 크다! 따라서, 두 frame이 가까울수록 penalty 곱하기 전 -log(분수) 값이 작아야 함.
2. penalty term은 가까운 frame을 더 가깝게 (-log(분수) 값이 작게) 하는 역할. i, j가 가까울수록 penalty term이 작아야 함. 일단은 abs|i-j|로 했음. 시간이 많으면 이것도 다 테스트 해보면 좋을텐데..
