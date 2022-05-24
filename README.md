# ANT_MIND

## 네이버 종목토론실 댓글 감성(공포/탐욕) 분석

## 0. 목차 

1. 개요 및 플로우 차트
   - 개요
   - 플로우차트
2. 굿

## 1. 개요 및 플로우 차트

- #### 개요
  ![감성분석플로우](./img/감성분석.png)

  

   개미 투자자들의 감정을 분석하기 위해서 그들의 감정이 제일 잘 들어나는 네이버 종목토론실의 댓글을 분석 및 학습한다. 최종적으로 이러한 모델로 이용자의 희망 주식종목의 당일 댓글을 분석하여 공포/탐욕지수를 시각화 하는 것이 목표이다.

   감성분석은 크게 두가지로 분류될 수 있다. 첫번째는 어휘 기반 분석, 두번째는 머신 러닝 분석이다.  이번 프로젝트에서 사용하는 비정형 데이터인 네이버 금융 종목토론실 댓글의 감정(공포/탐욕) 분석을 진행할 예정이다.하지만 이러한 데이터에는 현재 공포/탐욕 레이블링이 되어있는 학습데이터가 존재하지 않다. 그렇기 때문에 머신 러닝 분석에서 비지도 학습을 진행하거나, 어휘 기반 분석을 진행해야만 한다. 나는 지도 학습으로 분석을 하고 싶기 때문에, 내가 직접 공포/탐욕 단어 메뉴얼을 만들어서 어휘 기반 분석을 진행해서 학습데이터를 생성한 뒤, 그 데이터로 지도 학습을 진행할 계획이다. 지도학습은 최근 성능이 좋다고 유명한 BERT모델을 이용해서 학습할 계획이다.

   

  

  
  
  
  
- #### 플로우차트

## 2. 어휘 기반 공포/탐욕 레이블링

- #### 네이버 종목토론실 댓글 크롤링 

  asd

  asd

  asd

  asd

  asd

- #### 댓글 전처리

  크롤링한 카카오댓글과 삼성전자댓글 데이터를 불러옵니다.

  ```python
  df_ka = pd.read_csv('./src/comments_kakao_300000.csv')
  df_sam = pd.read_csv('./src/comments_samsung_200000.csv')
  df = df_ka.append(df_sam)
  df = df.reset_index(drop=True)
  print('데이터 크기: ',df.shape)
  # 데이터 크기:  (500000, 5)
  ```

  ![댓글전처리1](./img/댓글전처리1.jpeg)

  크롤링때 발생한 error와 결측치를 제거합니다.

  ```python
  df = df[df['댓글'] != 'error']
  df = df.dropna()
  print('데이터 크기: ',df.shape)
  # 데이터 크기:  (499984, 5)
  ```

  정규표현식을 통해 댓글을 정제하고, 공백으로 남은 댓글을 제거합니다. 필요한 컬럼만 남깁니다.

  ```python
  df['정제된 댓글'] = df['댓글'].str.replace('\[삭제된 게시물의 답글\]',' ')
  df['정제된 댓글'] = df['정제된 댓글'].str.replace('답글:',' ')
  df['정제된 댓글'] = df['정제된 댓글'].str.replace('[^가-힣]',' ').str.replace(' +',' ').str.strip()
  df = df[df['정제된 댓글'] != '']
  df = df.reset_index(drop=True)
  df = df[['댓글','정제된 댓글']]
  print('데이터 크기: ',df.shape)
  # 데이터 크기:  (493864, 2)
  ```

  <img src="./img/댓글전처리2.jpeg" alt="댓글전처리2"  />

  KoNLPy를 이용해서 정제된 댓글을 형태소분리를 진행합니다.

  ```python
  okt = Okt()
  tag_list = ['Noun','Verb','Adjective','VerbPrefix']
  tokenized_data = []
  for i in range(df.shape[0]):
      tokenized_sentence = okt.pos(df['정제된 댓글'][i], stem=True)
      tag_checked_sentence = []
      for j in tokenized_sentence:
          x,y = j
          if y in tag_list:
              tag_checked_sentence.append(x)
      tokenized_data.append(tag_checked_sentence)
      print(f'\r{i+1}개 형태소분석 완료.',end='')
  df['토큰화 댓글'] = tokenized_data
  df = df.reset_index(drop=True)
  df = df[df['토큰화 댓글'] != '[]']
  print('\n데이터 크기: ',df.shape)
  # 493864개 형태소분석 완료.
  # 데이터 크기:  (493864, 3)
  ```

  토큰화 댓글을 기준으로 중복된 댓글을 제거합니다.

  ```python
  df = df.drop_duplicates('토큰화 댓글')
  df = df.reset_index(drop=True)
  print('\n데이터 크기: ',df.shape)
  # 데이터 크기:  (362544, 3)
  ```

  ![댓글전처리3](./img/댓글전처리3.jpeg)

- #### 공포/탐욕 단어 메뉴얼 지정

  asd

  asd

  asd

  asd

  asd

- #### 공포/탐욕 레이블링

## 3. 



## 4.





참고문헌

- [감성 분석 참고 블로그](https://yngie-c.github.io/nlp/2020/07/31/sentiment_analysis/)
- [감성사전에 기반한 준지도학습 감성분석 모델](https://realblack0.github.io/portfolio/pmi)
- [준지도 학습을 이용한 감성분석](https://github.com/realblack0/semi-supervised-sentiment-analysis/blob/master/sample/(sample)형태소 분석.ipynb)
- [SKTbrain팀 KoBERT 실습사례 네이버 영화리뷰 감성분석](https://github.com/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb)
- [BERT모델을 이용해서 금융뉴스 긍부정 분석](https://github.com/ukairia777/finance_sentiment_corpus/blob/main/BERT_sentiment_analysis_kor.ipynb)
- [BERT모델을 이용해서 주52시간근무제 관련 댓글 감성분석](https://projectlog-eraser.tistory.com/25)

