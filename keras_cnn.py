import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
import os


class KCNN(object):
    # 초기화
    def __init__(self):
        self.classifier = Sequential()
        self.training_set = None
        self.test_set = None

    # 모델 구성
    # CNN 모델 구성을 위해서 레이어를 쌓는다
    # 마지막 테스트할 사진이 학습된 사람과 일치하는지 아닌지를 분별하는 문제이기 때문에 마지막 compile layer는 loss를 binary로 지정한다.
    def create_model(self):
        self.classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        self.classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units = 128, activation = 'relu'))
        self.classifier.add(Dense(units = 1, activation = 'sigmoid'))
        self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # 모델 학습과정 설정
    # 만약 미리 학습된 내용이 있는 h5파일이 있다면 다시 학습을 하지 않고 저장된 모델을 사용하여 테스트 사진을 판별한다.
    # 학습된 내용이 없다면 학습을 시작한다.
    def fit(self, train_path, test_path, epochs):
        if os.path.isfile('./models/model.h5'):
            print("모델 로드")
            self.classifier = load_model('./models/model.h5')
        else:
            print('학습 시작')
            self.create_model()
            # train 데이터의 양이 작기 때문에 데이터를 불러올 때 증폭시켜서 가져온다.
            train_datagen = ImageDataGenerator(rescale = 1./255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True
                                                )
            # 위와 동일
            test_datagen = ImageDataGenerator(rescale = 1./255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True
                                                )
            # 위에서 선언한 train_datagen을 활용해서 train data를 불러온다. 
            self.training_set = train_datagen.flow_from_directory(train_path,
                                                                target_size = (128, 128),
                                                                batch_size = 3,
                                                                class_mode = 'binary')
            # 위와 동일
            self.test_set = test_datagen.flow_from_directory(test_path,
                                                        target_size = (128, 128),
                                                        batch_size = 3,
                                                        class_mode = 'binary')
            
            hist = self.classifier.fit_generator(self.training_set,
                                        steps_per_epoch = 18,
                                        epochs = epochs,
                                        validation_data = self.test_set,
                                        validation_steps = 7)

            self.classifier.save('./model.h5')
        
    #모델 평가
    def predict(self,test_path):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_set = test_datagen.flow_from_directory(test_path,
                                                            target_size=(128, 128),
                                                            batch_size=3,
                                                            class_mode='binary')

        if self.classifier is not None:
            output = self.classifier.predict_generator(test_set, steps=1)
            return test_set.class_indices, output
        else:
            print('학습모델이 없습니다.')



if __name__=='__main__':
    # 클래스 객체 생성
    model = KCNN()

    # parameter1 : 훈련데이터 셋위치(디렉토리형태),
    # parameter2 : 테스트 데이터위치(디렉토리형태),
    # parameter3 : 훈련반복횟수(h5파일이 없는 경우)
    model.fit('./dataset2/training_set', './dataset2/test_set', 50)
    _, result = model.predict('./dataset2/test')

    for i in range(len(result)):
        if result[i] > 0.8:
            print("학습된 사진과 일치", "확률:", result[i]*100)
        else:
            print("학습된 사진과 일치하지 않음", "확률:", (1-result[i])*100)
