import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

data = pd.read_excel("bicycle_prices.xlsx")
data.head() # baştaki 5 elemanı alırız
sbn.pairplot(data)

##Veriyi test/train olarak ikiye ayırmak
#train_test_split
from sklearn.model_selection import train_test_split

#y = wx + b
#y -> label
y = data["Fiyat"].values

#x -> feature (özellik)
x = data[["BisikletOzellik1" , "BisikletOzellik2"]].values

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.33 , random_state = 15)

x_train.shape
x_test.shape

#scaling boyut değiştirmek
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
#model oluşturuyor

model.add(Dense(4,activation = "relu"))
model.add(Dense(4,activation = "relu"))
model.add(Dense(4,activation = "relu"))
#3 tane hidden layer oluşturuyor ve 5 nöronlu aktivasyon fonksiyonu relu olarak seçiyor ve bir model oluşturuyor

model.add(Dense(1))
#çıktı olarak bir tane nöron alıyor

model.compile(optimizer = "rmsprop" , loss = "mse")
#loss mse ise Mean Squared error hata payı gibi düşünülebilir
#Gradient araması yapıyor
#Birleştirip çalışmaya hazır hale getiriyor

model.fit(x_train , y_train , epochs = 250 ) ## traine geçiyoruz epochs kaç defa complie edicek
loss = model.history.history["loss"]

# loss tablosunu çıkarıyoruz
sbn.lineplot(x = range(len(loss)) , y = loss)
trainLoss = model.evaluate(x_train,y_train , verbose=0)
testLoss = model.evaluate(x_test,y_test , verbose = 0)


#test tahmin modelini oluşturuyoruxz
testTahminleri = model.predict(x_test)

tahminDf = pd.DataFrame(y_test, columns = ["Real Y"] )
testTahminleri = pd.Series(testTahminleri.reshape(330,))

tahminDf = pd.concat([tahminDf , testTahminleri ] ,axis = 1)

tahminDf.columns = ["Real Y ","Tahmin Y"]

sbn.scatterplot(x = "Real Y " , y = "Tahmin Y" , data = tahminDf)

from sklearn.metrics import mean_absolute_error , mean_squared_error
data.describe()
yeniBisikletOzellikleri = [[1760,1757]]
yeniBisikletOzellikleri = scaler.transform(yeniBisikletOzellikleri)
model.predict(yeniBisikletOzellikleri)
from tensorflow.keras.models import load_model

model.save("bisiklet_modeli.keras")

sonradanModel = load_model("bisiklet_modeli.keras")
sonradanModel.predict(yeniBisikletOzellikleri)