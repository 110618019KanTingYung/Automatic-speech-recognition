# Automatic-speech-recognition-110618019KanTingYung
* Taiwanese Speech Recognition

## File description
requirement.txt  程式所需安裝套件\
Speech Recognition.ipynb / speech_recognition.ip  為程式碼\
speech_recognition.sh  是測試和訓練腳本\
speech_predict2.csv 預測結果\

## Environment
Colab

## Concept

使用深度神經網絡（Deep Neural Networks，DNN）實現ASR的一般流程如下
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/image.png)\
* 從原始語音到聲學特徵
* 將聲學特徵輸入到神經網絡，輸出對應的概率
* 根據概率輸出文本序列

![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20003456.png)
將原始音檔切割成小片段後，得到每個片段相應的MFCC特徵，在使用Conv1D進行捲積，最後一層卷積的特徵圖個數和字典大小相同，經過softmax處理之後，每一個小片段對應的MFCC都能得到在整個字典上的概率分佈，計算完的
值經過CTC（Connectionist temporal classification）損失函數計算，其中需要標準答案去驗證，才能得出最後的得分。

## Architecture
1.將csv檔轉成txt檔，以便後續使用\
2.讀取音檔與文字檔\
3.取的信號轉成MFCC\
4.對MFCC作正規化\
5.建立模型\
6.訓練並預測\

## Program flow

### Import API
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20004028.png)

### Processing
 #### 先將csv檔轉成txt檔，只需要進行一次，完成後可以註解掉　
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20004408.png)

 #### 讀取音檔(wav)及文字檔
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20005318.png)
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20005331.png)
   * lower:將全部字母轉成小寫，因為答案都是小寫。若參雜大小寫錯誤率會很高。
   * 音檔需要與文字檔名稱一致，若沒有一致會跳出錯誤。 

 #### 觀察音波型圖與相應的梅爾頻譜
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20020347.png) 
   * LibROSA 是一個用於音樂和音訊分析的python包。它提供了建立音樂資訊檢索系統所需的構建塊。

 #### 把過小聲的音檔除掉
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20021002.png)
   * librosa.load(): 讀取wav,mp3等音檔
   * audio:音頻信號值
   * sr:採樣率
   * energy:計算每幀的均方根值

 #### MFCC取音檔特徵值
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20021023.png)
   * highfreq: 最高頻帶邊緣。設為8000，指把8000以上的音頻濾掉。
   * 將MFCC特徵進行歸一化: 預防訓練集與測試集不匹配的情況。

 #### 做字母字典
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20022447.png)
   * 讀取tran_texts的文字: 小寫字母+空格=27
   * 對每個字母做編號: 可以對訓練結果做對照
   * 做拼音轉數字與數字轉拼音兩種字典: 最後解碼需要使用id2char(數字轉拚音)

 #### 分割訓練集與測試集 (9:1)
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20023205.png)

 #### 定義產生批數據的函式
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20023421.png)

 #### 定義訓練參數與訓練模型
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20023946.png)
   * 由於MFCC特徵為一維序列，所以使用Conv1D進行捲積
   * Filter :輸出空間的維度
   * kernel_size: 1D 捲積窗口的長度
   * strides:捲積的步長
   * Causal:表示膨脹捲積
   * activation: 要使用的激活函數
   * Dilation_rate:指定用於膨脹捲積的膨脹率

![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20024107.png)
   * 因為特徵圖個數和字典大小相同，經過softmax處理之後輸出的機率分佈總和為 1，每一個小片段對應的MFCC都能得到在整個字典上的概率分布。
   * CTC:為損失函數的聲學模型訓練，是一種完全端到端的聲學模型訓練，不需要預先對數據做對齊，只需要一個輸入序列和一個輸出序列即可以訓練。因為音檔長度不一，此方法可以不取時間，只取特徵。
   * optimizer: 這次使用SGD優化器，其優點是梯度更新很快，更新時loss值比較震盪，有時會出現局部最優點的最優點。缺點是參數輸出呈現鋸齒狀，且也很看手氣。
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20024120.png)
   * 開始訓練，並設定EarlyStopp
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20025831.png)
   * 模型建置完成

 #### 保存模型和字典
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20030745.png)

 #### 繪製訓練過程中的損失函數曲線
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20031148.png)

 #### 對音檔重新命名並補零
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20031423.png)

如圖所示:
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20031830.png)

 #### 開始預測
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20031435.png)
   * 與對trandata的方式依樣，這樣預測結果才不會跑掉
   * 讀取要預測的檔案，並把它轉換成路徑形式，是後續讀檔的格式

![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20033233.png)
   * 去除小聲音檔
   * 正規化
   * 預測
   * 解碼: 還要去除-1(無法辨認的地方)，不等於-1則使用id2char獲得辨識文字
   
 #### 把解碼的字串輸出成csv檔
![](https://github.com/MachineLearningNTUT/automatic-speech-recognition-110618019KanTingYung/blob/main/photo/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-01-19%20031502.png)

### 嘗試改善

 * 更改模型建置，如:DeepSpeech2
![image](https://user-images.githubusercontent.com/94077930/150551959-08c488e7-d464-4ef9-86d0-c02f9a967600.png) \
有三層的Conv1D or Conv2D，加上七層的Recurrent or GRU ，並用Batch Normalization 做連接。
* 用不同的方式辨識音頻，如:MEL spectrogram，或修改MFCC裡的參數值 \
![image](https://user-images.githubusercontent.com/94077930/150553126-bbaf946e-a2f5-45f6-9b6e-891765aaf405.png)











