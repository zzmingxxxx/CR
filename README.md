# CR Characters Recognition

關於NIST Special Database 19數據集by_class.zip
1、資料描述
(1)png, 3channel，128*128，計張MB
(2)分別存在4a～79資料夾共62個，也分別代表0～9，a～z等字母
(3)train資料夾內為訓練資料約4仟，hsf_4內為驗證資料約500個，再乘62，約30餘萬張
2、資料讀取與存檔
(1)轉成1 channel 28*28 nupy array
(2)存成nupy npz壓縮檔

DataPath='~/MNIST_data/sd19_by_class/' #sd-19 data path
train_img=np.vstack((train_img,im2)) #將影像im2 加在 train_img後面
train_lbl = np.append(train_lbl, int(i1, 16)) #將label i1 加在train_lbl後面;
