# Laporan Proyek Machine Learning - Rizky Adhi Nugroho
---
## Domain Proyek  
penyakit jantung dan pembuluh darah telah menggantikan peran penyakit tuberkulosis paru sebagai penyakit epidemik, terutama pada laki-laki. Pada saat ini penyakit jantung merupakan penyebab kematian nomor satu di dunia. Pada tahun 1999 sedikitnya 55,9 juta atau setara dengan 30,3 % kematian diseluruh dunia disebabkan oleh penyakit jantung. Menurut Data dari Badan Kesehatan Dunia (WHO), 60 % dari seluruh penyebab kematian penyakit jantung adalah Heart disease atau penyakit jantung koroner (PJK). Di Indonesia, penyakit jantung juga cenderung meningkat sebagai penyebab kematian.
Melonjaknya jumlah kasus penyakit ini menunjukkan suatu permasalahan, yaitu banyaknya orang yang kurang waspada terhadap gejala penyakit ini, dengan melakukan analisis klasifikasi, kita bisa mengetahui analisis hubungan faktor faktor yang memengaruhi heart disease. penelitian sebelumnya pernah dilakukan oleh derisma pada tahun 2020 mengenai Perbandingan Kinerja Algoritma untuk analisis Prediksi Penyakit Jantung , berikut adalah referensi nya : https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/2152
## Busines Understanding
Heart Disease atau biasa dikenal Penyakit Jantung adalah penyebab kematian nomor satu pada orang dewasa di Amerika. Di seluruh dunia, jumlah penderita penyakit ini terus bertambah. Penyakit jantung tidak Datang dengan tiba tiba, penyakit ini pasti juga mempunyai gejala yang bisa di deteksi. Dengan melakukan analisis klasifikasi, kita bisa mengetahui apa saja faktor yang berpengaruh terhadap munculnya penyakit ini, sehingga hasil analisis tersebut bisa di jadikan referensi informasi agar manusia bisa lebih waspada terhadap penyakit ini.
### problem statements
- Dari serangkaian variabel/faktor yang ada, faktor apa yang paling berkorelasi terhadap penyakit jantung?
- mengetahui apakah manusia terkena penyakit jantung dengan kondisi tertentu

### Goals
- Mengetahui nilai korelasi antar variabel dengan penyakit jantung.
- Membuat model machine learning yang dapat memprediksi penyebab penyakit jantung seakurat mungkin berdasarkan faktor yang ada.

### Solutions Statements
- Dengan melakukan EDA pada dataset ini, kita dapat mengetahui korelasi antar data, kita dapat mengetahui variabel apa yang paling berkorelasi dengan variabel target.
- kita dapat melakukan analisis prediksi klasifikasi dengan menggunakan 2 algoritma machine learning sebagai perbandingan mana algoritma dari kedua nya yang terbaik 
  - model 1 : Logistic Regression, model regresi logistik ini sangat baik untuk melakukan analisis klasifikasi , model regresi logistik biner sangat cocok untuk dataset di mana variabel independen mempunyai 2 nilai yaitu 0(tidak menderita) atau 1(menderita).
  - model 2 : Model K-Nearest Neighboor (KNN), KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. model ini mempunyai kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar. model ini sangat cocok digunakan untuk dataset yang tidak terlalu besar seperti pada dataset penyakit jantung yang digunakan.

## Data Understanding
Data yang dipakai adalah Data dari kaggle mengenai penyakit jantung (https://www.kaggle.com/johnsmith88/heart-disease-dataset/code?datasetId=216167), data ini diambil tahun 1988, di mana di dalamnya terdapat survei mengenai beberapa orang yang mempunyai penyakit heart Disease dan tidak, beserta dengan variabel faktor kondisi yang mempengaruhinya seperti tekanan darah, kolestrol, dan lainnya. pada data ini saya menggunakan 13 variabel pada data, dan dengan melakukan EDA pada data, didapatkan bahwa ada 1025 data dengan 13 kolom variabel, untuk lebih jelasnya dapat dilihat pada list di bawah:
1. Age = umur 
2. Sex = jenis kelamin, dengan kategori (1 = laki, 2 = perempuan )
3. Cp : tipe nyeri dada (0 = typical angina, 1=Atypical angina, 2=non-anginal pain, 3=asymptomatic)
4. Trestbps: tekanan darah istirahat (dalam mm Hg )
5. Chol : serum cholestoral dalam mg/dl 
6. Restecg: hasil resting electrocardiographic  (dengan nilai 0,1,2)
7. Thalach: detak jantung maksimum
8. Exang : olahraga menyebabkan nyeri dada ( 1 = iya, 0 = tidak )
9. oldpeak = depresi
10. Slope : the slope of the peak exercise ST segment
11. Ca : jumlah pembuluh darah utama ( dengan nilai 0, 1, dan 2 )
12. Thal : thalasemia: 0 = normal; 1 = fixed defect; 2 = reversable defect
13. Target : Pasien , 0 = tidak menderita, 1 = menderita.


dengan melakukan visualisasi yaitu data membuat bar chart pada variabel target, didapatkan bahwa terdapat 526 (51,3%) data penderita dan 499 (48,7%) data bukan penderita

![visualisasi bar chart](https://user-images.githubusercontent.com/88422709/136927806-afb10534-9b39-4de1-899b-4c59b75a5e71.png)

## Data Preparation
### EDA(cek data Missing value, duplikasi, inkonsisten, dan Outliers)
teknik ini dilakukan untuk membersihkan data yang masih mentah sebelum masuk tahap analisis, teknik ini meliputi pengecekan dan penanganan missing value, duplikasi data, konsistensi, dan outliers data.
- setelah dilakukan pengecekan, ternyata tidak ada missing value pada data di setiap variabel yang ada
- Terdapat duplikasi sebanyak 723 data, data duplikasi ini tidak di drop karena tidak berpengaruh pada analisis.
- Tidak ada data yang bernilai negatif, artinya nilai pada data konsisten
- terdapat banyak ouliers (nilai yang tidak wajar) pada data, maka dilakukan penanganan.
- karena jumlah outlier banyak, maka di lakukan penanganan outlier dengan metode imputasi dengan median, teknik ini lebih disarankan daripada melakukan drop pada outlier, agar tidak mengurangi ke aslian informasi pada data.
- setelah data outliers di tangani, maka data masih berjumlah 1025 data.

### EDA (analisis univariat)
Dengan melakukan EDA Univariat pada data yang sudah di cleaning, kita dapat mengetahui perbandingan antar data, di antaranya : 
- terdapat 526 sampel data target yang menderita penyakit
- terdapat 499 sampel data target yang tidak menderita penyakit
- variabel numerik pada data yaitu = 'age', 'trestbps', 'chol', 'thalach', dan 'oldpeak'.
- variabel kategorik pada data yaitu = 'sex', 'cp', 'restecg', 'exang', 'slope'. 'ca','thal', dan 'target'

### EDA ( Analisis Multivariat ) 
analisis ini dilakukan menggunakan heatmap, kita dapat mengetahui nilai korelasi antar variabel, didapatkan bahwa ada 3 variabel yang mempunyai nilai korelasi paling tinggi terhadap variabel target, yaitu variabel 'cp' (0,43), 'oldpeak' (0,43), dan 'exang' (0,44) , didapat juga bahwa variabel 'chol'(0,13), 'trestbps'(0,11) dan 'restecg'(0,13) mempunyai nilai korelasi  terkecil pada variabel target. variabel variabel tersebut tidak didrop dengan tujuan menjaga kualitas informasi pada data. sehingga untuk variabel data yang sudah siap dianalisis yaitu :
- variabel numerik pada data = 'age', 'trestbps', 'chol', 'thalach', dan 'oldpeak'.
- variabel kategorik pada data = 'sex', 'cp', 'restecg', 'exang', 'slope'. 'ca','thal', dan 'target'

### split data
dengan data yang sudah di cleaning, saya membagi data :
- Total seluruh data(100%) pada dataset = 1025
- Total train data(80%) pada dataset = 820
- Total test data(20%) pada dataset = 205

## Modeling
membandingkan antara kedua algoritma machine learning yaitu model logistic regression dan model K-nearest neighbor (KNN), dari kedua model tersebut akan ditentukan model yang lebih baik di antara kedua nya.

### model 1( Logistic Regression )
Regresi logistik merupakan teknik regresi yang fungsinya untuk memisahkan dataset menjadi dua bagian (kelompok). Regresi logistik adalah regresi yang variabel dependennya bersifat biner atau memiliki dua kategori sedangkan variabel independennya bisa berupa data biner atau kontinu, model ini sangat cocok dengan dataset karena variabel dependent nya hanya bernilai 0 dan 1. pada model ini kita menggunakan solver 'lbfgs' dan multi_class='auto' sebagai default.

### model 2 ( KNN )
KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. karena jumlah klasifikasi nya 2 (genap), maka Kita menggunakan k = 3 tetangga ( ganjil ) , pemilihan k ini juga untuk menghindari overfit dan mendapatkan kemampuan prediksi yang optimal.

## Evaluation
Dalam pemodelan machine learning, istilah metrik mengacu pada suatu nilai yang dapat digunakan untuk merepresentasikan performa model yang dihasilkan. pada kasus ini, saya menggunakan metrik yang paling umum digunakan untuk kasus klasifikasi yaitu akurasi, presisi, recall, dan F1 score.

- Akurasi adalah metrik paling awam / paling diketahui pada pemodelan klasifikasi. Akurasi adalah persentase jumlah data yang diprediksi secara benar terhadap jumlah keseluruhan - data.
- precision adalah perbandingan antara True Positive (TP) dengan banyaknya data yang diprediksi positif
- Recall adalah perbandingan antara True Positive (TP) dengan banyaknya data yang sebenarnya positif.
- F1-Score adalah metriks yang merangkum presisi & sensitivitas dengan mengambil rataan harmonik dari keduanya.

rumus dari metrik berdasarkan confussion matriks adalah:

![rumus metrik](https://static.packt-cdn.com/products/9781785282287/graphics/B04223_10_02.jpg)

deskripsi:
- True Negative (TN): Model memprediksi seorang pasien tidak terkena penyakit dan memang kenyataannya, pasien itu tidak terkena penyakit.
- True Positive (TP): Model memprediksi seorang pasien terkena penyakit dan memang - kenyataannya, pasien itu terkena penyakit.
- False Negative (FN): Model memprediksi seorang pasien tidak terkena penyakit, padahal pasien terkena penyakit.
- False Positive (FP): Model memprediksi seorang pasien terkena penyakit, padahal pasien tidak terkena penyakit.

hasil metrik akurasi :
- akurasi logistic regression : 0,85
- akurasi KNN : 0,93

Berdasarkan metrik evaluasi, dapat disimpulkan bahwa model terbaik untuk dataset ini adalah model KNN dengan nilai akurasi 0,95.
