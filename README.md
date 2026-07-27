# 🏃‍♂️ AI & ML Destekli Aktivite ve Sağlık Koçu

Bu proje, kullanıcıların günlük fiziksel aktivite verilerini makine öğrenmesi (Machine Learning) ve Büyük Dil Modelleri (LLM) ile analiz ederek kişiselleştirilmiş sağlık, egzersiz ve yaşam tarzı önerileri sunan hibrit bir yapay zeka web uygulamasıdır.

## ✨ Öne Çıkan Özellikler

* **K-Means Segmentasyonu:** Kullanıcıların adım, aktif dakika ve hareketsiz süre verilerini analiz ederek onları otomatik olarak "Masa Başı Çalışanı", "Hafif Aktif" veya "Düzenli Sporcu" olarak profillendirir.
* **Random Forest Regresyonu:** Manuel girilen efor verilerine (adım, aktif süre vb.) dayanarak kullanıcının gün sonunda yakacağı kaloriyi bilimsel bir doğrulukla tahmin eder.
* **Gemini LLM Entegrasyonu:** Makine öğrenmesi algoritmalarından elde edilen istatistikleri ve segmentasyon verilerini kullanarak kullanıcıya tamamen kişiselleştirilmiş, hedefine yönelik (kilo verme, stres yönetimi vb.) motive edici günlük görevler sunar.
* **Dinamik Veri Girişi:** Sisteme veri seti üzerinden mevcut kullanıcıları seçerek veya manuel hedefler girerek erişilebilir.

## 🛠️ Kullanılan Teknolojiler

* **Arayüz & Web Çerçevesi:** [Streamlit](https://streamlit.io/)
* **Veri Analizi ve Manipülasyonu:** Pandas, NumPy
* **Makine Öğrenmesi (ML):** Scikit-Learn (K-Means, Random Forest Regressor)
* **Büyük Dil Modeli (LLM):** Google Generative AI (Gemini 2.5 Flash)
* **Programlama Dili:** Python 3.x
   
