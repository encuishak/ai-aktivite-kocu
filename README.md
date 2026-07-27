#  AI & ML Destekli Aktivite ve Sağlık Koçu

##  Proje Hakkında
Bu proje, kullanıcıların günlük aktivite verilerini analiz ederek sağlık ve fitness hedeflerine ulaşmalarını sağlayan yapay zeka destekli bir web uygulamasıdır. **Makine Öğrenmesi (ML)** algoritmaları ile kullanıcı profillemesi ve kalori tahmini yaparken, **Büyük Dil Modelleri (LLM)** kullanarak kişiselleştirilmiş, motive edici günlük tavsiyeler üretir.

Uygulama, veri bilimi ve yapay zeka entegrasyonunu hızlı ve etkileşimli bir arayüzle sunmak için **Streamlit** kullanılarak geliştirilmiştir.

---

##  Temel Özellikler

*   **Kullanıcı Segmentasyonu (K-Means):** Kullanıcıların adım sayısı, aktif ve hareketsiz geçirdikleri sürelere göre *Masa Başı Çalışanı*, *Hafif Aktif* veya *Düzenli Sporcu* olarak 3 farklı grupta profillenmesi.
*   **Akıllı Kalori Tahmini (Random Forest):** Mevcut verisi olmayan kullanıcılar için manuel girilen efor verileri üzerinden yakılacak kalorinin yüksek doğrulukla tahmin edilmesi.
*   **Dinamik Hedef Belirleme:** Kilo verme, stres yönetimi, uyku kalitesi veya genel sağlık gibi kişisel hedeflere göre analiz yapılması.
*   **Generative AI Tavsiyeleri:** Google Gemini 2.5 Flash modeli ile, kullanıcının seçtiği hedefe, ML profiline ve aktivite özetine uygun; spesifik ve eyleme geçirilebilir tavsiyelerin (prompt mühendisliği ile) üretilmesi.
*   **Etkileşimli Görselleştirme:** Metriklerin, ilerleme çubuklarının (10.000 adım hedefi) ve zaman dağılım grafiklerinin dinamik sunumu.

---

##  Kullanılan Teknolojiler

*   **Arayüz & Web Çerçevesi:** [Streamlit](https://streamlit.io/)
*   **Veri İşleme:** Pandas, NumPy
*   **Makine Öğrenmesi:** Scikit-Learn (`KMeans`, `RandomForestRegressor`)
*   **Yapay Zeka (LLM):** Google Generative AI (Gemini 2.5 Flash)

## Kurulum

Projeyi klonlayın:
```bash
git clone https://github.com/kullanici_adi/proje_adi.git
```

Gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

Uygulamayı başlatın:
```bash
streamlit run app.py
```

## Kullanım
1. Uygulama açıldığında, kullanıcıdan bilgiler istenecek.
2. Kullanıcı bilgileri girdikten sonra, önerilen aktiviteleri görüntüleyebilir.
3. Kullanıcılar önerilen aktiviteleri kaydedebilir ve takvimlerine ekleyebilir.

## Katkıda Bulunma
Eğer bu projeye katkıda bulunmak isterseniz, lütfen aşağıdaki adımları izleyin:
1. Projeyi fork'layın.
2. Yeni bir dal oluşturun: `git checkout -b feature/YeniOzellik`
3. Değişikliklerinizi yapın ve kaydedin: `git commit -m 'Yeni bir özellik ekledim'`
4. Dalınızı gönderin: `git push origin feature/YeniOzellik`
5. Bir Pull Request açın.

## İletişim
Herhangi bir sorunuz varsa, lütfen iletişime geçin:
- **E-posta:** encuishak613@gmail.com

---
*Bu README dosyası, AI Aktivite Koçu uygulaması hakkında temel bilgilere ve kullanım talimatlarına yönlendirmektedir.*
