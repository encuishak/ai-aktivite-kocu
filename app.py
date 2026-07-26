import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="AI Aktivite Koçu", page_icon="🏃‍♂️", layout="wide")
st.title("🏃‍♂️ AI & ML Destekli Aktivite ve Sağlık Koçu")
st.markdown("Makine öğrenmesi algoritmaları ve Büyük Dil Modellerini (LLM) kullanarak sağlığınızı optimize edin.")


# --- VERİ İŞLEME VE MAKİNE ÖĞRENMESİ MODELLERİ ---
@st.cache_resource
def load_and_train_models(file_path):
    try:
        # 1. Veriyi Oku ve İşle
        df = pd.read_csv(file_path)
        profile_df = df.groupby('Id').agg({
            'TotalSteps': 'mean',
            'SedentaryMinutes': 'mean',
            'LightlyActiveMinutes': 'mean',
            'VeryActiveMinutes': 'mean',
            'FairlyActiveMinutes': 'mean',
            'Calories': 'mean',
            'ActivityDate': 'count'
        }).reset_index()

        profile_df.rename(columns={'ActivityDate': 'DaysRecorded'}, inplace=True)
        profile_df['ActiveMinutes'] = profile_df['VeryActiveMinutes'] + profile_df['FairlyActiveMinutes']

        # 2. K-MEANS MODELİ (Kullanıcı Segmentasyonu)
        # Adım, Hareketsiz Süre ve Aktif Dakika bazında 3 gruba ayırıyoruz
        X_cluster = profile_df[['TotalSteps', 'SedentaryMinutes', 'ActiveMinutes']]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        profile_df['Cluster'] = kmeans.fit_predict(X_cluster)

        # Kümeleri anlamlı isimlere dönüştür (Adım ortalamasına göre sırala)
        cluster_centers = profile_df.groupby('Cluster')['TotalSteps'].mean().sort_values()
        segment_labels = {
            cluster_centers.index[0]: "Masa Başı Çalışanı (Düşük Aktivite)",
            cluster_centers.index[1]: "Hafif Aktif (Orta Seviye)",
            cluster_centers.index[2]: "Düzenli Sporcu (Yüksek Aktivite)"
        }
        profile_df['Segment'] = profile_df['Cluster'].map(segment_labels)

        # 3. RANDOM FOREST MODELİ (Kalori Tahmini)
        X_reg = profile_df[['TotalSteps', 'ActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes']]
        y_reg = profile_df['Calories']
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_reg, y_reg)

        return profile_df, kmeans, rf_model, segment_labels

    except FileNotFoundError:
        st.error(f"'{file_path}' dosyası bulunamadı. Lütfen dosyanın projenin kök dizininde olduğundan emin olun.")
        return None, None, None, None


# Modelleri ve Veriyi Yükle
df_profiles, kmeans_model, rf_model, segment_labels = load_and_train_models('dailyActivity_merged.csv')

# --- YAN MENÜ (SIDEBAR) AYARLARI ---
with st.sidebar:
    st.header("⚙️ Ayarlar")

    st.subheader("📂 Veri Kaynağı")
    data_source = st.radio(
        "Lütfen veri giriş yöntemini seçin:",
        ["Mevcut Kullanıcı Seç", "Manuel Veri Gir (ML Tahmini)"]
    )

    st.divider()

    user_data = None
    display_name = ""

    if data_source == "Mevcut Kullanıcı Seç" and df_profiles is not None:
        st.header("👤 Kullanıcı Seçimi")
        user_list = df_profiles['Id'].astype(str).tolist()
        selected_user_id = st.selectbox("Bir Kullanıcı ID Seçin:", user_list)
        user_data = df_profiles[df_profiles['Id'] == int(selected_user_id)].iloc[0]
        display_name = f"Kullanıcı ID: {selected_user_id}"

    elif data_source == "Manuel Veri Gir (ML Tahmini)":
        st.header("✍️ Hedeflerinizi Girin")
        st.caption(
            "Makine öğrenmesi modelimiz, gireceğiniz efor değerlerine göre harcayacağınız kaloriyi ve profilinizi tahmin edecektir.")

        m_steps = st.number_input("Günlük Adım Sayısı:", min_value=0, max_value=100000, value=6500, step=500)
        m_active = st.number_input("Aktif Egzersiz (Dk):", min_value=0, max_value=1440, value=20, step=5)
        m_light = st.number_input("Hafif Hareket (Dk):", min_value=0, max_value=1440, value=120, step=10)
        m_sedentary = st.number_input("Hareketsiz Süre (Dk):", min_value=0, max_value=1440, value=800, step=30)

        if rf_model is not None and kmeans_model is not None:
            # ML Kalori Tahmini
            input_features = pd.DataFrame([[m_steps, m_active, m_light, m_sedentary]],
                                          columns=['TotalSteps', 'ActiveMinutes', 'LightlyActiveMinutes',
                                                   'SedentaryMinutes'])
            predicted_calories = rf_model.predict(input_features)[0]

            # ML Profil (Segment) Tahmini
            cluster_features = pd.DataFrame([[m_steps, m_sedentary, m_active]],
                                            columns=['TotalSteps', 'SedentaryMinutes', 'ActiveMinutes'])
            pred_cluster = kmeans_model.predict(cluster_features)[0]
            predicted_segment = segment_labels[pred_cluster]

            user_data = pd.Series({
                'TotalSteps': m_steps,
                'ActiveMinutes': m_active,
                'LightlyActiveMinutes': m_light,
                'SedentaryMinutes': m_sedentary,
                'Calories': predicted_calories,
                'Segment': predicted_segment,
                'DaysRecorded': 1
            })
            display_name = "Manuel Verileriniz (ML Tahminli)"

    st.divider()

    st.header("🎯 Analiz Hedefi")
    user_goal = st.selectbox(
        "Yapay zeka hangi hedefe odaklansın?",
        ["Genel Sağlık ve Zindelik", "Kilo Vermek / Yağ Yakmak", "Stres Yönetimi", "Uyku Kalitesini Artırmak"]
    )

# --- ANA EKRAN VE GÖRSELLEŞTİRME ---
if user_data is not None:
    st.subheader(f"📊 {display_name} - Aktivite Özeti")

    # ML Tarafından Bulunan Kullanıcı Segmentini Vurgulu Göster
    st.info(f"🤖 **Makine Öğrenmesi Profiliniz:** {user_data['Segment']}")

    # Metrikleri yan yana göster
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Ortalama Adım", value=f"{int(user_data['TotalSteps']):,}")
    col2.metric(label="Aktif Dakika", value=f"{int(user_data['ActiveMinutes'])} dk")
    col3.metric(label="Hareketsiz Süre", value=f"{int(user_data['SedentaryMinutes'])} dk")

    # Kalori kısmında manuel giriş varsa bunun bir "Tahmin" olduğunu belirt
    cal_label = "Tahmini Kalori (ML)" if data_source == "Manuel Veri Gir (ML Tahmini)" else "Yakılan Kalori"
    col4.metric(label=cal_label, value=f"{int(user_data['Calories'])} kcal")

    st.write("")

    # 10.000 Adım İlerleme Çubuğu
    step_goal = 10000
    current_steps = int(user_data['TotalSteps'])
    progress_val = min(current_steps / step_goal, 1.0)
    st.progress(progress_val, text=f"Günlük 10.000 Adım Hedefine Ulaşım Oranı: %{int(progress_val * 100)}")

    st.write("")

    with st.expander("📈 Günlük Zaman Dağılımı Grafiğini Gör"):
        chart_data = pd.DataFrame(
            {"Dakika": [user_data['ActiveMinutes'], user_data['LightlyActiveMinutes'], user_data['SedentaryMinutes']]},
            index=["Orta/Yüksek Aktif", "Hafif Aktif", "Hareketsiz"]
        )
        st.bar_chart(chart_data, color="#ff4b4b")

    st.divider()
    st.subheader("💡 Yapay Zeka Tavsiyeleri")

    # --- LLM ENTEGRASYONU ---
    if st.button("Tavsiye Üret", type="primary"):
        with st.spinner(f"AI verileri analiz ediyor ve '{user_goal}' hedefine uygun tavsiyeler üretiyor..."):
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

                # Geliştirilmiş Dinamik LLM Context
                llm_context = (
                    f"Kullanıcının makine öğrenmesi algoritmaları tarafından belirlenen profili: '{user_data['Segment']}'. "
                    f"Günde ortalama {int(user_data['TotalSteps'])} adım atıyor. "
                    f"{int(user_data['SedentaryMinutes'])} dakika hareketsiz kalırken, "
                    f"{int(user_data['ActiveMinutes'])} dakika orta/yüksek eforlu egzersiz yapıyor ve "
                    f"{int(user_data['Calories'])} kalori harcıyor."
                )

                user_prompt = f"""
                    SENİN ROLÜN: Sen uzman bir sağlık koçu ve fitness danışmanısın. Amacın, kullanıcıların 
                    günlük aktivite verilerini analiz ederek onlara tamamen kişiselleştirilmiş, 
                    motive edici ve pratik günlük hedefler sunmaktır. Çıktıların net, samimi 
                    ve maddeler halinde olmalıdır.

                    KULLANICININ ÖNCELİKLİ HEDEFİ: {user_goal}
                    Lütfen tüm analizini ve tavsiyelerini bu hedefe ulaşmasını kolaylaştıracak şekilde tasarla.

                    Aşağıda kullanıcının aktivite özeti ve ML profili bulunmaktadır:
                    '{llm_context}'

                    Lütfen bu kullanıcı için şu formatta bir yanıt hazırlayın:
                    * **Profil Analizi:** Kullanıcının mevcut durumu ve '{user_data['Segment']}' profili hakkında hedefine yönelik kısa bir değerlendirme.
                    * **Hareket Hedefi:** Hareketsiz süreyi (sedanter) azaltmak için ofis veya ev ortamına uygun 1 pratik tavsiye.
                    * **Aktivite Hedefi:** Adım sayısını ve aktif dakikaları artırmak için gerçekçi ve spesifik 2 kolay tavsiye.
                    """

                model = genai.GenerativeModel(model_name="gemini-2.5-flash")
                response = model.generate_content(user_prompt)

                st.success("Analiz Tamamlandı!")
                st.info(response.text)

            except Exception as e:
                st.error(f"API isteği sırasında bir hata oluştu: {e}")