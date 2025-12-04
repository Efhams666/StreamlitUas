import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Fungsi Euclidean Distance
# ----------------------------------------------------
def euclidean_distance(a, b):
    """Menghitung jarak Euclidean antara dua titik"""
    return np.sqrt(np.sum((a - b) ** 2))

# ----------------------------------------------------
# Fungsi Normalisasi
# ----------------------------------------------------
def normalize_data(X):
    """Min-Max Normalization (0-1)"""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denominator = X_max - X_min
    denominator[denominator == 0] = 1  # Hindari division by zero
    return (X - X_min) / denominator, X_min, X_max

# ----------------------------------------------------
# KNN dengan Tie-Breaking
# ----------------------------------------------------
def knn_predict(x_train, y_train, x_test, k=3):
    """Prediksi dengan KNN dan penanganan seri"""
    distances = []
    
    # Hitung jarak ke semua data training
    for i in range(len(x_train)):
        distance = euclidean_distance(x_train[i], x_test)
        distances.append((distance, y_train[i]))
    
    # Urutkan berdasarkan jarak terkecil
    distances.sort(key=lambda x: x[0])
    
    # Ambil k tetangga terdekat
    neighbors = distances[:k]
    
    # Voting
    votes = {}
    for d, label in neighbors:
        votes[label] = votes.get(label, 0) + 1
    
    # Handle tie-breaking jika voting seri
    max_votes = max(votes.values())
    candidates = [label for label, count in votes.items() if count == max_votes]
    
    if len(candidates) > 1:
        # Pilih label dengan jarak terdekat di antara kandidat seri
        best_label = None
        min_distance = float('inf')
        for d, label in neighbors:
            if label in candidates and d < min_distance:
                min_distance = d
                best_label = label
        return best_label
    
    return max(votes, key=votes.get)

# ----------------------------------------------------
# Fungsi untuk membersihkan data (hanya 1-5)
# ----------------------------------------------------
def clean_data(df):
    """Membersihkan data: hanya menerima nilai 1-5"""
    df_clean = df.copy()
    
    # Untuk setiap kolom kecuali label
    for col in df_clean.columns:
        if col != "How would you rate your stress levels?":
            # Paksa nilai ke range 1-5
            df_clean[col] = df_clean[col].apply(
                lambda x: max(1, min(5, round(float(x)))) if pd.notnull(x) else 3
            ).astype(int)
    
    # Label juga kita bersihkan
    if "How would you rate your stress levels?" in df_clean.columns:
        df_clean["How would you rate your stress levels?"] = df_clean["How would you rate your stress levels?"].apply(
            lambda x: max(1, min(5, round(float(x)))) if pd.notnull(x) else 3
        ).astype(int)
    
    return df_clean

# ----------------------------------------------------
# STREAMLIT APP - VERSI LENGKAP
# ----------------------------------------------------
def main():
    # Konfigurasi halaman
    st.set_page_config(page_title="Student Stress Classifier", page_icon="📊", layout="wide")
    
    # Judul utama
    st.title("📊 Student Stress Classification - KNN Complete")
    st.markdown("---")
    
    # Sidebar untuk navigasi
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ["📁 Data Overview", "🔧 Preprocessing", "🎯 Prediction", "📈 Visualization"]
    )
    
    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv("Student Stress Factors.csv")
        if "Timestamp" in df.columns:
            df = df.drop(columns=["Timestamp"])
        return df
    
    df = load_data()
    
    # ====================================================
    # HALAMAN 1: DATA OVERVIEW
    # ====================================================
    if page == "📁 Data Overview":
        st.header("📁 Dataset Overview")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Dataset Info")
            st.write(f"**Jumlah Baris:** {df.shape[0]}")
            st.write(f"**Jumlah Kolom:** {df.shape[1]}")
            st.write(f"**Nama Kolom:**")
            for col in df.columns:
                st.write(f"- {col}")
        
        st.subheader("📋 Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Tampilkan distribusi setiap fitur
        st.subheader("📊 Distribution of Each Feature")
        feature_cols = [col for col in df.columns if col != "How would you rate your stress levels?"]
        
        for col in feature_cols:
            col1, col2 = st.columns([3, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(8, 3))
                df[col].hist(ax=ax, bins=20, edgecolor='black')
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            with col2:
                st.write(f"**{col}**")
                st.write(f"Min: {df[col].min()}")
                st.write(f"Max: {df[col].max()}")
                st.write(f"Mean: {df[col].mean():.2f}")
                st.write(f"Std: {df[col].std():.2f}")
    
    # ====================================================
    # HALAMAN 2: PREPROCESSING
    # ====================================================
    elif page == "🔧 Preprocessing":
        st.header("🔧 Data Preprocessing")
        
        # Pilihan preprocessing
        preprocessing_option = st.radio(
            "Pilih metode preprocessing:",
            ["Data Asli", "Normalisasi 0-1", "Skala 1-5 Saja"]
        )
        
        if preprocessing_option == "Normalisasi 0-1":
            st.subheader("🔧 Normalisasi Data (Min-Max Scaling)")
            
            # Pisahkan fitur dan label
            label_col = "How would you rate your stress levels?"
            feature_cols = [col for col in df.columns if col != label_col]
            
            X_original = df[feature_cols].values
            y_original = df[label_col].values
            
            # Normalisasi
            X_normalized, X_min, X_max = normalize_data(X_original)
            
            # Buat DataFrame normalisasi
            df_normalized = pd.DataFrame(
                X_normalized,
                columns=[f"{col} (Norm)" for col in feature_cols]
            )
            df_normalized[label_col] = y_original
            
            # Tampilkan
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Asli (5 baris pertama):**")
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.write("**Data Normalisasi (5 baris pertama):**")
                st.dataframe(df_normalized.head(), use_container_width=True)
            
            # Tampilkan parameter normalisasi
            st.subheader("📋 Parameter Normalisasi")
            param_df = pd.DataFrame({
                'Fitur': feature_cols,
                'Min Asli': X_min,
                'Max Asli': X_max,
                'Range': X_max - X_min
            })
            st.dataframe(param_df, use_container_width=True)
            
            # Visualisasi perbandingan
            st.subheader("📈 Perbandingan Sebelum & Sesudah")
            selected_feature = st.selectbox("Pilih fitur untuk visualisasi:", feature_cols)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Sebelum normalisasi
            ax1.hist(df[selected_feature], bins=20, color='skyblue', edgecolor='black')
            ax1.set_title(f'Before Normalization: {selected_feature}')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Frequency')
            
            # Sesudah normalisasi
            norm_col = f"{selected_feature} (Norm)"
            ax2.hist(df_normalized[norm_col], bins=20, color='lightcoral', edgecolor='black')
            ax2.set_title(f'After Normalization: {selected_feature}')
            ax2.set_xlabel('Normalized Value (0-1)')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download options
            st.subheader("💾 Download Options")
            col1, col2 = st.columns(2)
            with col1:
                csv_original = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data Asli",
                    data=csv_original,
                    file_name="student_stress_original.csv",
                    mime="text/csv"
                )
            with col2:
                csv_normalized = df_normalized.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data Normalisasi",
                    data=csv_normalized,
                    file_name="student_stress_normalized.csv",
                    mime="text/csv"
                )
        
        elif preprocessing_option == "Skala 1-5 Saja":
            st.subheader("🎯 Data Skala 1-5 (Likert Scale)")
            st.info("Semua nilai akan dipaksa ke skala 1-5")
            
            # Bersihkan data
            df_clean = clean_data(df)
            
            # Tampilkan
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Asli (5 baris pertama):**")
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.write("**Data Bersih 1-5 (5 baris pertama):**")
                st.dataframe(df_clean.head(), use_container_width=True)
            
            # Tampilkan distribusi
            st.subheader("📊 Distribusi Nilai 1-5")
            feature_cols = [col for col in df_clean.columns if col != "How would you rate your stress levels?"]
            
            for col in feature_cols[:3]:  # Tampilkan 3 fitur pertama
                counts = df_clean[col].value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(8, 3))
                bars = ax.bar(counts.index.astype(str), counts.values, color='lightgreen', edgecolor='black')
                
                # Tambahkan nilai di atas bar
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom')
                
                ax.set_title(f'Distribution of {col} (1-5 Scale)')
                ax.set_xlabel('Score (1-5)')
                ax.set_ylabel('Count')
                ax.set_ylim(0, max(counts.values) * 1.2)
                
                st.pyplot(fig)
            
            # Download cleaned data
            st.subheader("💾 Download Clean Data")
            csv_clean = df_clean.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data 1-5",
                data=csv_clean,
                file_name="student_stress_clean_1-5.csv",
                mime="text/csv"
            )
        
        else:
            st.write("**Data Asli (tanpa preprocessing):**")
            st.dataframe(df, use_container_width=True)
    
    # ====================================================
    # HALAMAN 3: PREDICTION
    # ====================================================
    elif page == "🎯 Prediction":
        st.header("🎯 Stress Level Prediction")
        
        # Pilih metode preprocessing untuk prediction
        st.subheader("1. Pilih Metode Preprocessing")
        method = st.radio(
            "Pilih metode data yang akan digunakan:",
            ["Normalisasi 0-1", "Skala 1-5 Saja"],
            horizontal=True
        )
        
        # Siapkan data berdasarkan pilihan
        label_col = "How would you rate your stress levels?"
        feature_cols = [col for col in df.columns if col != label_col]
        
        if method == "Normalisasi 0-1":
            X_original = df[feature_cols].values
            y = df[label_col].values
            X, X_min, X_max = normalize_data(X_original)
            use_normalized = True
        else:
            df_clean = clean_data(df)
            X = df_clean[feature_cols].values
            y = df_clean[label_col].values
            X_min = np.ones(len(feature_cols))  # dummy values
            X_max = np.ones(len(feature_cols)) * 5  # dummy values
            use_normalized = False
        
        # Input user
        st.subheader("2. Input Nilai Faktor Stres")
        st.info("Masukkan nilai untuk setiap faktor stres:")
        
        user_input = []
        
        # Buat 2 kolom untuk input
        cols = st.columns(2)
        for idx, col in enumerate(feature_cols):
            with cols[idx % 2]:
                if method == "Skala 1-5 Saja":
                    # Deskripsi untuk skala 1-5
                    descriptions = {
                        1: "1 - Sangat Rendah",
                        2: "2 - Rendah",
                        3: "3 - Sedang",
                        4: "4 - Tinggi",
                        5: "5 - Sangat Tinggi"
                    }
                    
                    val = st.select_slider(
                        f"{col}",
                        options=[1, 2, 3, 4, 5],
                        value=3,
                        format_func=lambda x: descriptions[x]
                    )
                else:
                    # Untuk normalisasi, gunakan slider dengan range asli
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    default_val = float(df[col].mean())
                    
                    val = st.slider(
                        f"{col} ({min_val:.0f}-{max_val:.0f})",
                        min_val,
                        max_val,
                        default_val
                    )
                
                user_input.append(val)
        
        user_input = np.array(user_input)
        
        # Normalisasi input jika menggunakan metode normalisasi
        if use_normalized:
            user_input_normalized = (user_input - X_min) / (X_max - X_min)
            user_input_for_prediction = user_input_normalized
        else:
            user_input_for_prediction = user_input
        
        # Tampilkan input user
        st.subheader("3. Input Anda")
        input_df = pd.DataFrame({
            'Faktor Stres': feature_cols,
            'Nilai Input': user_input
        })
        
        if use_normalized:
            input_df['Nilai Normalisasi'] = user_input_normalized
        
        st.dataframe(input_df, use_container_width=True)
        
        # Parameter KNN
        st.subheader("4. Parameter KNN")
        col1, col2, col3 = st.columns(3)
        with col1:
            k_value = st.slider("Nilai K (jumlah tetangga)", 1, 15, 5)
        with col2:
            show_neighbors = st.checkbox("Tampilkan tetangga terdekat", value=True)
        with col3:
            show_details = st.checkbox("Tampilkan detail perhitungan", value=False)
        
        # Tombol prediksi
        if st.button("🚀 Lakukan Prediksi", type="primary", use_container_width=True):
            # Lakukan prediksi
            prediction = knn_predict(X, y, user_input_for_prediction, k=k_value)
            
            # Tampilkan hasil
            st.markdown("---")
            
            # Header hasil dengan styling
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h1 style='color: {'#e74c3c' if prediction >= 4 else '#f39c12' if prediction == 3 else '#27ae60'};'>
                    🎯 Stress Level: {prediction}/5
                </h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretasi
            interpretations = {
                1: ("✅ Stres Sangat Rendah", "Kondisi mental sangat baik, tetap pertahankan!"),
                2: ("👍 Stres Rendah", "Kondisi masih terkendali dengan baik."),
                3: ("⚠️ Stres Sedang", "Perlu perhatian dan manajemen stres."),
                4: ("🚨 Stres Tinggi", "Butuh penanganan dan konsultasi."),
                5: ("🔥 Stres Sangat Tinggi", "Segera cari bantuan profesional.")
            }
            
            if prediction in interpretations:
                icon, desc = interpretations[prediction]
                st.success(f"**{icon} Interpretasi:** {desc}")
            
            # Progress bar visual
            st.progress(prediction/5)
            st.caption(f"Tingkat stres: {prediction}/5")
            
            # Jika diminta, tampilkan tetangga terdekat
            if show_neighbors:
                st.subheader(f"👥 {k_value} Tetangga Terdekat")
                
                # Hitung semua jarak
                distances = []
                for i in range(len(X)):
                    dist = euclidean_distance(X[i], user_input_for_prediction)
                    distances.append((dist, y[i], X[i]))
                
                # Urutkan dan ambil k terdekat
                distances.sort(key=lambda x: x[0])
                nearest = distances[:k_value]
                
                # Buat tabel tetangga
                neighbor_data = []
                for idx, (dist, label, features) in enumerate(nearest, 1):
                    # Format fitur
                    if use_normalized:
                        features_str = " | ".join([f"{x:.3f}" for x in features])
                    else:
                        features_str = " | ".join([f"{int(x)}" for x in features])
                    
                    neighbor_data.append({
                        'Rank': idx,
                        'Jarak': f"{dist:.4f}",
                        'Stress Level': label,
                        'Features': features_str
                    })
                
                # Tampilkan dalam expander
                with st.expander("Lihat Detail Tetangga", expanded=True):
                    st.table(pd.DataFrame(neighbor_data))
                
                # Analisis voting
                st.subheader("📊 Analisis Voting")
                
                # Hitung persentase voting
                votes = {}
                for _, label, _ in nearest:
                    votes[label] = votes.get(label, 0) + 1
                
                # Tampilkan hasil voting
                for stress_level in range(1, 6):
                    count = votes.get(stress_level, 0)
                    percentage = (count / k_value) * 100
                    
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col1:
                        st.write(f"**{stress_level}:**")
                    with col2:
                        st.progress(percentage/100)
                    with col3:
                        st.write(f"{percentage:.1f}% ({count}/{k_value})")
            
            # Detail perhitungan (jika diminta)
            if show_details:
                st.subheader("🔍 Detail Perhitungan")
                
                # Hitung jarak ke 5 contoh data training
                st.write("**Contoh perhitungan jarak ke 5 data training pertama:**")
                
                sample_data = []
                for i in range(min(5, len(X))):
                    dist = euclidean_distance(X[i], user_input_for_prediction)
                    
                    if use_normalized:
                        features_str = ", ".join([f"{x:.3f}" for x in X[i]])
                    else:
                        features_str = ", ".join([f"{int(x)}" for x in X[i]])
                    
                    sample_data.append({
                        'Data #': i+1,
                        'Features': features_str,
                        'Label': y[i],
                        'Jarak': f"{dist:.4f}"
                    })
                
                st.table(pd.DataFrame(sample_data))
                
                # Rumus Euclidean
                st.write("**Rumus Euclidean Distance:**")
                st.latex(r"d = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}")
                st.write(f"dimana n = {len(feature_cols)} fitur")
    
    # ====================================================
    # HALAMAN 4: VISUALIZATION
    # ====================================================
    else:
        st.header("📈 Data Visualization")
        
        # Pilih metode data
        viz_method = st.radio(
            "Pilih data untuk visualisasi:",
            ["Data Asli", "Data Normalisasi"],
            horizontal=True
        )
        
        # Siapkan data
        label_col = "How would you rate your stress levels?"
        feature_cols = [col for col in df.columns if col != label_col]
        
        if viz_method == "Data Normalisasi":
            X_original = df[feature_cols].values
            y = df[label_col].values
            X, _, _ = normalize_data(X_original)
        else:
            df_clean = clean_data(df)
            X = df_clean[feature_cols].values
            y = df_clean[label_col].values
        
        # Pilih tipe visualisasi
        viz_type = st.selectbox(
            "Pilih tipe visualisasi:",
            ["Scatter Plot", "Correlation Heatmap", "Feature Distribution", "3D Scatter Plot"]
        )
        
        if viz_type == "Scatter Plot":
            st.subheader("📊 Scatter Plot Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("Pilih fitur untuk sumbu X:", feature_cols)
                x_idx = feature_cols.index(x_feature)
            
            with col2:
                y_feature = st.selectbox("Pilih fitur untuk sumbu Y:", feature_cols)
                y_idx = feature_cols.index(y_feature)
            
            # Buat scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Warna berdasarkan stress level
            scatter = ax.scatter(
                X[:, x_idx], 
                X[:, y_idx], 
                c=y, 
                cmap='RdYlGn_r',  # Red-Yellow-Green (reversed)
                alpha=0.6,
                s=100,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Tambahkan titik input user jika ada di session state
            if 'last_prediction_input' in st.session_state:
                if viz_method == "Data Normalisasi" and 'last_normalized_input' in st.session_state:
                    user_point = st.session_state.last_normalized_input
                else:
                    user_point = st.session_state.last_prediction_input
                
                ax.scatter(
                    user_point[x_idx], 
                    user_point[y_idx], 
                    c='blue', 
                    s=200,
                    marker='*',
                    edgecolors='black',
                    linewidth=1.5,
                    label='Input Anda'
                )
                ax.legend()
            
            ax.set_xlabel(x_feature + (" (Normalized)" if viz_method == "Data Normalisasi" else ""))
            ax.set_ylabel(y_feature + (" (Normalized)" if viz_method == "Data Normalisasi" else ""))
            ax.set_title(f'Scatter Plot: {x_feature} vs {y_feature}')
            
            # Color bar
            plt.colorbar(scatter, label='Stress Level')
            plt.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Interpretasi
            st.info("""
            **Interpretasi Warna:**
            - 🟢 Hijau: Stress rendah (1-2)
            - 🟡 Kuning: Stress sedang (3)
            - 🔴 Merah: Stress tinggi (4-5)
            - ⭐ Biru: Posisi input Anda
            """)
        
        elif viz_type == "Correlation Heatmap":
            st.subheader("🔥 Correlation Heatmap")
            
            # Buat DataFrame untuk korelasi
            if viz_method == "Data Normalisasi":
                viz_df = pd.DataFrame(X, columns=[f"{col} (Norm)" for col in feature_cols])
            else:
                viz_df = pd.DataFrame(X, columns=feature_cols)
            viz_df['Stress Level'] = y
            
            # Hitung korelasi
            corr_matrix = viz_df.corr()
            
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Atur ticks
            ax.set_xticks(np.arange(len(corr_matrix.columns)))
            ax.set_yticks(np.arange(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_matrix.columns)
            
            # Tambahkan nilai di setiap sel
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=9)
            
            ax.set_title("Correlation Matrix")
            plt.colorbar(im)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Interpretasi korelasi dengan stress
            st.subheader("📈 Korelasi dengan Stress Level")
            
            # Ambil korelasi dengan stress level
            stress_corr = corr_matrix['Stress Level'].sort_values(ascending=False)
            
            # Buat bar chart
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            colors = ['red' if x > 0 else 'blue' for x in stress_corr.values]
            bars = ax2.barh(stress_corr.index, stress_corr.values, color=colors)
            
            ax2.set_xlabel('Correlation Coefficient')
            ax2.set_title('Correlation with Stress Level')
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # Tambahkan nilai di bar
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width if width >= 0 else width
                ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', 
                        va='center', 
                        ha='left' if width >= 0 else 'right',
                        color='black')
            
            plt.tight_layout()
            st.pyplot(fig2)
            
            st.info("""
            **Interpretasi Korelasi:**
            - **> 0.7**: Korelasi sangat kuat positif
            - **0.5 - 0.7**: Korelasi kuat positif
            - **0.3 - 0.5**: Korelasi moderat positif
            - **0.1 - 0.3**: Korelasi lemah positif
            - **-0.1 - 0.1**: Tidak ada korelasi
            - **Negatif**: Hubungan terbalik
            """)
        
        elif viz_type == "Feature Distribution":
            st.subheader("📊 Feature Distribution by Stress Level")
            
            selected_feature = st.selectbox("Pilih fitur untuk dianalisis:", feature_cols)
            feature_idx = feature_cols.index(selected_feature)
            
            # Buat box plot per stress level
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Kumpulkan data per stress level
            data_by_stress = []
            labels = []
            
            for stress_level in sorted(np.unique(y)):
                mask = y == stress_level
                data_by_stress.append(X[mask, feature_idx])
                labels.append(f"Stress {stress_level}")
            
            # Box plot
            box = ax.boxplot(data_by_stress, labels=labels, patch_artist=True)
            
            # Warna box
            colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
            for patch, color in zip(box['boxes'], colors[:len(labels)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_xlabel('Stress Level')
            ax.set_ylabel(selected_feature + (" (Normalized)" if viz_method == "Data Normalisasi" else ""))
            ax.set_title(f'Distribution of {selected_feature} by Stress Level')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Tampilkan statistik deskriptif
            st.subheader(f"📋 Statistik {selected_feature}")
            
            stats_data = []
            for stress_level in sorted(np.unique(y)):
                mask = y == stress_level
                feature_data = X[mask, feature_idx]
                
                stats_data.append({
                    'Stress Level': stress_level,
                    'Count': len(feature_data),
                    'Mean': np.mean(feature_data),
                    'Std': np.std(feature_data),
                    'Min': np.min(feature_data),
                    'Median': np.median(feature_data),
                    'Max': np.max(feature_data)
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        else:  # 3D Scatter Plot
            st.subheader("🌐 3D Scatter Plot")
            st.info("Visualisasi 3D dari 3 fitur terpilih")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_feature = st.selectbox("Pilih fitur X:", feature_cols, key='3d_x')
                x_idx = feature_cols.index(x_feature)
            with col2:
                y_feature = st.selectbox("Pilih fitur Y:", feature_cols, key='3d_y')
                y_idx = feature_cols.index(y_feature)
            with col3:
                z_feature = st.selectbox("Pilih fitur Z:", feature_cols, key='3d_z')
                z_idx = feature_cols.index(z_feature)
            
            # Buat plot 3D
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Scatter plot 3D
            scatter = ax.scatter3D(
                X[:, x_idx], 
                X[:, y_idx], 
                X[:, z_idx], 
                c=y, 
                cmap='RdYlGn_r',
                s=50,
                alpha=0.6,
                depthshade=True
            )
            
            ax.set_xlabel(x_feature + (" (Norm)" if viz_method == "Data Normalisasi" else ""))
            ax.set_ylabel(y_feature + (" (Norm)" if viz_method == "Data Normalisasi" else ""))
            ax.set_zlabel(z_feature + (" (Norm)" if viz_method == "Data Normalisasi" else ""))
            ax.set_title(f'3D Visualization: {x_feature}, {y_feature}, {z_feature}')
            
            plt.colorbar(scatter, label='Stress Level')
            
            st.pyplot(fig)
            
            # Tambahkan kontrol rotasi
            st.slider("Rotasi sumbu X", 0, 360, 30, key='rot_x')
            st.slider("Rotasi sumbu Y", 0, 360, 45, key='rot_y')
            
            # Note tentang 3D plot
            st.info("""
            **Tips untuk plot 3D:**
            - Gunakan mouse untuk memutar plot (di jendela matplotlib)
            - Zoom in/out dengan scroll mouse
            - Warna menunjukkan tingkat stres
            """)
        
        # Download gambar
        if st.button("📥 Download Visualization as PNG"):
            # Untuk sementara, kita simpan gambar terakhir
            st.info("Fitur download gambar akan diimplementasikan pada versi berikutnya")
    
    # ====================================================
    # FOOTER
    # ====================================================
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Student Stress Classifier | KNN Algorithm | Built with Streamlit</p>
            <p>Data Normalization: Min-Max Scaling (0-1) | Data Cleaning: 1-5 Scale</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------------------------------
# RUN APLIKASI
# ----------------------------------------------------
if __name__ == "__main__":
    # Tambahkan session state untuk menyimpan input user
    if 'last_prediction_input' not in st.session_state:
        st.session_state.last_prediction_input = None
    if 'last_normalized_input' not in st.session_state:
        st.session_state.last_normalized_input = None
    
    # Jalankan aplikasi
    main()