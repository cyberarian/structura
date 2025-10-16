import streamlit as st
import base64
import os

def get_base64_image():
    img_path = os.path.join(os.path.dirname(__file__), "images_dir", "landing3.png")
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def show_landing_page():
    # Initialize session state
    if 'show_admin' not in st.session_state:
        st.session_state['show_admin'] = False

    try:
        img_base64 = get_base64_image()
        bg_image = f"data:image/jpg;base64,{img_base64}"
    except Exception as e:
        print(f"Error loading background image: {e}")
        bg_image = "none"

    # Inject custom HTML/CSS with proper background handling
    st.markdown(f"""
        <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                        url("{bg_image}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
            background-color: #1E1E1E !important; /* Fallback color */
        }}
        
        .landing-content {{
            animation: fadeIn 1s ease-in;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            max-width: 1400px;
            margin: 0.5rem auto;
            color: white;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }}
        
        .stButton > button {{
            background-color: #FF4B4B !important;
            color: white !important;
            padding: 12px 30px !important;
            border-radius: 10px !important;
            border: none !important;
            font-size: 1.1rem !important;
            font-weight: 500 !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            margin-top: 0.3rem !important;
            display: block !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }}
        
        .stButton > button:hover {{
            background-color: #FF3333 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(255, 75, 75, 0.3) !important;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @media (max-width: 768px) {{
            .landing-content {{
                margin: 2rem 1rem;
                padding: 2rem;
            }}
        }}

        .button-container {{
            text-align: center;
            padding-top: 1rem;
        }}
        </style>

        <div class="landing-content">
            <h2>Structura: RAG-Powered App for Construction Engineering Document</h2>
            <p>
            Selamat Datang di Era Baru Manajemen Dokumen Teknik
            </p>
            <p>
            Dokumen proyek konstruksi sering kali kompleks, tersebar, dan sulit dinavigasi. Menghabiskan waktu berjam-jam mencari spesifikasi teknis, standar, atau gambar tertentu? Structura hadir untuk mengubah itu.
            </p>
            <p>
            Structura adalah aplikasi revolusioner berbasis Retrieval-Augmented Generation (RAG) yang dirancang khusus untuk para profesional teknik konstruksi, manajer proyek, dan kontraktor. Kami mengubah tumpukan dokumen Anda menjadi sumber pengetahuan yang cerdas dan mudah diakses.
            <ul>
                <li>💡 Temukan Jawaban, Bukan Sekadar Kata Kunci.</li>
                <li>⏱️ Efisiensi Waktu yang Maksimal.</li>
                <li>🔒 Keandalan dan Akurasi Data Proyek.</li>
            </ul>
            </p>
            <h3>Siap Mengubah Alur Kerja Dokumen Anda?</h3
            <p>
            Hentikan kebingungan dokumen dan mulailah membangun dengan kepastian informasi. Structura adalah co-pilot cerdas Anda di lokasi proyek dan di kantor..
            </p>
            <div class="button-container">
    """, unsafe_allow_html=True)

    # --- Button Container ---
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Masuk ke Sistem", key="enter_system_button"):
        st.session_state.show_dashboard = True
        st.rerun() # Rerun to switch view
    st.markdown('</div>', unsafe_allow_html=True) # Close button container

    st.markdown('</div>', unsafe_allow_html=True) # Close landing-content div
