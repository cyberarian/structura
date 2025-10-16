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
            display: block !important; /* Make button a block element */
            width: auto !important; /* Let the button size to its content */
            margin: 0 auto !important; /* Center the block-level button within its container */
        }}
        
        .stButton > button:hover {{
            background-color: #E03030 !important; /* Darker red on hover */
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

        /* This class is no longer needed for centering but kept for compatibility */
        /* .button-container {{
            text-align: center;
            padding-top: 1rem;
        }} */

        /* Target the column containing the button to center its content */
        div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stButton"] {{ text-align: center; }}
        </style>

        <div class="landing-content">
            <h1>Structura: RAG-Powered App for Construction Engineering Document</h1>
            <h2>
            Intelligent Document Processing & Q&A for Engineering
            </h2>
            <p>
            Structura mengubah dokumen teknis statis—seperti gambar P&ID, kontrak, dan laporan—menjadi aset informasi yang interaktif dan terkelola. Kurangi waktu pencarian manual dan tingkatkan akurasi data proyek Anda dengan kekuatan AI.
            </p>
            <p>
            Kemampuan Utama:
            <ul>
                <li>Tanya Jawab Kontekstual: Ajukan pertanyaan dalam bahasa alami tentang dokumen apa pun yang telah diproses. Dapatkan jawaban akurat yang dirumuskan AI hanya dari konten dokumen tersebut, bukan dari internet.</li>
                <li>Ekstraksi MDR Otomatis: Unggah dokumen teknis dan biarkan AI (Gemini Pro) secara otomatis mengisi Master Document Register (MDR) Anda. Ekstrak puluhan atribut penting seperti Nomor Dokumen, Revisi, Judul, dan Tanggal secara instan.</li>
                <li>Manajemen Rekod Cerdas: Setiap dokumen yang diproses disimpan dalam basis data, lengkap dengan metadata, riwayat pemrosesan, dan kata kunci yang disarankan AI untuk kemudahan penelusuran dan tata kelola.</li>
            </ul>
            </p>
            <h3>Siap Mengubah Alur Kerja Dokumen Anda?</h3
            <p>
            Hentikan kebingungan dokumen dan mulailah membangun dengan kepastian informasi. Structura adalah co-pilot cerdas Anda di lokasi proyek dan di kantor..
            </p>
    """, unsafe_allow_html=True)

    # --- Centered Button using st.columns ---
    # Create three columns; the middle one will hold the button.
    # The outer columns act as spacers. Adjust the ratio for different widths.
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Masuk ke Sistem", key="enter_system_button", use_container_width=True):
            st.session_state.show_dashboard = True
            st.rerun() # Rerun to switch view

    st.markdown('</div>', unsafe_allow_html=True) # Close landing-content div
