import streamlit as st

def display_about_tab():
    """Menampilkan konten tab Tentang."""

    st.header("ℹ️ Tentang Structura")

    st.markdown("""
    **Structura** adalah aplikasi **Pemrosesan Dokumen Cerdas (Intelligent Document Processing - IDP) & Tanya Jawab (Q&A)** yang dirancang untuk meningkatkan efisiensi interaksi organisasi dengan aset informasinya, khususnya dalam konteks dokumen teknis dan rekayasa. Aplikasi ini memanfaatkan kecerdasan buatan (AI) untuk mengotomatisasi ekstraksi teks, analisis konten, dan penemuan informasi, mengubah dokumen statis menjadi sumber pengetahuan yang dinamis dan terkelola.

    Desain Structura mengadopsi prinsip-prinsip yang selaras dengan standar manajemen rekod (seperti ISO 15489), dengan fokus pada penciptaan metadata, keterlacakan proses, dan akses terkontrol untuk mendukung tata kelola informasi yang baik. Meskipun tidak dimaksudkan sebagai sistem Manajemen Dokumen dan Rekod Elektronik (EDRMS) yang lengkap, Structura berfungsi sebagai komponen pendukung yang kuat dalam ekosistem manajemen rekod digital.

    ### Kemampuan Inti
    Structura meningkatkan alur kerja dokumen melalui fungsi-fungsi berikut:
    *   **Ekstraksi Metadata Terstruktur (MDR):** Fitur unggulan untuk **Master Document Register (MDR)**. Menggunakan AI (Gemini Pro) untuk menganalisis dokumen teknis (seperti gambar P&ID atau kontrak) dan secara otomatis mengekstrak metadata kunci seperti Nomor Dokumen, Revisi, Judul, Tanggal, dan puluhan atribut lainnya ke dalam format yang terstruktur dan dapat diedit.
    *   **Tanya Jawab Kontekstual (RAG):** Pengguna dapat mengajukan pertanyaan dalam bahasa alami tentang isi dokumen yang telah diproses. Sistem menggunakan model RAG canggih (FAISS untuk pencarian vektor, BM25 untuk pencarian leksikal, dan LLM dari Groq) untuk memberikan jawaban yang akurat dan hanya berdasarkan konteks yang tersedia di dalam dokumen.
    *   **Pemrosesan Dokumen Fleksibel:** Mendukung berbagai format input (PDF, PNG, JPG, TXT, dll.). Menyediakan opsi ekstraksi teks langsung (untuk PDF digital) atau penggunaan layanan OCR terintegrasi (Groq, Google, Mistral, Tesseract) yang dapat dikonfigurasi oleh admin.
    *   **Analisis Konten & Rekod:** Selama pemrosesan, sistem menganalisis isi dokumen untuk memberikan *saran* klasifikasi, mengekstraksi kata kunci relevan, dan mengidentifikasi potensi indikator sensitivitas data. Fitur ini berfungsi sebagai alat bantu untuk penilaian awal dan pengorganisasian rekod.
    *   **Manajemen Basis Data & Kontrol Akses:** Semua metadata rekod disimpan dalam basis data SQLite. Aplikasi menerapkan kontrol akses berbasis peran ("admin", "creator"), di mana admin memiliki hak penuh untuk mengelola rekod (edit/hapus), mengonfigurasi sistem, dan mengakses semua data, sementara peran lain memiliki akses terbatas.
    *   **Ekstraksi Tabel Cerdas:** Secara otomatis mendeteksi, mengekstrak, dan menyimpan data tabel dari dokumen PDF. Dalam sesi tanya jawab, sistem dapat menyajikan kembali data tabel lengkap dalam format Markdown untuk memberikan jawaban yang lebih detail dan akurat.

    ### Fondasi Teknis dan Penanganan Data
    Dibangun dengan Python dan Streamlit, Structura menggunakan basis data **SQLite** untuk manajemen metadata rekod dan **FAISS** untuk pengindeksan vektor. Aplikasi ini dapat diintegrasikan dengan berbagai penyedia layanan AI seperti **Google (Gemini Pro), Groq (Llama), Mistral**, dan lainnya. Teks hasil pemrosesan, indeks FAISS, dan berkas terkait disimpan secara lokal di direktori `processed_docs`. Penggunaan layanan cloud untuk OCR atau analisis AI bersifat *opsional dan dapat dikonfigurasi*, memastikan kontrol atas residensi data inti tetap berada di lingkungan pengguna.

    ### Tujuan Penggunaan
    Structura dirancang untuk organisasi atau unit kerja yang bertujuan untuk:
    - Mempercepat pemrosesan, pemahaman, dan analisis koleksi dokumen digital.
    - Mengotomatisasi pengisian **Master Document Register (MDR)** dari dokumen teknis.
    - Menerapkan metode pencarian dan tanya jawab konten dokumen yang efisien dan kontekstual.
    - Mengotomatisasi langkah awal dalam analisis rekod, seperti ekstraksi kata kunci dan saran klasifikasi, sebagai input untuk sistem EDRMS.
    - Menyediakan data terstruktur (metadata pemrosesan dan analisis) yang dapat mendukung proses klasifikasi, penilaian, dan penelusuran rekod, meningkatkan *findability* dan *traceability*.
    """)
    
    st.subheader("⚠️ Penting")
    st.info("""
        * Untuk implementasi dalam lingkungan produksi, Structura memerlukan peningkatan infrastruktur
        * Berminat membangun solusi serupa untuk manajemen dokumen di organisasi Anda? Hubungi pengembang.
    """)
    # Versi dan detail teknis
    st.markdown("---")
    st.caption("Structura, Built by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")
