# LinkedIn Post — FraudShield-Lite

**Status:** Draft  
**Target:** Senior Data Scientist / DANA (fintech/DS roles)  
**Max chars:** 2000

---

**Repost + comment "🔍" if you want the full deep-dive.**

Jujur, setiap liat notif "Pembayaran Gagal — Transaksi Mencurigakan" di dompet digital… gw senyum sendiri. Di balik layar, ada model ML yang bela-belain ngeblokir fraud dalam milidetik sambil jangan sampe bikin pengalaman user boncos.

Baru aja gw wrap project yang paling seru tahun ini — **FraudShield-Lite** 🛡️, fraud detection system khusus buat e-wallet. Here's what went down:

📊 **Akurasi bikin kaget — dalam artian bagus:**
- **ROC-AUC 0.9999** (yup, you read that right)
- **Precision 91.9%** — FP kecil, tim investigasi gak kewalahan
- **Recall 99.8%** — hampir semua fraud kejiret
- **F1-Score 95.6%**

💰 **Dari angka model ke dampak bisnis:**
- Cost matrix real: Rp 500K per FN (fraud lolos) vs Rp 15K per FP (false alarm) — rasio **33:1**
- Iterasi pake **Walk-Forward CV** (3-fold, time-respecting) biar gak bocor masa depan
- **97% fraud bisa diredam** = Rp 84.79 Milyar setahun potensi saving
- **ROI 3.300%** 🚀

🛠 **Stack yang dipake:**
- XGBoost with 29 engineered features
- **SHAP** buat explainability — biar tim bisnis percaya, bukan cuma black box
- **FastAPI** backend & **Streamlit** dashboard (live demo-ready)
- Dataset: PaySim (~500K transaksi) — public tapi realistis banget

🔥 **Kenapa gw post ini:**
Gue mau apply ke **DANA** (hi, tim DANA 👋) sebagai Senior Data Scientist. Project ini adalah bukti konkret: dari problem framing, cost-sensitive modeling, sampe deployable API — end-to-end.

**Pesan buat rekan-rekan data scientist di luar sana:** Jangan Cuma ngejar AUC. Kaitkan setiap metric ke cost, ke user experience, ke rupiah. Itulah bedanya DS yang nge-push impact vs yang cuma jalanin notebook.

**Full code & documentation udah di publik:**
👉 https://github.com/hadijayyy/fraudshield-lite

Kalo lo tertarik bahas fraud detection, career switching ke fintech, atau kolaborasi — let's connect! Drop comment atau DM aja 🚀

---

*#FraudDetection #DataScience #MachineLearning #Fintech #DANA #XGBoost #SHAP #AIforGood*
