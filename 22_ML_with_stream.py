# ============================================================
# التثبيت: pip install streamlit scikit-learn pandas plotly xgboost
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, r2_score,
                             mean_absolute_error, classification_report)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ⚙️ إعدادات الصفحة + CSS المخصص
# ============================================================

st.set_page_config(
    page_title="🤖 ML Platform | ATHAR",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- التصميم  ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');

/* الأساس */
html, body, [class*="st-"] {direction: rtl; text-align: right; font-family: 'Cairo', sans-serif !important;}
.stApp {background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0f0f2e 100%);}
header[data-testid="stHeader"] {background: rgba(10,10,26,0.8); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(99,102,241,0.15);}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #12122a 0%, #1a1a3e 100%); border-left: 1px solid rgba(99,102,241,0.15);}

/* إخفاء العناصر الافتراضية */
#MainMenu, footer, header .stDeployButton {visibility: hidden;}
div[data-testid="stDecoration"] {display: none;}

/* العناوين */
h1 {background: linear-gradient(135deg, #fff 0%, #a5b4fc 50%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900 !important; font-size: 2.5rem !important; text-align: center !important; padding: 0.5rem 0 !important;}
h2 {color: #e2e8f0 !important; border-bottom: 2px solid rgba(99,102,241,0.3); padding-bottom: 0.5rem;}
h3 {color: #c4b5fd !important;}

/* البطاقات (metric) */
div[data-testid="stMetric"] {background: rgba(99,102,241,0.08); backdrop-filter: blur(10px); border: 1px solid rgba(99,102,241,0.2); border-radius: 16px; padding: 1.2rem; transition: all 0.3s;}
div[data-testid="stMetric"]:hover {border-color: rgba(99,102,241,0.5); transform: translateY(-2px); box-shadow: 0 8px 25px rgba(99,102,241,0.15);}
div[data-testid="stMetric"] label {color: #94a3b8 !important; font-size: 0.9rem !important;}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {color: #fff !important; font-size: 2rem !important; font-weight: 900 !important;}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {color: #10b981 !important;}

/* الأزرار */
.stButton > button {background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.6rem 2rem !important; font-weight: 700 !important; transition: all 0.3s !important; font-family: 'Cairo' !important;}
.stButton > button:hover {transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;}

/* الجدول */
div[data-testid="stDataFrame"] {border: 1px solid rgba(99,102,241,0.2); border-radius: 12px; overflow: hidden;}

/* Slider */
div[data-testid="stSlider"] label {color: #c4b5fd !important;}

/* الشريط الجانبي */
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {-webkit-text-fill-color: #e2e8f0; background: none;}
section[data-testid="stSidebar"] label {color: #c4b5fd !important;}
section[data-testid="stSidebar"] .stRadio > label {color: #e2e8f0 !important;}

/* تبويبات */
.stTabs [data-baseweb="tab-list"] {gap: 8px; background: rgba(99,102,241,0.05); border-radius: 12px; padding: 4px;}
.stTabs [data-baseweb="tab"] {border-radius: 10px; color: #94a3b8; font-weight: 600; font-family: 'Cairo' !important;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important;}
.stTabs [data-baseweb="tab-panel"] {background: rgba(99,102,241,0.03); border-radius: 0 0 12px 12px; padding: 1rem;}

/* Expander */
.streamlit-expanderHeader {background: rgba(99,102,241,0.08) !important; border-radius: 12px !important; color: #c4b5fd !important; font-weight: 600 !important;}

/* النصوص */
p, li, span, div {color: #cbd5e1;}
.stMarkdown a {color: #818cf8 !important;}

/* Selectbox / Multiselect */
div[data-baseweb="select"] {border-radius: 10px !important;}

/* Success/Info */
div.stSuccess {background: rgba(16,185,129,0.1) !important; border: 1px solid rgba(16,185,129,0.3) !important; border-radius: 12px !important; color: #6ee7b7 !important;}
div.stInfo {background: rgba(6,182,212,0.1) !important; border: 1px solid rgba(6,182,212,0.3) !important; border-radius: 12px !important;}

/* Plotly تعديل الخلفية */
.js-plotly-plot .plotly .main-svg {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🎯 الشريط الجانبي
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ لوحة التحكم")
    st.markdown("---")

    task = st.radio("📋 **اختر المهمة:**", [
        "🏠 توقع أسعار البيوت",
        "🏥 تشخيص سرطان الثدي"
    ], index=0)

    st.markdown("---")
    test_size = st.slider("📐 **نسبة الاختبار:**", 0.1, 0.4, 0.2, 0.05)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:1rem; background:rgba(99,102,241,0.08); border-radius:12px; border:1px solid rgba(99,102,241,0.2); margin-top:1rem;">
        <div style="font-size:1.5rem;">🤖</div>
        <div style="color:#a5b4fc; font-weight:700; font-size:0.9rem;">AI Bootcamp ATHAR</div>
        <div style="color:#64748b; font-size:0.75rem;">الدرس 07 — Streamlit</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 🏠 REGRESSION
# ============================================================

if "أسعار" in task:
    st.markdown("# 🏠 منصة توقع أسعار البيوت")
    st.markdown("<p style='text-align:center;color:#94a3b8;font-size:1.1rem;margin-top:-1rem;'>California Housing Dataset — 20,640 بيت حقيقي</p>", unsafe_allow_html=True)

    @st.cache_data
    def load_housing():
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["Price"] = data.target
        return df, data

    df, data = load_housing()

    # --- الإحصائيات ---
    st.markdown("### 📊 نظرة عامة")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 عدد البيوت", f"{len(df):,}")
    c2.metric("📋 المتغيرات", f"{df.shape[1] - 1}")
    c3.metric("💰 متوسط السعر", f"${df['Price'].mean()*100:,.0f}K")
    c4.metric("📈 أعلى سعر", f"${df['Price'].max()*100:,.0f}K")

    # --- التبويبات ---
    tab1, tab2, tab3, tab4 = st.tabs(["📈 استكشاف", "🤖 التدريب", "📊 الأهمية", "🔮 جرّب بنفسك"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="Price", nbins=50, title="توزيع الأسعار",
                             color_discrete_sequence=["#6366f1"])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Cairo"),
                xaxis=dict(gridcolor="rgba(99,102,241,0.1)", title="السعر ($100K)"),
                yaxis=dict(gridcolor="rgba(99,102,241,0.1)", title="العدد"),
                title_font=dict(size=18, color="#e2e8f0")
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            corr_feature = st.selectbox("اختر متغير للمقارنة:", data.feature_names, index=0)
            fig2 = px.scatter(df, x=corr_feature, y="Price", title=f"{corr_feature} vs السعر",
                            color_discrete_sequence=["#06b6d4"], opacity=0.4)
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Cairo"),
                xaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
                yaxis=dict(gridcolor="rgba(99,102,241,0.1)", title="السعر"),
                title_font=dict(size=18, color="#e2e8f0")
            )
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 عرض البيانات الخام"):
            st.dataframe(df.head(20), use_container_width=True)

    with tab2:
        features = st.multiselect("📋 اختر المتغيرات:", list(data.feature_names), default=list(data.feature_names))

        if features:
            X = df[features]; y = df["Price"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = LinearRegression().fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            m1, m2, m3 = st.columns(3)
            m1.metric("📊 R² Score", f"{r2:.4f}", delta=f"{(r2-0.5)*100:+.1f}%")
            m2.metric("📊 MAE", f"${mae*100:,.0f}K")
            m3.metric("📦 حجم التدريب", f"{len(X_train):,}")

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=y_test.values, y=y_pred, mode='markers', name='التوقعات',
                                     marker=dict(color='#6366f1', size=4, opacity=0.5)))
            fig3.add_trace(go.Scatter(x=[0,5], y=[0,5], mode='lines', name='الخط المثالي',
                                     line=dict(color='#ef4444', dash='dash', width=2)))
            fig3.update_layout(
                title="الحقيقي vs المتوقع", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Cairo"),
                xaxis=dict(title="السعر الحقيقي", gridcolor="rgba(99,102,241,0.1)"),
                yaxis=dict(title="السعر المتوقع", gridcolor="rgba(99,102,241,0.1)"),
                title_font=dict(size=18, color="#e2e8f0"), legend=dict(font=dict(color="#94a3b8"))
            )
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        if 'model' in dir() and hasattr(model, 'coef_') and features:
            imp = pd.DataFrame({"المتغير": features, "التأثير": np.abs(model.coef_)}).sort_values("التأثير", ascending=True)
            fig4 = px.bar(imp, x="التأثير", y="المتغير", orientation="h", title="أهمية المتغيرات",
                         color="التأثير", color_continuous_scale=["#312e81","#6366f1","#06b6d4"])
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Cairo"),
                xaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
                yaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
                title_font=dict(size=18, color="#e2e8f0"), showlegend=False
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("🔹 درّب الموديل أولاً في تبويب 'التدريب'")

    with tab4:
        st.markdown("### 🔮 توقع تفاعلي")
        tc1, tc2 = st.columns(2)
        with tc1:
            user_income = st.slider("💵 متوسط الدخل:", 0.5, 15.0, 4.0, 0.5)
            user_rooms = st.slider("🏠 متوسط الغرف:", 1.0, 10.0, 5.0, 0.5)
        with tc2:
            user_age = st.slider("📅 عمر البيت:", 1.0, 52.0, 20.0, 1.0)
            user_pop = st.slider("👥 عدد السكان:", 100, 5000, 1000, 100)

        if features and 'model' in dir():
            sample = np.zeros((1, len(features)))
            mapping = {"MedInc": user_income, "AveRooms": user_rooms, "HouseAge": user_age, "Population": user_pop}
            for k, v in mapping.items():
                if k in features: sample[0][features.index(k)] = v
            pred = model.predict(scaler.transform(sample))[0]
            st.success(f"💰 السعر المتوقع: **${max(0,pred)*100:,.0f}K**")

# ============================================================
# 🏥 CLASSIFICATION
# ============================================================

else:
    st.markdown("# 🏥 منصة التشخيص الطبي الذكي")
    st.markdown("<p style='text-align:center;color:#94a3b8;font-size:1.1rem;margin-top:-1rem;'>Breast Cancer Wisconsin — 569 حالة طبية حقيقية</p>", unsafe_allow_html=True)

    @st.cache_data
    def load_cancer():
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        df["التشخيص"] = df["target"].map({0: "❌ خبيث", 1: "✅ سليم"})
        return df, data

    df, data = load_cancer()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 إجمالي الحالات", len(df))
    c2.metric("✅ سليم", (df.target==1).sum())
    c3.metric("❌ خبيث", (df.target==0).sum())
    c4.metric("📋 القياسات", data.data.shape[1])

    tab1, tab2, tab3 = st.tabs(["🏆 مقارنة الموديلات", "📋 Confusion Matrix", "📊 أهمية المتغيرات"])

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 اختر الموديلات")
        all_models = {
            "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        model_names = st.multiselect("", list(all_models.keys()), default=list(all_models.keys()))

    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    with tab1:
        if model_names:
            results = []
            for name in model_names:
                m = all_models[name]; m.fit(X_train, y_train)
                acc = m.score(X_test, y_test)
                cv = cross_val_score(m, X, y, cv=5).mean()
                results.append({"الموديل": name, "الدقة": acc, "Cross-Val": cv, "الفرق": abs(acc-cv)})

            res_df = pd.DataFrame(results).sort_values("Cross-Val", ascending=False)

            st.dataframe(
                res_df.style.format({"الدقة":"{:.2%}", "Cross-Val":"{:.2%}", "الفرق":"{:.2%}"})
                .background_gradient(subset=["Cross-Val"], cmap="Purples"),
                hide_index=True, use_container_width=True
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(name="الدقة", x=res_df["الموديل"], y=res_df["الدقة"],
                                marker_color="#6366f1", marker_line=dict(width=0)))
            fig.add_trace(go.Bar(name="Cross-Val", x=res_df["الموديل"], y=res_df["Cross-Val"],
                                marker_color="#06b6d4", marker_line=dict(width=0)))
            fig.update_layout(
                barmode="group", title="مقارنة الموديلات",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Cairo"),
                xaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
                yaxis=dict(gridcolor="rgba(99,102,241,0.1)", tickformat=".0%"),
                title_font=dict(size=18, color="#e2e8f0"),
                legend=dict(font=dict(color="#94a3b8"))
            )
            st.plotly_chart(fig, use_container_width=True)

            best_name = res_df.iloc[0]["الموديل"]
            st.success(f"🏆 الأفضل: **{best_name}** — Cross-Val: **{res_df.iloc[0]['Cross-Val']:.2%}**")

    with tab2:
        if model_names:
            best_name = pd.DataFrame(results).sort_values("Cross-Val", ascending=False).iloc[0]["الموديل"]
            best_m = all_models[best_name]; best_m.fit(X_train, y_train)
            y_pred = best_m.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="التوقع", y="الحقيقي", color="العدد"),
                              x=["خبيث", "سليم"], y=["خبيث", "سليم"],
                              color_continuous_scale=["#1e1b4b","#4338ca","#6366f1","#818cf8","#a5b4fc"])
            fig_cm.update_layout(
                title=f"Confusion Matrix — {best_name}",
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", family="Cairo"),
                title_font=dict(size=18, color="#e2e8f0")
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            tn, fp, fn, tp = cm.ravel()
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("✅ TN (خبيث صح)", tn)
            e2.metric("✅ TP (سليم صح)", tp)
            e3.metric("⚠️ FP (فات خبيث!)", fp)
            e4.metric("⚠️ FN (قلق زايد)", fn)

    with tab3:
        if "Random Forest" in model_names:
            rf = all_models["Random Forest"]; rf.fit(X_train, y_train)
            imp = pd.DataFrame({"المتغير": data.feature_names, "الأهمية": rf.feature_importances_}
                             ).sort_values("الأهمية", ascending=False).head(15)
            fig_imp = px.bar(imp, x="الأهمية", y="المتغير", orientation="h",
                           title="أهم 15 متغير (Random Forest)",
                           color="الأهمية", color_continuous_scale=["#312e81","#6366f1","#06b6d4","#10b981"])
            fig_imp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Cairo"),
                xaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
                yaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
                title_font=dict(size=18, color="#e2e8f0"), showlegend=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("🔹 أضف Random Forest من الشريط الجانبي لرؤية أهمية المتغيرات")

# --- التذييل ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1.5rem; color:#64748b;">
    <span style="font-size:1.2rem;">🤖</span> <strong style="color:#a5b4fc;">AI Bootcamp ATHAR</strong> — منصة Machine Learning الاحترافية
</div>
""", unsafe_allow_html=True)
