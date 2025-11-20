import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search, get_model
import datetime
import pandas as pd
import os
import csv
import torch  # <-- используется для нарезки тензора эмбеддингов

st.set_page_config(page_title="Проверка фраз ЮЛ", layout="centered")

# Новогодние стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1a6e1a;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .snowflake {
        color: #87CEEB;
        font-size: 1.5rem;
        margin: 0 5px;
        animation: gentleFloat 3s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes gentleFloat {
        0%, 100% { 
            transform: translateY(0px) rotate(0deg); 
        }
        50% { 
            transform: translateY(-8px) rotate(180deg); 
        }
    }
    
    .snowflake:nth-child(2n) {
        animation-delay: 0.5s;
    }
    .snowflake:nth-child(3n) {
        animation-delay: 1s;
    }
    .snowflake:nth-child(4n) {
        animation-delay: 1.5s;
    }
    
    .christmas-banner {
        background: linear-gradient(90deg, #1a6e1a, #4caf50, #1a6e1a);
        padding: 12px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 8px rgba(26, 110, 26, 0.3);
    }
    
    .snow-row {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
        margin: 10px 0;
    }
    
    .christmas-card {
        background: linear-gradient(135deg, #f8fff8 0%, #e8f5e8 100%);
        padding: 16px;
        border-radius: 12px;
        border: 2px solid #1a6e1a;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(26,110,26,0.1);
    }
    
    .christmas-card-highlight {
        background: linear-gradient(135deg, #fff9e6 0%, #ffefbf 100%);
        border: 2px solid #ffd700;
        box-shadow: 0 4px 8px rgba(255,215,0,0.3);
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Новогодний баннер
st.markdown("""
<div class="christmas-banner">
    🎄 С Наступающим Новым Годом! 🎄
</div>
""", unsafe_allow_html=True)

# Заголовок с новогодними украшениями
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Верхний ряд снежинок
    st.markdown("""
    <div class="snow-row">
        <span class="snowflake">❄</span>
        <span class="snowflake">❅</span>
        <span class="snowflake">❆</span>
        <span class="snowflake">•</span>
        <span class="snowflake">❄</span>
        <span class="snowflake">❅</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Главный заголовок
    st.markdown('<h1 class="main-header">🤖 Проверка фраз</h1>', unsafe_allow_html=True)
    
    # Нижний ряд иконок
    st.markdown("""
    <div class="snow-row">
        <span class="snowflake">⭐</span>
        <span class="snowflake">🎄</span>
        <span class="snowflake">🎁</span>
        <span class="snowflake">🕯️</span>
        <span class="snowflake">⭐</span>
        <span class="snowflake">🎄</span>
    </div>
    """, unsafe_allow_html=True)

LOG_FILE = "query_log.csv"

# 🔧 Логирование
def log_query(query, semantic_count, keyword_count, status):
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["time", "query", "semantic_results", "keyword_results", "status"])
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query.strip(),
            semantic_count,
            keyword_count,
            status
        ])

@st.cache_data
def get_data():
    df = load_all_excels()
    model = get_model()
    # рассчитываем эмбеддинги для полной таблицы и сохраняем в attrs
    df.attrs['phrase_embs'] = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)
    return df

df = get_data()

# 🔘 Все уникальные тематики
all_topics = sorted({topic for topics in df['topics'] for topic in topics})
selected_topics = st.multiselect("Фильтр по тематикам (независимо от поиска):", all_topics)
filter_search_by_topics = st.checkbox("Искать только в выбранных тематиках", value=False)

# 📂 Фразы по выбранным тематикам
if selected_topics:
    st.markdown("### 📂 Фразы по выбранным тематикам:")
    filtered_df = df[df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))]
    for row in filtered_df.itertuples():
        with st.container():
            st.markdown(
                f"""<div class="christmas-card">
                    <div style="font-size: 18px; font-weight: 600; color: #1a472a;">🎁 {row.phrase_full}</div>
                    <div style="margin-top: 4px; font-size: 14px; color: #2e7d32;">🔖 Тематики: <strong>{', '.join(row.topics)}</strong></div>
                </div>""",
                unsafe_allow_html=True
            )
            if row.comment and str(row.comment).strip().lower() != "nan":
                with st.expander("💬 Комментарий", expanded=False):
                    st.markdown(row.comment)

# 📥 Поисковый запрос
query = st.text_input("Введите ваш запрос:")

if query:
    try:
        # Если включен фильтр, сужаем датафрейм для поиска
        search_df = df
        if filter_search_by_topics and selected_topics:
            mask = df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))
            search_df = df[mask]

            # Подрезаем/назначаем эмбеддинги для search_df, чтобы они соответствовали строкам
            # Берём полный тензор из оригинального df.attrs['phrase_embs'] и индексируем его по индексам search_df
            full_embs = df.attrs.get('phrase_embs', None)
            if full_embs is not None:
                try:
                    indices = search_df.index.tolist()
                    if isinstance(full_embs, torch.Tensor):
                        if indices:
                            # индексируем тензор по оригинальным индексам (они совпадают с порядком построения)
                            search_df.attrs['phrase_embs'] = full_embs[indices]
                        else:
                            # пустой набор — создаём пустой тензор нужной ширины
                            search_df.attrs['phrase_embs'] = full_embs.new_empty((0, full_embs.size(1)))
                    else:
                        # если это numpy array или похожее
                        import numpy as np
                        arr = np.asarray(full_embs)
                        search_df.attrs['phrase_embs'] = arr[indices]
                except Exception:
                    # В крайнем случае — пересчитаем эмбеддинги для search_df (медленнее, но безопасно)
                    model = get_model()
                    if not search_df.empty:
                        search_df.attrs['phrase_embs'] = model.encode(search_df['phrase_proc'].tolist(), convert_to_tensor=True)
                    else:
                        search_df.attrs['phrase_embs'] = None

        # Проверка на пустой результат
        if search_df.empty:
            st.warning("❄️ Нет данных для поиска по выбранным тематикам.")
        else:
            results = semantic_search(query, search_df)
            exact_results = keyword_search(query, search_df)

            # Запись в лог
            log_query(
                query,
                semantic_count=len(results),
                keyword_count=len(exact_results),
                status="найдено" if results or exact_results else "не найдено"
            )

            if results:
                st.markdown("### 🎯 Результаты умного поиска:")
                for score, phrase_full, topics, comment in results:
                    with st.container():
                        card_class = "christmas-card-highlight" if score > 0.8 else "christmas-card"
                        icon = "⭐" if score > 0.8 else "🎁"
                        
                        st.markdown(
                            f"""<div class="{card_class}">
                                <div style="font-size: 18px; font-weight: 600; color: #1a472a;">{icon} {phrase_full}</div>
                                <div style="margin-top: 4px; font-size: 14px; color: #2e7d32;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                                <div style="margin-top: 2px; font-size: 13px; color: #388e3c;">🎯 Релевантность: {score:.2f}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        if comment and str(comment).strip().lower() != "nan":
                            with st.expander("💬 Комментарий", expanded=False):
                                st.markdown(comment)
            else:
                st.warning("🎄 Совпадений не найдено в умном поиске.")

            if exact_results:
                st.markdown("### 🧷 Точный поиск:")
                for phrase, topics, comment in exact_results:
                    with st.container():
                        st.markdown(
                            f"""<div class="christmas-card">
                                <div style="font-size: 18px; font-weight: 600; color: #1b5e20;">🎯 {phrase}</div>
                                <div style="margin-top: 4px; font-size: 14px; color: #2e7d32;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        if comment and str(comment).strip().lower() != "nan":
                            with st.expander("💬 Комментарий", expanded=False):
                                st.markdown(comment)
            else:
                st.info("❄️ Ничего не найдено в точном поиске.")

    except Exception as e:
        st.error(f"🎄 Ошибка при обработке запроса: {e}")

# Блок логов
with st.expander("⚙️ Логи (для админов)", expanded=False):
    if st.button("⬇️ Скачать логи"):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "rb") as f:
                st.download_button("Скачать как CSV", f.read(), file_name="logs.csv", mime="text/csv")
        else:
            st.info("Файл логов отсутствует")

    if st.button("🗑 Очистить логи"):
        if os.path.exists(LOG_FILE):
            open(LOG_FILE, "w").close()
        st.success("Логи очищены!")

# Новогодний футер
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #1a6e1a; margin-top: 30px;">
        <p>🎄 <strong>С Наступающим Новым Годом, Коллеги❤️</strong> 🎄</p>
        <div style="font-size: 0.9rem; color: #666;">
            Пусть ваш код всегда будет чистым, а поиск — точным!
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
