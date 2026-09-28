import json

import pandas as pd

import config

class SLAAnalyzer:
    def __init__(self, file_path=None):
        """Инициализация и базовая очистка данных."""
        print("Инициализация SLA-Аналитика. Загрузка данных...")
        self.df = pd.read_excel(file_path or config.SLA_XLSX_PATH, sheet_name='Sheet0')
        self._prepare_data()

    def _parse_duration(self, duration_str):
        """
        Переводит формат 'ЧЧ:ММ:СС' (даже если часов > 24) в дробные часы.
        """
        if pd.isna(duration_str) or not isinstance(duration_str, str): 
            return 0.0
        try:
            parts = duration_str.split(':')
            if len(parts) == 3:
                return int(parts[0]) + int(parts[1])/60 + int(parts[2])/3600
            return 0.0
        except:
            return 0.0

    def _count_lines_involved(self, row):
        """Подсчитывает, сколько линий реально работали над обращением."""
        cols = [
            'Суммарное время работы 1 линии', 
            'Суммарное время работы 2 линии', 
            'Суммарное время работы 3 линии'
            # 4-ю линию убрали из расчетов
        ]
        count = 0
        for col in cols:
            val = str(row.get(col, '00:00:00'))
            if val not in ['00:00:00', 'nan', '0']:
                count += 1
        return count

    def _prepare_data(self):
        """Предобработка: создание признаков для аналитики."""
        
        # --- ФИЛЬТР БАГА ДАТАСЕТА ---
        # Удаляем единственный тикет 4-й линии, так как организаторы признали это ошибкой
        self.df = self.df[self.df['Кем решен (группа)'] != '(4 линия)']

        # 1. Определение просрочки (по готовому историческому флагу)
        self.df['is_overdue'] = self.df['Просрочен?*'] == 'Просрочен'
        
        # 2. Длительность обработки (Аппроксимация: переводим строку в часы)
        self.df['duration_hours'] = self.df['Фактическая длительность выполнения запроса (SLA)'].apply(self._parse_duration)
        
        # 3. Маркер "Много линий" (Пинг-понг заявкой)
        self.df['lines_count'] = self.df.apply(self._count_lines_involved, axis=1)
        self.df['multi_line'] = self.df['lines_count'] > 1
        
        # 4. Маркер "Много уточнений" (в среднем по датасету 1 уточнение. > 2 считаем "многим")
        self.df['many_clarifications'] = self.df['Количество уточнений'] >= 3

        # 5. Переводим дату регистрации в формат datetime для извлечения дней недели
        self.df['Дата регистрации'] = pd.to_datetime(self.df['Дата регистрации'], errors='coerce')
        # Вытаскиваем день недели (0=Понедельник, 6=Воскресенье)
        self.df['reg_weekday'] = self.df['Дата регистрации'].dt.dayofweek

    def get_dashboard_data(self, filters=None):
        """
        Генерирует полную статистику для Дашборда (с учетом фильтров, если они есть).
        Возвращает JSON.
        """
        data = self.df.copy()
        
        # Применение фильтров
        if filters:
            if 'service' in filters:
                data = data[data['Услуга'] == filters['service']]
            if 'priority' in filters:
                data = data[data['Приоритет'] == filters['priority']]

        total_requests = len(data)
        if total_requests == 0:
            return json.dumps({"error": "Нет данных по заданным фильтрам"})

        # --- БАЗОВАЯ СТАТИСТИКА ---
        overdue_count = data['is_overdue'].sum()
        overdue_rate = round((overdue_count / total_requests) * 100, 2)
        
        multi_line_count = data['multi_line'].sum()
        many_clarifications_count = data['many_clarifications'].sum()

        # --- Метрики времени (Медиана и 90-й перцентиль) ---
        median_time = round(data['duration_hours'].median(), 1)
        p90_time = round(data['duration_hours'].quantile(0.90), 1)

        # --- НАГРУЗКА И РИСКИ (ТОП-5 КАТЕГОРИЙ) ---
        cat_stats = data.groupby('Вид запроса').agg(
            total_tickets=('Номер запроса', 'count'),
            overdue_tickets=('is_overdue', 'sum')
        ).reset_index()
        cat_stats['risk_percent'] = round((cat_stats['overdue_tickets'] / cat_stats['total_tickets']) * 100, 1)
        top_load_categories = cat_stats.sort_values(by='total_tickets', ascending=False).head(5)
        high_risk_categories = cat_stats[cat_stats['total_tickets'] > 5].sort_values(by='risk_percent', ascending=False).head(5)

        # Риск нарушения SLA по ЛИНИЯМ поддержки (Закрываем требование ТЗ)
        line_stats = data.groupby('Кем решен (группа)').agg(
            total_tickets=('Номер запроса', 'count'),
            overdue_tickets=('is_overdue', 'sum')
        ).reset_index()
        line_stats['risk_percent'] = round((line_stats['overdue_tickets'] / line_stats['total_tickets']) * 100, 1)
        high_risk_lines = line_stats[line_stats['total_tickets'] > 5].sort_values(by='risk_percent', ascending=False)

        # "Индекс Пинг-понга" - какие категории чаще всего кидают между линиями
        ping_pong_stats = data[data['multi_line']].groupby('Вид запроса').size().reset_index(name='bounced_tickets')
        top_ping_pong = ping_pong_stats.sort_values(by='bounced_tickets', ascending=False).head(3)

        # --- Нагрузка по дням недели (Сезонность) ---
        weekday_map = {0: 'Пн', 1: 'Вт', 2: 'Ср', 3: 'Чт', 4: 'Пт', 5: 'Сб', 6: 'Вс'}
        load_by_day = data['reg_weekday'].map(weekday_map).value_counts().to_dict()

        # --- СРЕДНЕЕ ВРЕМЯ ОБРАБОТКИ ---
        avg_time_by_line = data.groupby('Кем решен (группа)')['duration_hours'].mean().round(1).to_dict()
        avg_time_by_priority = data.groupby('Приоритет')['duration_hours'].mean().round(1).to_dict()

        # Пакуем всё в красивый словарь
        dashboard = {
            "summary": {
                "total_requests": int(total_requests),
                "overdue_rate_percent": float(overdue_rate),
                "multi_line_tickets": int(multi_line_count),
                "high_clarification_tickets": int(many_clarifications_count)
            },
            "processing_time_hours": {
                "median_50_percent": float(median_time),
                "percentile_90": float(p90_time)
            },
            "top_load_categories": top_load_categories[['Вид запроса', 'total_tickets']].to_dict(orient='records'),
            "high_risk_sla_categories": high_risk_categories[['Вид запроса', 'risk_percent']].to_dict(orient='records'),
            "high_risk_sla_lines": high_risk_lines[['Кем решен (группа)', 'risk_percent']].to_dict(orient='records'),
            "top_ping_pong_categories": top_ping_pong.to_dict(orient='records'),
            "average_processing_hours": {
                "by_line": avg_time_by_line,
                "by_priority": avg_time_by_priority
            },
            "load_by_weekday": load_by_day
        }

        return json.dumps(dashboard, ensure_ascii=False, indent=2)

# Блок тестирования
if __name__ == "__main__":
    analyzer = SLAAnalyzer()
    
    print("\n--- ДАННЫЕ ДЛЯ ДАШБОРДА (БЕЗ ФИЛЬТРОВ) ---")
    dashboard_json = analyzer.get_dashboard_data()
    print(dashboard_json)
