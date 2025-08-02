import dash
from dash import dcc, html, Input, Output, State, callback, ctx, dash_table
import plotly.graph_objects as go
import swisseph as swe
from datetime import datetime, timezone, timedelta, date, time
import os
import numpy as np
import pandas as pd

# --- 定数定義 ---
SIGN_NAMES = ["牡羊座", "牡牛座", "双子座", "蟹座", "獅子座", "乙女座", "天秤座", "蠍座", "射手座", "山羊座", "水瓶座", "魚座"]
SIGN_SYMBOLS = ["♈", "♉", "♊", "♋", "♌", "♍", "♎", "♏", "♐", "♑", "♒", "♓"]
DEGREES_PER_SIGN = 30

PLANET_NAMES = {
    "太陽": swe.SUN, "月": swe.MOON, "水星": swe.MERCURY, "金星": swe.VENUS, "火星": swe.MARS,
    "木星": swe.JUPITER, "土星": swe.SATURN, "天王星": swe.URANUS, "海王星": swe.NEPTUNE,
    "冥王星": swe.PLUTO, "キロン": swe.CHIRON, "リリス": swe.MEAN_APOG, "ドラゴンヘッド": swe.MEAN_NODE
}
PLANET_SYMBOLS = {
    "太陽": "☉", "月": "☽", "水星": "☿", "金星": "♀", "火星": "♂", "木星": "♃", "土星": "♄",
    "天王星": "♅", "海王星": "♆", "冥王星": "♇", "キロン": "⚷", "リリス": "⚸",
    "ドラゴンヘッド": "☊", "ドラゴンテイル": "☋", "ASC": "ASC", "MC": "MC"
}
SENSITIVE_POINTS = ["ASC", "MC"]
PREFECTURE_DATA = {
    "北海道": {"lat": 43.064, "lon": 141.348}, "青森県": {"lat": 40.825, "lon": 140.741},
    "岩手県": {"lat": 39.704, "lon": 141.153}, "宮城県": {"lat": 38.269, "lon": 140.872},
    "秋田県": {"lat": 39.719, "lon": 140.102}, "山形県": {"lat": 38.240, "lon": 140.364},
    "福島県": {"lat": 37.750, "lon": 140.468}, "茨城県": {"lat": 36.342, "lon": 140.447},
    "栃木県": {"lat": 36.566, "lon": 139.884}, "群馬県": {"lat": 36.391, "lon": 139.060},
    "埼玉県": {"lat": 35.857, "lon": 139.649}, "千葉県": {"lat": 35.605, "lon": 140.123},
    "東京都": {"lat": 35.690, "lon": 139.692}, "神奈川県": {"lat": 35.448, "lon": 139.643},
    "新潟県": {"lat": 37.902, "lon": 139.023}, "富山県": {"lat": 36.695, "lon": 137.211},
    "石川県": {"lat": 36.594, "lon": 136.626}, "福井県": {"lat": 36.065, "lon": 136.222},
    "山梨県": {"lat": 35.664, "lon": 138.568}, "長野県": {"lat": 36.651, "lon": 138.181},
    "岐阜県": {"lat": 35.391, "lon": 136.722}, "静岡県": {"lat": 34.977, "lon": 138.383},
    "愛知県": {"lat": 35.180, "lon": 136.907}, "三重県": {"lat": 34.730, "lon": 136.509},
    "滋賀県": {"lat": 35.005, "lon": 135.869}, "京都府": {"lat": 35.021, "lon": 135.756},
    "大阪府": {"lat": 34.686, "lon": 135.520}, "兵庫県": {"lat": 34.691, "lon": 135.183},
    "奈良県": {"lat": 34.685, "lon": 135.833}, "和歌山県": {"lat": 34.226, "lon": 135.168},
    "鳥取県": {"lat": 35.504, "lon": 134.238}, "島根県": {"lat": 35.472, "lon": 133.051},
    "岡山県": {"lat": 34.662, "lon": 133.934}, "広島県": {"lat": 34.396, "lon": 132.459},
    "山口県": {"lat": 34.186, "lon": 131.471}, "徳島県": {"lat": 34.066, "lon": 134.559},
    "香川県": {"lat": 34.340, "lon": 134.043}, "愛媛県": {"lat": 33.842, "lon": 132.765},
    "高知県": {"lat": 33.560, "lon": 133.531}, "福岡県": {"lat": 33.607, "lon": 130.418},
    "佐賀県": {"lat": 33.249, "lon": 130.299}, "長崎県": {"lat": 32.745, "lon": 129.874},
    "熊本県": {"lat": 32.790, "lon": 130.742}, "大分県": {"lat": 33.238, "lon": 131.613},
    "宮崎県": {"lat": 31.911, "lon": 131.424}, "鹿児島県": {"lat": 31.560, "lon": 130.558},
    "沖縄県": {"lat": 26.212, "lon": 127.681}
}

# --- ヘルパー関数 ---
def get_degree_parts(d):
    d %= 360
    sign_index = int(d / DEGREES_PER_SIGN)
    pos_in_sign = d % DEGREES_PER_SIGN
    return SIGN_NAMES[sign_index], f"{int(pos_in_sign):02d}°{int((pos_in_sign - int(pos_in_sign)) * 60):02d}'"

def get_house_number(degree, cusps):
    cusps_with_13th = list(cusps) + [(cusps[0] + 360) % 360]
    for i in range(12):
        start, end = cusps[i], cusps_with_13th[i+1]
        if start > end:
            if degree >= start or degree < end: return i + 1
        else:
            if start <= degree < end: return i + 1
    return 12

# --- 計算関数 ---
def _calculate_celestial_bodies(jd_ut, lat, lon, calc_houses=False):
    celestial_bodies = {}
    iflag = swe.FLG_SWIEPH | swe.FLG_SPEED
    for name, p_id in PLANET_NAMES.items():
        res = swe.calc_ut(jd_ut, p_id, iflag)
        celestial_bodies[name] = {'pos': res[0][0], 'is_retro': res[0][3] < 0}
    head_pos = celestial_bodies["ドラゴンヘッド"]['pos']
    celestial_bodies["ドラゴンテイル"] = {'pos': (head_pos + 180) % 360, 'is_retro': False}
    if calc_houses:
        try:
            cusps, ascmc = swe.houses(jd_ut, lat, lon, b'P')
            celestial_bodies["ASC"] = {'pos': ascmc[0], 'is_retro': False}
            celestial_bodies["MC"] = {'pos': ascmc[1], 'is_retro': False}
            return celestial_bodies, cusps, ascmc
        except swe.Error:
            return celestial_bodies, None, None
    return celestial_bodies, None, None

def perform_calculations(birth_dt_utc, transit_dt_utc, lat, lon):
    project_path = '/home/shigemiyagi/horoscope_plotly'
    ephe_path = os.path.join(project_path, 'ephe')
    swe.set_ephe_path(ephe_path)
    
    jd_ut_natal, _ = swe.utc_to_jd(birth_dt_utc.year, birth_dt_utc.month, birth_dt_utc.day, birth_dt_utc.hour, birth_dt_utc.minute, birth_dt_utc.second, 1)
    natal_bodies, cusps, ascmc = _calculate_celestial_bodies(jd_ut_natal, lat, lon, calc_houses=True)
    if not cusps: return None

    age_in_years = (datetime.now(timezone.utc) - birth_dt_utc).days / 365.2425
    prog_dt_utc = birth_dt_utc + timedelta(days=age_in_years)
    jd_ut_prog, _ = swe.utc_to_jd(prog_dt_utc.year, prog_dt_utc.month, prog_dt_utc.day, prog_dt_utc.hour, prog_dt_utc.minute, prog_dt_utc.second, 1)
    progressed_bodies, _, _ = _calculate_celestial_bodies(jd_ut_prog, lat, lon)
    
    jd_ut_transit, _ = swe.utc_to_jd(transit_dt_utc.year, transit_dt_utc.month, transit_dt_utc.day, transit_dt_utc.hour, transit_dt_utc.minute, transit_dt_utc.second, 1)
    transit_bodies, _, _ = _calculate_celestial_bodies(jd_ut_transit, lat, lon)
    return natal_bodies, progressed_bodies, transit_bodies, cusps, ascmc

# --- 描画関数 (Plotly版) ---
def create_tri_chart_plotly(natal, prog, trans, cusps, ascmc):
    fig = go.Figure()
    rotation_offset = 180 - ascmc[0]
    def apply_rotation(pos): return (pos - rotation_offset + 360) % 360

    for i in range(12):
        fig.add_trace(go.Barpolar(
            r=[1], base=9, width=30, theta=[i * 30 + 15 - rotation_offset],
            marker_color="aliceblue" if i % 2 == 0 else "white",
            marker_line_color="lightgray", marker_line_width=1,
            text=SIGN_SYMBOLS[i], textfont_size=20, hoverinfo='none'
        ))

    for i, cusp_deg in enumerate(cusps):
        fig.add_trace(go.Scatterpolar(
            r=[0, 9], theta=[cusp_deg - rotation_offset, cusp_deg - rotation_offset],
            mode='lines', line_color='gray', line_dash='dash', hoverinfo='none'
        ))
        next_cusp_deg = cusps[(i + 1) % 12]
        mid_angle_deg = cusp_deg + (((next_cusp_deg - cusp_deg) + 360) % 360) / 2
        fig.add_trace(go.Scatterpolar(
            r=[3.5], theta=[mid_angle_deg - rotation_offset],
            mode='text', text=str(i + 1), textfont_color='gray', hoverinfo='none'
        ))

    radii = {'natal': 4.4, 'prog': 6.2, 'trans': 8.0}
    for chart_type, bodies in [('natal', natal), ('prog', prog), ('trans', trans)]:
        for name, data in bodies.items():
            if name in SENSITIVE_POINTS: continue
            angle = apply_rotation(data['pos'])
            retro_str = ' R' if data.get('is_retro') else ''
            fig.add_trace(go.Scatterpolar(
                r=[radii[chart_type]], theta=[angle], mode='text',
                text=PLANET_SYMBOLS[name], textfont_size=16,
                hovertext=f"{name}: {get_degree_parts(data['pos'])[1]}{retro_str}", hoverinfo='text'
            ))

    fig.add_trace(go.Scatterpolar(r=[9.2], theta=[apply_rotation(ascmc[0])], mode='text', text="ASC", hoverinfo='none'))
    fig.add_trace(go.Scatterpolar(r=[9.2], theta=[apply_rotation(ascmc[1])], mode='text', text="MC", hoverinfo='none'))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 10]),
            angularaxis=dict(visible=False, direction="clockwise", rotation=rotation_offset)
        ),
        showlegend=False, margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# --- Dash App ---
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("🪐 三重円ホロスコープ作成 (Dash版)"),
    html.Div([
        html.Div([
            html.H3("出生情報"),
            html.Label("生年月日"),
            dcc.DatePickerSingle(id='birth-date', date=date(1990, 1, 1), display_format='YYYY-MM-DD'),
            html.Label("出生時刻 (HH:MM)"),
            dcc.Input(id='birth-time', type='text', value='12:00'),
            html.Label("出生地"),
            dcc.Dropdown(id='prefecture', options=[{'label': k, 'value': k} for k in PREFECTURE_DATA.keys()], value='東京都'),
            html.Hr(),
            html.H3("トランジット指定"),
            dcc.DatePickerSingle(id='transit-date', date=date.today(), display_format='YYYY-MM-DD'),
            dcc.Input(id='transit-time', type='text', value=datetime.now().strftime('%H:%M')),
            html.Button('⏪ 1日戻す', id='prev-day-button', n_clicks=0),
            html.Button('1日進む ⏩', id='next-day-button', n_clicks=0),
            html.Hr(),
            html.Button('ホロスコープを作成', id='submit-button', n_clicks=0, style={'fontWeight': 'bold', 'width': '100%'})
        ], style={'width': '30%', 'float': 'left', 'padding': '10px'}),
        
        html.Div([
            dcc.Loading(id="loading", children=[
                dcc.Graph(id='horoscope-chart'),
                html.Div(id='data-tables')
            ], type="circle"),
        ], style={'width': '70%', 'float': 'right'})
    ])
])

@callback(
    Output('horoscope-chart', 'figure'),
    Output('data-tables', 'children'),
    Output('transit-date', 'date'),
    Input('submit-button', 'n_clicks'),
    Input('prev-day-button', 'n_clicks'),
    Input('next-day-button', 'n_clicks'),
    State('birth-date', 'date'),
    State('birth-time', 'value'),
    State('prefecture', 'value'),
    State('transit-date', 'date'),
    State('transit-time', 'value'),
)
def update_chart(submit_clicks, prev_clicks, next_clicks, birth_date_str, birth_time_str, prefecture, transit_date_str, transit_time_str):
    # ▼▼▼▼▼ ここを修正 ▼▼▼▼▼
    # ページの初回ロード時（どのボタンも押されていない時）だけ初期メッセージを表示
    if not ctx.triggered_id:
    # ▲▲▲▲▲ ここを修正 ▲▲▲▲▲
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text="情報を入力して「ホロスコープを作成」を押してください",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16)
                )
            ]
        )
        return fig, None, dash.no_update

    triggered_id = ctx.triggered_id
    
    try:
        birth_dt = datetime.fromisoformat(birth_date_str)
        birth_time_obj = time.fromisoformat(f"{birth_time_str}:00")
    except ValueError:
        return go.Figure(), "出生時刻の形式が正しくありません (HH:MM)", dash.no_update

    birth_dt_local = datetime.combine(birth_dt, birth_time_obj)
    birth_dt_utc = birth_dt_local.replace(tzinfo=timezone(timedelta(hours=9))).astimezone(timezone.utc)
    
    transit_dt = datetime.fromisoformat(transit_date_str)
    if triggered_id == 'prev-day-button':
        transit_dt -= timedelta(days=1)
    elif triggered_id == 'next-day-button':
        transit_dt += timedelta(days=1)
    
    try:
        transit_time_obj = time.fromisoformat(f"{transit_time_str}:00")
    except ValueError:
        return go.Figure(), "トランジット時刻の形式が正しくありません (HH:MM)", transit_dt.date()

    transit_dt_local = datetime.combine(transit_dt, transit_time_obj)
    transit_dt_utc = transit_dt_local.replace(tzinfo=timezone(timedelta(hours=9))).astimezone(timezone.utc)
    
    lat = PREFECTURE_DATA[prefecture]["lat"]
    lon = PREFECTURE_DATA[prefecture]["lon"]
    
    calc_data = perform_calculations(birth_dt_utc, transit_dt_utc, lat, lon)
    if not calc_data:
        return go.Figure(), "計算に失敗しました。入力値を確認してください。", transit_dt.date()
        
    natal, prog, trans, cusps, ascmc = calc_data
    fig = create_tri_chart_plotly(natal, prog, trans, cusps, ascmc)
    
    tables = []
    for chart_type, bodies in [('ネイタル', natal), ('プログレス', prog), ('トランジット', trans)]:
        planet_data = []
        for name, data in bodies.items():
            sign, deg_str = get_degree_parts(data['pos'])
            retro = "R" if data.get('is_retro') else ""
            house = get_house_number(data['pos'], cusps) if name not in SENSITIVE_POINTS else "-"
            name_str = f"{PLANET_SYMBOLS.get(name, '')} {name}"
            planet_data.append({'天体': name_str, 'サイン': sign, '度数': deg_str, '逆行': retro, 'ハウス': house})
        
        df = pd.DataFrame(planet_data)
        tables.append(html.Div([
            html.H4(chart_type),
            dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], style_table={'overflowX': 'auto'})
        ]))

    return fig, html.Div(tables), transit_dt.date()

if __name__ == '__main__':
    app.run_server(debug=True)
