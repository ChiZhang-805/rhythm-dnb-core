"""One contract for the workbook, training heads and prediction API."""

CATEGORIES = {
    'emotion': ('情绪', ('mood_valence','joy_intensity','sadness_intensity','anxiety_intensity','irritability_intensity')),
    'stress': ('压力', ('stress_intensity',)),
    'diet': ('饮食', ('appetite_loss_intensity','excess_intake_intensity','meal_irregularity_intensity')),
    'sleep': ('睡眠', ('sleep_quality','sleep_onset_difficulty','sleep_disruption_intensity','post_sleep_fatigue')),
    'social': ('社交', ('social_willingness','social_satisfaction','loneliness_intensity','social_burden_intensity')),
}
METRICS = (
    ('mood_valence','emotion','总体心情','very_bad_to_very_good'),
    ('joy_intensity','emotion','愉快程度','absent_to_extreme'),
    ('sadness_intensity','emotion','悲伤程度','absent_to_extreme'),
    ('anxiety_intensity','emotion','焦虑程度','absent_to_extreme'),
    ('irritability_intensity','emotion','烦躁程度','absent_to_extreme'),
    ('stress_intensity','stress','压力程度','absent_to_extreme'),
    ('appetite_loss_intensity','diet','食欲下降程度','absent_to_extreme'),
    ('excess_intake_intensity','diet','进食过量程度','absent_to_extreme'),
    ('meal_irregularity_intensity','diet','进餐不规律程度','absent_to_extreme'),
    ('sleep_quality','sleep','主观睡眠质量','very_bad_to_very_good'),
    ('sleep_onset_difficulty','sleep','入睡困难程度','absent_to_extreme'),
    ('sleep_disruption_intensity','sleep','睡眠中断程度','absent_to_extreme'),
    ('post_sleep_fatigue','sleep','醒后疲劳程度','absent_to_extreme'),
    ('social_willingness','social','社交意愿','none_to_strong'),
    ('social_satisfaction','social','互动满意度','very_bad_to_very_good'),
    ('loneliness_intensity','social','孤独程度','absent_to_extreme'),
    ('social_burden_intensity','social','社交负担程度','absent_to_extreme'),
)
CONTRACT_ID='chinese_category_regression_17'
FORMAL_CATEGORIES=('emotion','social','diet','sleep')


def normalize_category(value):
    # PSEUDOCODE: map Chinese or canonical category names -> reject unsupported categories.
    if not isinstance(value,str):
        raise ValueError('请选择一个文本类别。')
    aliases={name:key for key,(name,_) in CATEGORIES.items()}
    value=aliases.get(value.strip(),value.strip())
    if value not in CATEGORIES:
        raise ValueError('未知文本类别。')
    return value


def validate_input(category,text):
    # PSEUDOCODE: normalize category -> require nonempty Chinese text within the input limit.
    category=normalize_category(category)
    if not isinstance(text,str) or not text.strip():
        raise ValueError('请输入非空文本。')
    text=text.strip()
    if len(text)>10000:
        raise ValueError('文本超过10000字符，请缩短。')
    if not any('\u3400'<=c<='\u9fff' for c in text):
        raise ValueError('当前模型仅针对中文文本，请输入中文。')
    return category,text


def schema():
    # PSEUDOCODE: export each category with its score names, directions and numeric bounds.
    return {'contract_id':CONTRACT_ID,'categories':[
        {'id':key,'label':name,'auxiliary':key not in FORMAL_CATEGORIES,
         'outputs':[{'key':metric,'label':next(m[2] for m in METRICS if m[0]==metric),
                     'direction':next(m[3] for m in METRICS if m[0]==metric),
                     'minimum':0,'maximum':100} for metric in keys]}
        for key,(name,keys) in CATEGORIES.items()]}
