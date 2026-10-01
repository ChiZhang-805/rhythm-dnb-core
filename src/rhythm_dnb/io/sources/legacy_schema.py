"""The 38 collection inputs and 49 nullable quantitative candidate features."""

from copy import deepcopy
from datetime import datetime
from decimal import Decimal
import math
from numbers import Real
import re

from ...text.schema import FORMAL_CATEGORIES,METRICS


SCHEMA_ID='rhythm_quantitative_49'
METADATA_KEYS=('record_id','participant_id','observed_at','source_dataset','split',
               'exercise_type','text_model_id')
REQUIRED_METADATA_KEYS=METADATA_KEYS[:5]
TRUTH_KEYS=('rhythm_state_gt','label_observed_at','event_onset_at','followup_end_at','label_source')
SPLITS=('reference','train','validation','test')
CLOCK_SOURCES={
    'sleep_start_time':'sleep_start_hour',
    'sleep_end_time':'sleep_end_hour',
    'caffeine_last_time':'caffeine_last_hour',
    'exercise_time_of_day':'exercise_hour',
}
CLOCK_KEYS=tuple(CLOCK_SOURCES.values())
TEXT_SOURCES={category+'_description_text':category for category in FORMAL_CATEGORIES}

# Labels, units and IDs follow the original collection_entry_qa/field_map.json.
_FIELDS=(
    ('sleep_start_time','入睡时间','日期+时间','datetime'),
    ('sleep_end_time','起床时间','日期+时间','datetime'),
    ('sleep_duration_h','总睡眠时长','小时','number'),
    ('time_in_bed_h','卧床时长','小时','derived'),
    ('nap_count','白天小睡次数','次','count'),
    ('nap_duration_min','小睡总时长','分钟','number'),
    ('late_night_eating_flag','夜宵','是 / 否','binary'),
    ('breakfast_kcal','早餐热量','千卡','derived'),
    ('lunch_kcal','午餐热量','千卡','derived'),
    ('dinner_kcal','晚餐热量','千卡','derived'),
    ('water_ml','饮水量','毫升','number'),
    ('caffeine_mg','咖啡因','毫克','derived'),
    ('caffeine_last_time','末次咖啡因时间','日期+时间','datetime'),
    ('meal_interval_cv','餐间隔变异系数','无单位','derived'),
    ('exercise_minutes','运动总时长','分钟','number'),
    ('exercise_time_of_day','运动时段','日期+时间','derived'),
    ('exercise_type','运动类型','文字','text'),
    ('heart_rate_bpm','心率','次/分钟','number'),
    ('resting_hr_bpm','静息心率','次/分钟','number'),
    ('hr_max_bpm','最高心率','次/分钟','number'),
    ('hr_sd_bpm','心率标准差','次/分钟','derived'),
    ('skin_temp_c','皮肤温度','摄氏度','number'),
    ('spo2_pct','血氧饱和度','%','number'),
    ('spo2_min_pct','夜间最低血氧','%','number'),
    ('stress_score','主观压力','0～10分','score10'),
    ('general_mood','总体心情','0～10分','score10'),
    ('energy_score','精力水平','0～10分','score10'),
    ('self_rated_state','状态自评','1～5分','score5'),
    ('fatigue_score','疲劳评分','0～10分','score10'),
    ('social_interaction_min','社交时长','分钟','number'),
    ('social_contact_count','社交人数','人','count'),
    ('ambient_temp_c','环境温度','摄氏度','number'),
    ('noise_db','环境噪声','分贝','number'),
    ('screen_time_min','屏幕时长','分钟','number'),
    ('emotion_description_text','情绪描述','现在','text'),
    ('social_description_text','社交描述','昨天','text'),
    ('diet_description_text','饮食内容描述','昨天','text'),
    ('sleep_description_text','睡眠主观描述','昨晚','text'),
)
ORIGINAL_FIELDS=[{'id':i,'key':key,'label':label,'unit':unit,'type':kind}
                 for i,(key,label,unit,kind) in enumerate(_FIELDS,1)]

# Broad input guards, not clinical thresholds or normal-reference intervals.
_BOUNDS={
    'sleep_duration_h':(0,24),'time_in_bed_h':(0,24),
    'nap_duration_min':(0,1440),'exercise_minutes':(0,1440),
    'social_interaction_min':(0,1440),'screen_time_min':(0,1440),
    'heart_rate_bpm':(0,400),'resting_hr_bpm':(0,400),
    'hr_max_bpm':(0,400),'hr_sd_bpm':(0,400),
    'skin_temp_c':(0,60),'ambient_temp_c':(-100,100),'noise_db':(0,200),
    'spo2_pct':(0,100),'spo2_min_pct':(0,100),
}


def _feature_spec(field):
    # PSEUDOCODE: translate collection field metadata into nullable quantitative units and bounds.
    source=field['key']
    key=CLOCK_SOURCES.get(source,source)
    kind={'count':'count','binary':'binary','score10':'ordinal','score5':'ordinal'}.get(field['type'],'continuous')
    minimum,maximum=_BOUNDS.get(key,(0,None))
    unit=field['unit']
    if source in CLOCK_SOURCES:
        kind,unit,minimum,maximum='clock','小时（当地时钟）',0,24
    elif field['type'] in ('score10','score5','binary'):
        minimum,maximum={'score10':(0,10),'score5':(1,5),'binary':(0,1)}[field['type']]
        if kind=='binary':
            unit='0/1'
    return {'key':key,'label':field['label'],'unit':unit,'source_id':field['id'],
            'source_key':source,'kind':kind,'minimum':minimum,'maximum':maximum,
            'maximum_exclusive':kind=='clock','nullable':True}


FEATURE_SPECS=[_feature_spec(field) for field in ORIGINAL_FIELDS if field['type']!='text']
for _metric,_category,_label,_direction in METRICS:
    if _category in FORMAL_CATEGORIES:
        _source=_category+'_description_text'
        FEATURE_SPECS.append({'key':'text_'+_metric,'label':_label,'unit':'0～100分',
            'source_id':next(f['id'] for f in ORIGINAL_FIELDS if f['key']==_source),
            'source_key':_source,'kind':'text_score','category':_category,'direction':_direction,
            'minimum':0,'maximum':100,'maximum_exclusive':False,'nullable':True})
FEATURE_KEYS=tuple(spec['key'] for spec in FEATURE_SPECS)
TEXT_FEATURE_KEYS=tuple(spec['key'] for spec in FEATURE_SPECS if spec['kind']=='text_score')
RECORD_KEYS=METADATA_KEYS+TRUTH_KEYS+FEATURE_KEYS
_SPEC_BY_KEY={spec['key']:spec for spec in FEATURE_SPECS}
_MISSING=frozenset(('', 'na','n/a','null','none','-','--','\u2014','\u2013',
                    '未测','未填','不适用','未知'))
_ISO_DATETIME=re.compile(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)?$')


def _missing(value):
    # PSEUDOCODE: recognize null and documented workbook missing markers without inventing zero.
    return value is None or isinstance(value,str) and value.strip().lower() in _MISSING


def _number(value,key):
    # PSEUDOCODE: reject booleans and nonfinite inputs -> parse supported numeric representations.
    if isinstance(value,bool) or not isinstance(value,(Real,Decimal,str)):
        raise ValueError(f'{key}: expected a finite number')
    try:
        number=float(value)
    except (ValueError,TypeError,OverflowError) as error:
        raise ValueError(f'{key}: expected a finite number') from error
    if not math.isfinite(number):
        raise ValueError(f'{key}: expected a finite number')
    return number


def _feature_value(value,key):
    # PSEUDOCODE: retain missingness -> enforce units/bounds -> require exact integers for discrete fields.
    if _missing(value):
        return None
    spec=_SPEC_BY_KEY[key]
    number=_number(value,key)
    if spec['minimum'] is not None and number<spec['minimum']:
        raise ValueError(f"{key}: must be >= {spec['minimum']}")
    maximum=spec['maximum']
    if maximum is not None and (number>maximum or spec['maximum_exclusive'] and number==maximum):
        operator='<' if spec['maximum_exclusive'] else '<='
        raise ValueError(f'{key}: must be {operator} {maximum}')
    if spec['kind'] in ('count','binary','ordinal'):
        exact=Decimal(str(value).strip())
        if exact!=exact.to_integral_value():
            raise ValueError(f'{key}: expected an integer')
        return int(exact)
    return number


def _timestamp(value,key,require_timezone=True):
    # PSEUDOCODE: parse a full ISO date-time -> require an offset when used as a real timestamp.
    if isinstance(value,datetime):
        result=value
    elif isinstance(value,str) and _ISO_DATETIME.fullmatch(value.strip()):
        try:
            result=datetime.fromisoformat(value.strip().replace('Z','+00:00'))
        except ValueError as error:
            raise ValueError(f'{key}: invalid ISO datetime') from error
    else:
        raise ValueError(f'{key}: expected an ISO datetime with date and time')
    if require_timezone and result.utcoffset() is None:
        raise ValueError(f'{key}: ISO datetime must include a timezone offset')
    return result


def validate_record(record:dict,require_identity=True)->dict:
    """Return a new canonical row; reject unknown keys and invalid supplied values.

    Missing features and optional metadata become None. Identity requires the
    first five METADATA_KEYS, including source_dataset and split. Numeric strings
    and the collection's missing markers are accepted for workbook ingestion.
    Clock hours are local values in [0,24). Truth dates never become features.
    """
    # PSEUDOCODE: reject unknown fields -> normalize identities and values -> retain copied source evidence.
    if not isinstance(record,dict):
        raise ValueError('record: expected a dict')
    unknown=set(record)-set(RECORD_KEYS)-{'warnings','provenance'}
    if unknown:
        raise ValueError('record: unknown fields: '+', '.join(sorted(map(str,unknown))))
    result={key:None for key in RECORD_KEYS}
    for key in METADATA_KEYS+TRUTH_KEYS:
        value=record.get(key)
        if _missing(value):
            continue
        if key in ('observed_at','label_observed_at','event_onset_at','followup_end_at'):
            value=_timestamp(value,key).isoformat()
        elif key=='rhythm_state_gt':
            number=_number(value,key)
            if number not in (0,1) or Decimal(str(value).strip()) not in (0,1):
                raise ValueError(f'{key}: expected 0, 1 or None')
            value=int(number)
        elif not isinstance(value,str) or not value.strip():
            raise ValueError(f'{key}: expected a nonempty string or None')
        else:
            value=value.strip()
        result[key]=value
    if require_identity:
        for key in REQUIRED_METADATA_KEYS:
            if result[key] is None:
                raise ValueError(f'{key}: required identity field')
    if result['split'] is not None and result['split'] not in SPLITS:
        raise ValueError('split: expected reference, train, validation or test')
    for key in FEATURE_KEYS:
        result[key]=_feature_value(record.get(key),key)
    if any(result[key] is not None for key in TEXT_FEATURE_KEYS) and result['text_model_id'] is None:
        raise ValueError('text_model_id: required when text scores are present; use a fixed trained run ID')
    warnings=record.get('warnings',[])
    provenance=record.get('provenance',{})
    if not isinstance(warnings,list) or any(not isinstance(item,str) for item in warnings):
        raise ValueError('warnings: expected a list of strings')
    if not isinstance(provenance,dict):
        raise ValueError('provenance: expected a dict')
    result['warnings']=list(warnings)
    result['provenance']=deepcopy(provenance)
    return result
