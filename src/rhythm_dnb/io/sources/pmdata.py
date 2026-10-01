"""Legacy raw-source parser; explicit path, retrospective semantics retained.

Returned legacy records are NOT automatically prospective Observation objects.
Use the hospital/normalized adapter after supplying verified time and lineage.
"""
"""Import measured PMData person-days without assigning rhythm outcomes."""

from datetime import datetime,timedelta
from hashlib import sha256
from pathlib import Path
import csv
import json
import math

from .legacy_schema import validate_record
from ...timebase import instant


MIRROR_REVISION='2c97bcd6713edd075e1d88fa8401ca0b881c9464'
MIRROR='https://huggingface.co/datasets/aai530-group6/pmdata/resolve/'+MIRROR_REVISION
ORIGINAL='https://datasets.simula.no/pmdata/'
FILES=('fitbit/sleep.json','fitbit/sleep_score.csv','fitbit/exercise.json','pmsys/wellness.csv')
P01_OFFICIAL_SLEEP_SCORE_SHA256='01fc09f685cc5844eea5b64cc47b4a5b486f41eeba3b96b8302e480cbe58690f'


def _digest(path):
    # PSEUDOCODE: hash exact local source bytes for the import provenance receipt.
    return sha256(Path(path).read_bytes()).hexdigest()






def _local_datetime(value):
    # PSEUDOCODE: parse the source local date-time while preserving its supplied offset.
    if not isinstance(value,str):
        raise ValueError('Missing PMData local timestamp.')
    return datetime.fromisoformat(value.replace(' ','T'))


def _clock(value):
    # PSEUDOCODE: extract fractional local hour from the source timestamp.
    point=_local_datetime(value)
    return point.hour+point.minute/60+point.second/3600+point.microsecond/3600000000


def _main_sleep_by_date(entries):
    # PSEUDOCODE: select the longest designated main sleep per source date with a stable ID tie-break.
    selected={}
    for item in entries:
        if item.get('mainSleep') is not True or not item.get('dateOfSleep'):
            continue
        date=item['dateOfSleep']
        candidate=(int(item.get('minutesAsleep') or 0),str(item.get('logId','')))
        previous=selected.get(date)
        if previous is None or candidate>(int(previous.get('minutesAsleep') or 0),str(previous.get('logId',''))):
            selected[date]=item
    return selected


def _prior_day_exercise(entries):
    # PSEUDOCODE: group valid exercises by source date -> sum duration and calculate weighted circular midpoint.
    totals={}
    for item in entries:
        try:
            started=_local_datetime(item['startTime'])
            day=started.date().isoformat()
            duration=float(item['duration'])/60000
        except (KeyError,TypeError,ValueError,OverflowError):
            continue
        if math.isfinite(duration) and 0<duration<=1440:
            totals.setdefault(day,[]).append((started,duration,str(item.get('logId',''))))
    results={}
    for day,events in totals.items():
        minutes=sum(duration for _,duration,_ in events)
        ids=[event_id for _,_,event_id in events]
        x=sum(duration*math.cos(2*math.pi*((start.hour*60+start.minute+
             start.second/60+duration/2)%1440)/1440) for start,duration,_ in events)
        y=sum(duration*math.sin(2*math.pi*((start.hour*60+start.minute+
             start.second/60+duration/2)%1440)/1440) for start,duration,_ in events)
        hour=(math.atan2(y,x)%(2*math.pi))*24/(2*math.pi) if math.hypot(x,y)>1e-9 else None
        results[day]=(minutes,ids,hour)
    return results


def _available_wellness(rows,anchor):
    # PSEUDOCODE: retain reports from the preceding 24 hours -> select the most recent available report.
    eligible=[]
    for row in rows:
        try:
            reported=instant(row['effective_time_frame'])
        except (KeyError,TypeError,ValueError):
            continue
        age=(instant(anchor)-reported).total_seconds()
        if 0<=age<=86400:
            eligible.append((reported,row))
    return max(eligible,key=lambda item:item[0]) if eligible else None


def build_records(root,participants=None):
    """A sleep-score timestamp anchors each row; unmeasured fields remain NULL.

    These are retrospective date-bucket research rows. PMData supplies no
    independent rhythm-disorder diagnosis or future-onset ground truth.
    """
    # PSEUDOCODE: verify local input presence -> join matched sleep scores -> retain prior exercise and source-backed wellness.
    root=Path(root)
    participants=tuple(participants or (f'p{i:02d}' for i in range(1,17)))
    records=[]
    for participant in participants:
        base=root/participant
        files={relative:base/relative for relative in FILES}
        if not all(path.is_file() for path in files.values()):
            raise FileNotFoundError('Selected PMData source files missing for '+participant)
        digests={relative:_digest(path) for relative,path in files.items()}
        sleeps=_main_sleep_by_date(json.loads(files['fitbit/sleep.json'].read_text(encoding='utf-8-sig')))
        exercises=_prior_day_exercise(json.loads(files['fitbit/exercise.json'].read_text(encoding='utf-8-sig')))
        with files['fitbit/sleep_score.csv'].open(encoding='utf-8-sig',newline='') as source:
            scores=list(csv.DictReader(source))
        with files['pmsys/wellness.csv'].open(encoding='utf-8-sig',newline='') as source:
            wellness=list(csv.DictReader(source))
        seen=set()
        for score in scores:
            try:
                anchor=datetime.fromisoformat(score['timestamp'].replace('Z','+00:00'))
                instant(anchor)
                date=anchor.date().isoformat()
            except (KeyError,TypeError,ValueError):
                continue
            if date in seen or date not in sleeps:
                continue
            sleep=sleeps[date]
            if str(sleep.get('logId'))!=str(score.get('sleep_log_entry_id')):
                continue
            seen.add(date)
            raw={'record_id':f'PMData-{participant}-{date}',
                 'participant_id':f'PMData-{participant}',
                 'observed_at':anchor.isoformat(),
                 'source_dataset':'PMData','split':'train'}
            provenance={'source':{'original':ORIGINAL,'mirror':MIRROR,
                                  'sha256':digests,'participant':participant,'date':date},
                        'observed_at':{'source_key':'sleep_score.timestamp',
                                       'meaning':'sleep-score anchor; other date-bucket values may not have been available at this instant'}}
            warnings=['Retrospective date-bucket row; do not use for prospective validation without event-availability audit.']
            for key,source_key,divisor in (('sleep_duration_h','minutesAsleep',60),
                                           ('time_in_bed_h','timeInBed',60)):
                try:
                    value=float(sleep[source_key])/divisor
                    if 0<=value<=24:
                        raw[key]=value
                        provenance[key]={'source_file':'fitbit/sleep.json','source_key':source_key,
                                         'logId':str(sleep['logId']),'conversion':f'/{divisor}'}
                except (KeyError,TypeError,ValueError,OverflowError):
                    pass
            levels=sleep.get('levels',{}).get('data',[])
            onset=next((item.get('dateTime') for item in levels if item.get('level') not in ('wake','restless')),None)
            if onset:
                try:
                    raw['sleep_start_hour']=_clock(onset)
                    provenance['sleep_start_hour']={'source_file':'fitbit/sleep.json',
                       'source_key':'levels.data.first_nonwake.dateTime','logId':str(sleep['logId']),
                       'meaning':'device-estimated sleep onset, local clock; source timezone unspecified'}
                except ValueError:
                    pass
            try:
                raw['sleep_end_hour']=_clock(sleep['endTime'])
                provenance['sleep_end_hour']={'source_file':'fitbit/sleep.json',
                    'source_key':'endTime','logId':str(sleep['logId']),
                    'meaning':'device-estimated end of sleep, proxy for wake/exit time'}
            except (KeyError,TypeError,ValueError):
                pass
            try:
                value=float(score['resting_heart_rate'])
                if 0<value<=400:
                    raw['resting_hr_bpm']=value
                    provenance['resting_hr_bpm']={'source_file':'fitbit/sleep_score.csv',
                        'source_key':'resting_heart_rate','logId':str(sleep['logId'])}
            except (KeyError,TypeError,ValueError,OverflowError):
                pass
            yesterday=(anchor.date()-timedelta(days=1)).isoformat()
            if yesterday in exercises:
                minutes,ids,hour=exercises[yesterday]
                if minutes<=1440:
                    raw['exercise_minutes']=minutes
                    provenance['exercise_minutes']={'source_file':'fitbit/exercise.json',
                        'source_key':'duration','period':'previous local calendar date',
                        'source_date':yesterday,'logIds':ids,'conversion':'milliseconds/60000'}
                    if hour is not None:
                        raw['exercise_hour']=hour
                        provenance['exercise_hour']={'source_file':'fitbit/exercise.json',
                            'source_keys':['startTime','duration'],
                            'period':'previous local calendar date','source_date':yesterday,
                            'logIds':ids,'conversion':'duration-weighted circular midpoint of local event clocks',
                            'timezone':'source timestamps have no offset; local clock only'}
            available=_available_wellness(wellness,anchor)
            if available:
                reported,survey=available
                for key,source_key in (('stress_score','stress'),('general_mood','mood'),
                                       ('fatigue_score','fatigue')):
                    try:
                        original=int(survey[source_key])
                    except (KeyError,TypeError,ValueError):
                        continue
                    if 1<=original<=5:
                        raw[key]=math.floor((original-1)*2.5+0.5)
                        provenance[key]={'source_file':'pmsys/wellness.csv',
                            'source_key':source_key,'source_value':original,
                            'reported_at':reported.isoformat(),
                            'conversion':'round_half_up((source_1_to_5 - 1) * 2.5)',
                            'scale_source':'Thambawita et al. 2020 PMData paper'}
            raw['warnings']=warnings
            raw['provenance']=provenance
            records.append(validate_record(raw))
    return sorted(records,key=lambda row:(row['participant_id'],row['observed_at']))
