"""Scope references require exact spans, honest provenance and disjoint semantic families."""
import copy
import unittest
from rhythm_dnb.text.annotations import validate_scope_annotations
from rhythm_dnb.text.behavior import scope_consistency_report


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        for order,predicate in [('lower','我很快睡着。'),('higher','我久久睡不着。')]:
            key=order+':anchor'
            for variant,text in [('anchor',predicate),('other','别人说起失眠。'+predicate)]:
                start=text.index(predicate)
                self.rows.append({'record_id':order+':'+variant,'family_id':'scene','split':'train',
                    'category':'sleep','metric':'sleep_onset_difficulty','variant':variant,'text':text,
                    'target':{'experiencer':'self','period':'latest_sleep','evidence':[{'start':start,'end':start+len(predicate),'quote':predicate}],
                              'excluded':[] if variant=='anchor' else [{'start':0,'end':start,'quote':text[:start],'reason':'other_person'}]},
                    'expectation':{'scale_direction':'absent_to_extreme','order':order,'equivalent_to':key,'explicit_absence':order=='lower'},
                    'annotation':{'author_kind':'assistant','human_reviewed':False}})

    def test_consistency_retains_absolute_context_error(self):
        self.assertEqual(validate_scope_annotations(self.rows)['human_reviewed'],0)
        pred=[{'scores':{'sleep_onset_difficulty':v}} for v in [2.,62.,90.,94.]]
        result=scope_consistency_report(self.rows,pred)
        self.assertEqual(result['ordered_pairs'],1)
        self.assertEqual(result['mean_anchor_drift'],32.)
        self.assertEqual(result['mean_absence_score'],62.)

    def test_offsets_and_false_human_review_are_rejected(self):
        broken=copy.deepcopy(self.rows);broken[1]['target']['evidence'][0]['start']+=1
        with self.assertRaises(ValueError):validate_scope_annotations(broken)
        broken=copy.deepcopy(self.rows);broken[0]['annotation']['human_reviewed']=True
        with self.assertRaises(ValueError):validate_scope_annotations(broken)

    def test_cross_split_and_wrong_polarity_links_are_rejected(self):
        broken=copy.deepcopy(self.rows);broken[1]['split']='test'
        with self.assertRaises(ValueError):validate_scope_annotations(broken)
        broken=copy.deepcopy(self.rows);broken[1]['expectation']['equivalent_to']='higher:anchor'
        with self.assertRaises(ValueError):validate_scope_annotations(broken)


if __name__=='__main__':unittest.main()
