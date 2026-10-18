import unittest
from tablesight.evaluate import evaluate,count_vector
from tablesight.sampling import weighted_manifest


class CountingTests(unittest.TestCase):
    def test_counting_by_table_and_repeats(self):
        gt=[dict(table=1,goal_id='CA001',count=2)]
        rows=[dict(scene_id='a',tables=[1],ground_truth=gt,prediction=gt,background_total=2,background_counted=0),
              dict(scene_id='a',tables=[1],ground_truth=gt,prediction=[dict(table=1,goal_id='CA001',count=1)],background_total=2,background_counted=1)]
        r=evaluate(rows)
        self.assertEqual(r['table_exact_match_rate'],.5)
        self.assertEqual(r['mean_absolute_table_count_error'],.5)
        self.assertEqual(r['pairwise_repeat_consistency'],0)
        self.assertEqual(r['background_false_inclusion_rate'],.25)

    def test_same_category_on_different_tables_is_not_merged(self):
        g=[dict(table=1,goal_id='CA001',count=1),dict(table=2,goal_id='CA001',count=2)]
        r=evaluate([dict(ground_truth=g,prediction=g)])
        self.assertEqual(r['table_count'],2);self.assertEqual(r['table_exact_match_rate'],1)

    def test_empty_table_and_zero_rows(self):
        r=evaluate([dict(tables=[1],ground_truth=[],prediction=[dict(table=1,goal_id='CA001',count=0)])])
        self.assertEqual(r['table_exact_match_rate'],1)
        self.assertIsNone(r['relative_absolute_count_error'])

    def test_failed_prediction_still_counts(self):
        r=evaluate([dict(status='FAILED',ground_truth=[dict(table=1,goal_id='CA001',count=2)])])
        self.assertEqual(r['table_exact_match_rate'],0);self.assertEqual(r['mean_absolute_table_count_error'],2)

    def test_invalid_counts_and_duplicates_are_rejected(self):
        for value in (-1,True,1.5):
            with self.assertRaises(ValueError):count_vector([dict(table=1,goal_id='CA001',count=value)])
        row=dict(table=1,goal_id='CA001',count=1)
        with self.assertRaises(ValueError):count_vector([row,row])

    def test_different_pipelines_are_not_repeat_trials(self):
        g=[dict(table=1,goal_id='CA001',count=1)]
        r=evaluate([dict(scene_id='same',pipeline=p,ground_truth=g,prediction=g) for p in ('baseline','full')])
        self.assertEqual(r['repeated_scene_pair_count'],0)


class SamplingTests(unittest.TestCase):
    def row(self,sid='1',split='train',weight='3',group='g1',sha='a'*64):
        return dict(sample_id=sid,path=sid+'.jpg',split=split,weight=weight,scene_group=group,source_sha=sha,priority='hard')

    def test_weighted_entries_do_not_duplicate_validation(self):
        a=self.row();b=self.row('2','val','1','g2','b'*64)
        lines,report=weighted_manifest([a,b])
        self.assertEqual(lines,['1.jpg']*3);self.assertEqual(report['unique_training_images'],1)

    def test_scene_leakage_is_rejected(self):
        with self.assertRaises(ValueError):weighted_manifest([self.row(),self.row('2','val','1','g1','b'*64)])

    def test_physical_source_leakage_is_rejected(self):
        with self.assertRaises(ValueError):weighted_manifest([self.row(),self.row('2','val','1','g2','a'*64)])

    def test_validation_reweighting_is_rejected(self):
        with self.assertRaises(ValueError):weighted_manifest([self.row(split='val',weight='3')])

    def test_duplicate_paths_are_rejected(self):
        a=self.row();b=self.row('2',group='g2',sha='b'*64);b['path']=a['path']
        with self.assertRaises(ValueError):weighted_manifest([a,b])


if __name__=='__main__':unittest.main()
