from edgeai.evaluation import evaluate

def test_reference_lifecycle(tmp_path):
 r=evaluate(tmp_path)
 assert r['float']['accuracy']>.95
 assert r['int8_weight']['accuracy']>.95
 assert r['int8_dynamic']['accuracy']>.94
 assert r['deployment']['rollback_ok'] and r['deployment']['active_after_rollback']=='1.0.1'
