import pytest
from edgeai.evaluation import evaluate

def test_explicit_evaluation_workspace_is_rerunnable(tmp_path):
    p=tmp_path/'work';a=evaluate(p);b=evaluate(p);assert a['float']['accuracy']==b['float']['accuracy']

def test_evaluation_refuses_unowned_nonempty_directory(tmp_path):
    p=tmp_path/'work';p.mkdir();(p/'user.txt').write_text('keep')
    with pytest.raises(ValueError):evaluate(p)
    assert (p/'user.txt').read_text()=='keep'
