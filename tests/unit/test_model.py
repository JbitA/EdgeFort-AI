import numpy as np, pytest
from edgeai.model import LinearModel
from edgeai.errors import ValidationError

def test_model_logits_and_probabilities():
    m=LinearModel([[1,0],[0,1]],[0,0]); x=np.array([[2,1]],np.float32)
    assert m.predict(x).tolist()==[0]; assert np.allclose(m.probabilities(x).sum(1),1)
@pytest.mark.parametrize('W,b', [([1,2],[0]), ([[1,2],[3,4]],[0]), ([[1,np.nan]],[0])])
def test_model_rejects_bad_shapes_values(W,b):
    with pytest.raises(ValidationError): LinearModel(W,b)
def test_model_input_validation():
    m=LinearModel([[1,2]],[0])
    with pytest.raises(ValidationError): m.logits([1,2])
    with pytest.raises(ValidationError): m.logits([[1,np.inf]])
def test_roundtrip(tmp_path):
    m=LinearModel([[1,2],[3,4]],[.1,.2]); p=tmp_path/'m.npz';m.save(p);r=LinearModel.load(p)
    assert np.allclose(m.W,r.W) and np.allclose(m.b,r.b)
def test_rejects_wrong_abi(tmp_path):
    p=tmp_path/'x.npz'; np.savez_compressed(p,W=np.ones((1,2)),b=np.zeros(1),abi=np.array('bad'))
    with pytest.raises(ValidationError): LinearModel.load(p)
