import numpy as np, pytest
from edgeai.model import LinearModel
from edgeai.quantization import quantize_weights,quantize_dynamic,WeightOnlyInt8Model,DynamicInt8Model
from edgeai.errors import ValidationError

def test_weight_quantization_error_bounded():
    rng=np.random.default_rng(1); m=LinearModel(rng.normal(size=(4,16)),rng.normal(size=4)); q=quantize_weights(m)
    deq=q.qW.astype(np.float32)*q.scales[:,None]
    assert np.max(np.abs(deq-m.W)) <= np.max(q.scales)/2 + 1e-6

def test_dynamic_int8_uses_integer_accumulation_behavior():
    m=LinearModel([[.2,-.1,.4],[-.3,.8,.1]],[.01,-.02]);q=quantize_dynamic(m);x=np.array([[.1,.8,.5],[.9,.2,.3]],np.float32)
    assert q.logits(x).shape==(2,2); assert np.max(np.abs(q.logits(x)-m.logits(x)))<.03
@pytest.mark.parametrize('cls', [WeightOnlyInt8Model,DynamicInt8Model])
def test_quantized_rejects_invalid(cls):
    with pytest.raises(ValidationError): cls(np.ones((2,3),np.int8),[0,1],[0,0])

def test_quantized_roundtrip(tmp_path):
    m=LinearModel([[.2,.5]], [.1]); q=quantize_dynamic(m); p=tmp_path/'q.npz';q.save(p);r=DynamicInt8Model.load(p)
    assert np.array_equal(q.qW,r.qW); assert np.allclose(q.scales,r.scales)

def test_dynamic_int8_rejects_dimension_that_can_overflow_int32():
    qweights=np.full((1,133145),127,dtype=np.int8)
    with pytest.raises(ValidationError, match='safe int32 accumulation'):
        DynamicInt8Model(qweights,[1.0],[0.0])


def test_quantized_rejects_nonfinite_bias():
    with pytest.raises(ValidationError):
        WeightOnlyInt8Model(np.ones((1,2),np.int8),[1.0],[float('nan')])
