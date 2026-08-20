import json,pytest
from edgeai.util import canonical_json_bytes,validate_semver,atomic_write

def test_canonical_json(): assert canonical_json_bytes({'b':1,'a':2})==b'{"a":2,"b":1}'
@pytest.mark.parametrize('v',['1.0.0','0.1.2','2.0.0-alpha'])
def test_semver_valid(v):validate_semver(v)
@pytest.mark.parametrize('v',['1','v1.0.0','../x','01.0.0'])
def test_semver_invalid(v):
 with pytest.raises(ValueError):validate_semver(v)
def test_atomic_write(tmp_path):
 p=tmp_path/'x';atomic_write(p,b'abc');assert p.read_bytes()==b'abc'
