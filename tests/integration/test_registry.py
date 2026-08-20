import pytest,json
from tests.helpers import signed_registry
from edgeai.errors import RegistryError,SignatureError
from edgeai.artifact import *
from edgeai.types import ModelMetrics
from edgeai.model import LinearModel

def test_registry_add_inspect(tmp_path):
 r,_,*_=signed_registry(tmp_path);d,m=r.inspect('1.0.0');assert d.name=='1.0.0' and m.release_sequence==1
def test_registry_immutable(tmp_path):
 r,m,*_=signed_registry(tmp_path)
 d,man=r.inspect('1.0.0')
 with pytest.raises(RegistryError):r.add(d/'model.npz',man,(d/'signature.txt').read_text().strip(),(d/'key_id.txt').read_text().strip())
def test_registry_tamper_detected(tmp_path):
 r,_,*_=signed_registry(tmp_path);d,_=r.inspect('1.0.0');(d/'model.npz').write_bytes(b'evil')
 with pytest.raises(Exception):r.inspect('1.0.0')
def test_bad_version_path_rejected(tmp_path):
 r,_,*_=signed_registry(tmp_path)
 with pytest.raises(ValueError):r.inspect('../etc/passwd')

def test_registry_rejects_symlink_entry(tmp_path):
    r, _, *_ = signed_registry(tmp_path)
    target = r.root / '1.0.0'
    alias = r.root / '1.0.1'
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(RegistryError):
        r.inspect('1.0.1')


def test_registry_rejects_path_manifest_version_mismatch(tmp_path):
    r, _, *_ = signed_registry(tmp_path)
    source = r.root / '1.0.0'
    copied = r.root / '1.0.1'
    import shutil
    shutil.copytree(source, copied)
    with pytest.raises(RegistryError):
        r.inspect('1.0.1')
