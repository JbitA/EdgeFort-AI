import pytest
from edgeai.artifact import *
from edgeai.registry import Registry
from edgeai.model import LinearModel
from edgeai.types import ModelMetrics
from edgeai.errors import SignatureError

def test_key_rotation_accepts_new_and_revokes_old(tmp_path):
 old_priv,old_pub=generate_keypair();new_priv,new_pub=generate_keypair();old_id=public_key_id(old_pub);new_id=public_key_id(new_pub)
 m=LinearModel([[1,0],[0,1]],[0,0]);p=tmp_path/'m.npz';m.save(p);man=build_manifest(p,'1.0.0',1,'float32-linear',m.ABI,ModelMetrics(1,0,1,1),created_utc='2026-01-01T00:00:00+00:00')
 r=Registry(tmp_path/'r',TrustedKeyring({old_id:old_pub,new_id:new_pub},{old_id}))
 with pytest.raises(SignatureError):r.add(p,man,sign_manifest(man,old_priv),old_id)
 r.add(p,man,sign_manifest(man,new_priv),new_id);assert r.inspect('1.0.0')[1].version=='1.0.0'
