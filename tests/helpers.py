from edgeai.artifact import generate_keypair,public_key_id,build_manifest,sign_manifest,TrustedKeyring
from edgeai.registry import Registry
from edgeai.types import ModelMetrics
from edgeai.model import LinearModel

def signed_registry(tmp_path, versions=(('1.0.0',1),)):
    priv,pub=generate_keypair();kid=public_key_id(pub);reg=Registry(tmp_path/'registry',TrustedKeyring({kid:pub})); made={}
    for version,seq in versions:
        m=LinearModel([[1,0],[0,1]],[0,0]); p=tmp_path/f'{version}.npz';m.save(p)
        man=build_manifest(p,version,seq,'float32-linear',m.ABI,ModelMetrics(1,0,1,100),created_utc='2026-01-01T00:00:00+00:00')
        sig=sign_manifest(man,priv);made[version]=reg.add(p,man,sig,kid)
    return reg,made,priv,pub,kid
