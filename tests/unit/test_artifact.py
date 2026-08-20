import json,pytest
from edgeai.artifact import *
from edgeai.types import ModelMetrics
from edgeai.errors import SignatureError,ArtifactIntegrityError,ValidationError

def artifact(tmp_path):
    p=tmp_path/'m';p.write_bytes(b'abc');metrics=ModelMetrics(.95,.1,1.0,100);return p,build_manifest(p,'1.2.3',4,'x','abi',metrics,created_utc='2026-01-01T00:00:00+00:00')
def test_ed25519_sign_verify(tmp_path):
    p,m=artifact(tmp_path);priv,pub=generate_keypair();s=sign_manifest(m,priv);verify_manifest(m,s,pub);verify_artifact(p,m)
def test_tampered_model_rejected(tmp_path):
    p,m=artifact(tmp_path);p.write_bytes(b'abd')
    with pytest.raises(ArtifactIntegrityError):verify_artifact(p,m)
def test_wrong_key_rejected(tmp_path):
    p,m=artifact(tmp_path);a,b=generate_keypair();c,d=generate_keypair();s=sign_manifest(m,a)
    with pytest.raises(SignatureError):verify_manifest(m,s,d)
def test_revoked_key_rejected(tmp_path):
    p,m=artifact(tmp_path);priv,pub=generate_keypair();kid=public_key_id(pub);kr=TrustedKeyring({kid:pub},{kid});s=sign_manifest(m,priv)
    with pytest.raises(SignatureError):kr.verify(kid,m,s)
def test_manifest_validation(tmp_path):
    p=tmp_path/'m';p.write_bytes(b'a')
    with pytest.raises(Exception):build_manifest(p,'not-semver',0,'x','abi',ModelMetrics(.9,.1,1))
    d={'schema_version':99,'release_sequence':1,'version':'1.0.0','model_kind':'x','runtime_abi':'a','model_sha256':'0'*64,'model_bytes':1,'metrics':{'accuracy':.9,'ece':.1,'p95_latency_ms':1,'max_rss_mb':0},'created_utc':'x','metadata':{}}
    with pytest.raises(ValidationError):manifest_from_dict(d)

def test_manifest_rejects_malformed_digest_timestamp_and_oversized_metadata(tmp_path):
    p, manifest = artifact(tmp_path)
    data = manifest.to_dict()
    data['model_sha256'] = 'g' * 64
    with pytest.raises(ValidationError):
        manifest_from_dict(data)

    data = manifest.to_dict()
    data['created_utc'] = '2026-01-01T00:00:00'
    with pytest.raises(ValidationError):
        manifest_from_dict(data)

    with pytest.raises(ValidationError):
        build_manifest(
            p,
            '1.2.4',
            5,
            'x',
            'abi',
            ModelMetrics(.95, .1, 1.0, 100),
            metadata={'blob': 'x' * (17 * 1024)},
            created_utc='2026-01-01T00:00:00+00:00',
        )
