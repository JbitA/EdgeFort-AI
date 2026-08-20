import pytest
from edgeai.config import load_config,RuntimeConfig

def test_config_load(tmp_path):
 p=tmp_path/'c.yaml';p.write_text('deployment_root: d\nregistry_root: r\n');c=load_config(p);assert c.queue_capacity==64
def test_unknown_config_rejected(tmp_path):
 p=tmp_path/'c.yaml';p.write_text('deployment_root: d\nregistry_root: r\nevil: true\n')
 with pytest.raises(ValueError):load_config(p)
def test_invalid_queue_policy():
 with pytest.raises(ValueError):RuntimeConfig('d','r',queue_policy='x').validate()


def test_invalid_request_body_limit():
 with pytest.raises(ValueError):RuntimeConfig('d','r',max_request_bytes=100).validate()
